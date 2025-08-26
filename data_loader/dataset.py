from s3fs import S3FileSystem
import json
from torch.utils.data import Dataset
from .encoder.positional_encodings import field_pos, header_pos
from configs.config import Config
from scripts.s3_utils import S3DataFetcher

class PacketSequenceDataset(Dataset):
    def __init__(self, config: Config, manifest_path, tokenizer, chunk_size):
        self.tokenizer = tokenizer
        self.config = config
        self.manifest_path = manifest_path
        self.files = self._load_manifest()
        self.fs = S3FileSystem()

        # packets per sample returned in the dataset
        self.chunk_size = chunk_size
        self.total_chunks = []

        for packet_path, _, _, _ in self.files:
            num_lines = len(self._read_file(packet_path).splitlines())
            num_chunks = (num_lines + self.chunk_size - 1) // self.chunk_size
            self.total_chunks.append(num_chunks)

        self.total_len = sum(self.total_chunks)

    def _load_manifest(self):
        with open(self.manifest_path, 'r') as f:
            data = json.load(f)

        files = [(m["packet"], m["header"], m["field"], m["direction"]) for m in data]
        return files

    def _read_file(self, s3_path):
        with self.fs.open(s3_path, "r") as f:
            return f.read()


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        cumulative_chunks = 0
        for file_idx, num_chunks in enumerate(self.total_chunks):
            if cumulative_chunks + num_chunks > idx:
                line_idx = idx - cumulative_chunks
                break
            cumulative_chunks += num_chunks
        else:
            raise IndexError("Index out of range")

        packet_path, header_path, field_path, _ = self.files[file_idx]

        hex_dumps = self._read_file(packet_path).splitlines()
        padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(hex_dumps)

        # Slice out the chunk from token_ids
        chunk_start = line_idx * self.chunk_size
        chunk_end = min((line_idx + 1) * self.chunk_size, token_ids.size(0))
        chunk = token_ids[chunk_start:chunk_end]

        field_position = field_pos(field_path, chunk_start, chunk_end)
        header_position = header_pos(header_path, chunk_start, chunk_end)

        return chunk, field_position, header_position, packet_path
