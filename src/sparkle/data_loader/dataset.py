import json
import logging
import os
from pathlib import Path
from tqdm import tqdm
from s3fs import S3FileSystem
from torch.utils.data import Dataset

from sparkle.configs.config import Config
from sparkle.data_loader.encoder.positional_encodings import field_pos, header_pos

logger = Config.init_logger()


class PacketSequenceDataset(Dataset):
    def __init__(self, config: Config, manifest_path, tokenizer, chunk_size):
        logger.info(f"Initializing PacketSequenceDataset with manifest: {manifest_path}")

        self.tokenizer = tokenizer
        self.config = config
        self.manifest_path = manifest_path

        # Log the start of manifest loading
        self.files = self._load_manifest()
        logger.info(f"Loaded {len(self.files)} entries from manifest.")

        self.fs = S3FileSystem()

        # packets per sample returned in the dataset
        self.chunk_size = chunk_size
        self.total_chunks = []

        self.cache = {
            "packet": [],
            "header": [],
            "field": [],
            "direction": []
        }

        for file in self.files:
            num_lines = len(self._read_file(file["packet"]).splitlines())
            num_chunks = (num_lines + self.chunk_size - 1) // self.chunk_size
            self.total_chunks.append(num_chunks)

        self.total_len = sum(self.total_chunks)
        logger.info(f"Dataset initialization complete. Total samples (chunks): {self.total_len}")

    def _load_manifest(self):
        try:
            with open(self.manifest_path, 'r') as f:
                data = json.load(f)

            files = [
                {k: m[k] for k in ("packet", "header", "field", "direction")}
                for m in data
            ]
            return files
        except FileNotFoundError:
            logger.error(f"Manifest file not found at {self.manifest_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Manifest file at {self.manifest_path} is not valid JSON")
            raise

    def _read_s3_file(self, s3_path):
        # Change to logger.info if you want to see every single file read
        with self.fs.open(s3_path, 'r') as f:
            return f.read()


    def _read_file(self, path):
        path = Path(path)
        return path.read_text(encoding="utf-8")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        cumulative_chunks = 0
        file_idx = 0
        line_idx = 0

        for i, num_chunks in enumerate(self.total_chunks):
            if cumulative_chunks + num_chunks > idx:
                line_idx = idx - cumulative_chunks
                break
            cumulative_chunks += num_chunks
        else:
            raise IndexError("Index out of range")


        entry = self.files[file_idx]

        packet_path, header_path, field_path, direction_path = (
            entry["packet"],
            entry["header"],
            entry["field"],
            entry["direction"]
        )

        try:
            hex_dumps = self._read_file(packet_path).splitlines()

            # path = Path(os.path.join(self.config.logging_dir, "output.txt"))
            # path.parent.mkdir(parents=True, exist_ok=True)
            #
            # print("Writing to file")
            # with open(Path(path), "w", encoding="utf-8") as f:
            #     f.write(hex_dumps)
            #
            #
            # for i, line in enumerate(hex_dumps):
            #     test = line.strip()
            #     try:
            #         bytes.fromhex(test)
            #     except Exception as e
            #         logger.error(f"Full packet at file_idx={file_idx}: {hex_dumps}")
            #         raise

            padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(hex_dumps)

            # Slice out the chunk from token_ids
            chunk_start = line_idx * self.chunk_size
            chunk_end = min((line_idx + 1) * self.chunk_size, token_ids.size(0))
            chunk = token_ids[chunk_start:chunk_end]

            field_position = field_pos(field_path, chunk_start, chunk_end)
            header_position = header_pos(header_path, chunk_start, chunk_end)

            return chunk, field_position, header_position, entry

        except Exception as e:
            logger.error(f"Error processing item at idx {idx} (file_idx {file_idx}): {e}")
            raise