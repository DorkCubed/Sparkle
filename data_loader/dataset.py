from torch.utils.data import Dataset
from .encoder.positional_encodings import field_pos, header_pos
from configs.config import Config
from scripts.s3_utils import S3DataFetcher

class PacketSequenceDataset(Dataset):
    def __init__(self, config: Config, s3_fetcher: S3DataFetcher, files, tokenizer, chunk_size):
        self.tokenizer = tokenizer
        self.config = config
        self.files = files
        self.s3_fetcher = s3_fetcher

        # packets per sample returned in the dataset
        self.chunk_size = chunk_size


        self.total_chunks = []
        for packet_path, _, _, _ in self.files:
            num_lines = len(self.s3_fetcher.read_files(packet_path).splitlines())
            num_chunks = (num_lines + self.chunk_size - 1) // self.chunk_size
            self.total_chunks.append(num_chunks)

        self.total_len = sum(self.total_chunks)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        """
        Return a chunk of packets from the dataset along with their corresponding
        field positions and header positions.

        Args:
            idx (int): The index of the chunk to return.

        Returns:
            tuple: a tuple containing:
                - chunk (torch.Tensor): a tensor of shape (chunk_size, max_length) containing
                  the tokenized packets.
                - field_position (torch.Tensor): a tensor of shape (chunk_size, max_length)
                  containing the field positions.
                - header_position (torch.Tensor): a tensor of shape (chunk_size, max_length)
                  containing the header positions.
                - packet_path (str): the path to the packet file that the chunk was read from.
        """
        cumulative_chunks = 0
        for file_idx, num_chunks in enumerate(self.total_chunks):
            if cumulative_chunks + num_chunks > idx:
                line_idx = idx - cumulative_chunks
                break
            cumulative_chunks += num_chunks
        else:
            raise IndexError("Index out of range")

        packet_path, header_path, field_path, _ = self.files[file_idx]

        hex_dumps = self.s3_fetcher.read_files(packet_path).splitlines()
        padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(hex_dumps)

        # Slice out the chunk from token_ids
        chunk_start = line_idx * self.chunk_size
        chunk_end = min((line_idx + 1) * self.chunk_size, token_ids.size(0))
        chunk = token_ids[chunk_start:chunk_end]

        field_position = field_pos(field_path, chunk_start, chunk_end)
        header_position = header_pos(header_path, chunk_start, chunk_end)

        return chunk, field_position, header_position, packet_path
