import torch
from torch.utils.data import DataLoader, Dataset
from model.embedding_new import FlowEmbedding
import os
from encoder.positional_encodings import field_pos, header_pos
from tokenizer.tokenizer import Tokenizer
from config import Config

class PacketSequenceDataset(Dataset):
    def __init__(self, packet_folder, field_folder, header_folder, tokenizer, chunk_size):
        self.packet_seq = sorted(
            [os.path.join(packet_folder, file) for file in os.listdir(packet_folder) if file.endswith('.txt')])
        self.field_pos_files = sorted(
            [os.path.join(field_folder, file) for file in os.listdir(field_folder) if file.endswith('.txt')])
        self.header_pos_files = sorted(
            [os.path.join(header_folder, file) for file in os.listdir(header_folder) if file.endswith('.txt')])
        self.tokenizer = tokenizer

        # packets per sample returned in the dataset
        self.chunk_size = chunk_size

        self.total_chunks = []

        for file in self.packet_seq:
            num_lines = len(open(file, 'r').readlines())
            num_chunks = (num_lines + self.chunk_size - 1) // self.chunk_size
            self.total_chunks.append(num_chunks)
        self.total_len = sum(self.total_chunks)

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

        packet_seq = self.packet_seq[file_idx]
        field_pos_file = self.field_pos_files[file_idx]
        header_pos_file = self.header_pos_files[file_idx]

        with open(packet_seq, 'r', encoding='utf-8') as f:
            hex_dumps = f.readlines()

        padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(hex_dumps)

        # Slice out the chunk from token_ids
        chunk_start = line_idx * self.chunk_size
        chunk_end = min((line_idx + 1) * self.chunk_size, token_ids.size(0))
        chunk = token_ids[chunk_start:chunk_end]

        field_position = field_pos(field_pos_file, chunk_start, chunk_end)
        header_position = header_pos(header_pos_file, chunk_start, chunk_end)

        return chunk, field_position, header_position, packet_seq


class DataModule:
    def __init__(self):
        self.config = Config()
        self.tokenizer = self._load_tokenizer()
        self.dataset = self._init_dataset()
        self.train_loader = self._init_dataloader()

    def _load_tokenizer(self):
        return Tokenizer(vocab_file=self.config.tokenizer_path)

    def _init_dataset(self):
        return PacketSequenceDataset(packet_folder, header_folder, fields_folder, self.tokenizer, chunk_size=32)

    def _init_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.chunk_size, shuffle=False, num_workers=self.config.num_workers)

    def _get_loader(self):
        return self.train_loader


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = DataModule()

    tokenizer = data_module.tokenizer

    bucket = 'netml-s3-bucket'
    packet_folder = 'Working_folder/input_aws/split/packets'
    header_folder = 'Working_folder/input_aws/split/headers'
    fields_folder = 'Working_folder/input_aws/split/fields'
    direction_folder = 'Working_folder/input_aws/split/direction'

    dataset = PacketSequenceDataset(packet_folder, header_folder, fields_folder, tokenizer, chunk_size=32)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)







