import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Input_Tokenizer import Tokenizer

# custom_vocab_path = r"C:\Users\Kavish\OneDrive\Desktop\SPARKLE\Custom-Vocab.txt"
# tokenizer = Tokenizer(vocab_file=custom_vocab_path)


class PacketSequenceDataset(Dataset):
    def __init__(self, packet_seq_dir, field_pos_dir, header_pos_dir, direction_dir, tokenizer):
        self.packet_seq_dir = packet_seq_dir
        self.field_pos_dir = field_pos_dir
        self.header_pos_dir = header_pos_dir
        self.direction_dir = direction_dir
        self.tokenizer = tokenizer
        self.packet_sequences = []
        self.field_pos_sequences = []
        self.header_pos_sequences = []
        self.direction_sequences = []

        # Preprocess data
        packet_seq_files = [os.path.join(packet_seq_dir, file) for file in os.listdir(
            packet_seq_dir) if file.endswith('.txt')]
        field_pos_files = [os.path.join(field_pos_dir, file) for file in os.listdir(
            field_pos_dir) if file.endswith('.txt')]
        header_pos_files = [os.path.join(header_pos_dir, file) for file in os.listdir(
            header_pos_dir) if file.endswith('.txt')]
        direction_file = [os.path.join(direction_dir, file) for file in os.listdir(
            direction_dir) if file.endswith('.txt')]

        for packet_seq_file, field_pos_file, header_pos_file, direction_file in zip(packet_seq_files, field_pos_files, header_pos_files, direction_file):
            with open(packet_seq_file, 'r', encoding='utf-8') as f:
                hex_dumps = f.readlines()
                padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(
                    hex_dumps)
                self.packet_sequences.append(token_ids)

            '''
            with open(packet_seq_file, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = self.tokenizer.encode_packet(
                        line.strip(), add_special_tokens=True, truncation=True, padding='max_length')
                    self.packet_sequences.append(tokens)
                '''

            with open(field_pos_file, 'r', encoding='utf-8') as f:
                for line in f:
                    field_pos = line.strip().split()
                    field_pos = [int(pos) for pos in field_pos]
                    self.field_pos_sequences.append(field_pos)

            with open(header_pos_file, 'r', encoding='utf-8') as f:
                for line in f:
                    header_pos = line.strip().split()
                    header_pos = [int(pos) for pos in header_pos]
                    self.header_pos_sequences.append(header_pos)

            with open(direction_file, 'r', encoding='utf-8') as f:
                for line in f:
                    direction = line.strip().split()
                    direction = [int(pos) for pos in direction]
                    self.direction_sequences.append(direction)

    def __len__(self):
        return len(self.packet_sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.packet_sequences[idx]),
            torch.tensor(self.field_pos_sequences[idx]),
            torch.tensor(self.header_pos_sequences[idx]),
            torch.tensor(self.direction_sequences[idx])
        )


# # Create the dataset and data loader
# dataset = PacketSequenceDataset(
#     packet_seq_dir=r"C:\Users\Kavish\OneDrive\Desktop\SPARKLE\split\packet", field_pos_dir=r"C:\Users\Kavish\OneDrive\Desktop\SPARKLE\split\field", header_pos_dir=r"C:\Users\Kavish\OneDrive\Desktop\SPARKLE\split\header", direction_dir=r"C:\Users\Kavish\OneDrive\Desktop\SPARKLE\split\direction", tokenizer=tokenizer)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# packet_seq, field_seq, header_seq, direction = dataset.__getitem__(0)
# print(field_seq.shape)
