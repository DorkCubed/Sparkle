import torch
from torch.utils.data import DataLoader, Dataset
from model.embedding_new import FlowEmbedding
import os
from encoder.positional_encodings import field_pos, header_pos
from tokenizer.tokenizer import Tokenizer
import torch.nn as nn
import boto3
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PacketSequenceDataset(Dataset):
    def __init__(self, packet_folder, field_folder, header_folder, tokenizer, batch_size):
        self.packet_seq = sorted(
            [os.path.join(packet_folder, file) for file in os.listdir(packet_folder) if file.endswith('.txt')])
        self.field_pos_files = sorted(
            [os.path.join(field_folder, file) for file in os.listdir(field_folder) if file.endswith('.txt')])
        self.header_pos_files = sorted(
            [os.path.join(header_folder, file) for file in os.listdir(header_folder) if file.endswith('.txt')])
        self.tokenizer = tokenizer

        # packets per sample returned in the dataset
        self.batch_size = batch_size

        self.total_chunks = []

        for file in self.packet_seq:
            num_lines = len(open(file, 'r').readlines())
            num_chunks = (num_lines + self.batch_size - 1) // self.batch_size
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
        chunk_start = line_idx * self.batch_size
        chunk_end = min((line_idx + 1) * self.batch_size, token_ids.size(0))
        chunk = token_ids[chunk_start:chunk_end]

        field_position = field_pos(field_pos_file, chunk_start, chunk_end)
        header_position = header_pos(header_pos_file, chunk_start, chunk_end)

        return chunk, field_position, header_position, packet_seq


class FlowLevelTrainer:
    def __init__(self, flow_embedding):
        self.flow_embedding = flow_embedding
        flow_embedding = FlowEmbedding(embed_dim, max_flow_length, dropout, vocab).to(device)

    def process_encodings(encodings, packet_number, direction_folder):
        """ Process accumulated encodings and load corresponding direction data. """
        final_packet_encodings = torch.cat(encodings, dim=0).to(device)
        print("Final concatenated shape:", final_packet_encodings.shape)

        # Load the corresponding direction data from the text file
        direction_file_name = f'direction_{packet_number}.txt'
        direction_file_path = os.path.join(direction_folder, direction_file_name)

        with open(direction_file_path, 'r') as file:
            direction_data = [int(line.strip()) for line in file.readlines()]

        # Convert direction data to a tensor
        direction_tensor = torch.tensor(direction_data, device=device)
        print(final_packet_encodings.device, direction_tensor.device)
        # Call FlowEmbedding with the accumulated packet encodings and direction data
        flow_embeddings, pad_indices = flow_embedding(final_packet_encodings, direction_tensor)
        print("Flow embeddings computed for packet:", packet_number)
        print("flow embeddings: ", flow_embeddings.shape)
        print("-------------------------------------------------------------")

        # Compute flow encoding and MPM loss
        flow_encoding, mpm_loss = flow_encoder(flow_embeddings, pad_indices)

        # Ensure mpm_loss is a tensor and on the same device
        mpm_loss_tensor = mpm_loss[0].to(device)
        print(mpm_loss_tensor)
        return mpm_loss_tensor



if __name__ == '__main__':
    bucket = 'netml-s3-bucket'
    packet_folder = 'Working_folder/input_aws/split/packets'
    header_folder = 'Working_folder/input_aws/split/headers'
    fields_folder = 'Working_folder/input_aws/split/fields'
    direction_folder = 'Working_folder/input_aws/split/direction'

    tokenizer = Tokenizer(vocab_file=os.path.join("tokenizer", "vocab.txt"))
    dataset = PacketSequenceDataset(packet_folder, header_folder, fields_folder, tokenizer, batch_size=32)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)







