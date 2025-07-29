import torch
import torch.nn as nn
import torch.optim as optim
from Input_Tokenizer import Tokenizer
from embedding_new import PacketEmbedding, FlowEmbedding
from packet_encoder import PacketLevelEncoder
from flow_encoder import FlowLevelEncoder
import os
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
vocab_size = 30000
embed_dim = 768
num_heads = 12
num_layers = 6
dropout = 0.1
max_flow_length = 510
mask_prob = 0.15
num_epochs = 10
max_len = 512 #change 1 

# loading vocabulary
vocab = {}
custom_vocab_path = r"/home/satvik/spark/spark/vocab_1.txt"
with open(r"/home/satvik/spark/spark/vocab_1.txt", 'r', encoding='utf-8') as f:
    for line in f:
        token, token_id = line.strip().split('\t')
        vocab[token] = int(token_id)

# initialise the tokenizer
tokenizer = Tokenizer(vocab_file=custom_vocab_path)

# Initialize packet embedding and encoding modules
packet_embedding = PacketEmbedding(
    vocab_size, max_len=512, embed_dim=embed_dim, dropout=dropout)
packet_encoder = PacketLevelEncoder(vocab_size, embed_dim, max_len, num_heads, num_layers, dropout)


# Initialize flow embedding and encoding module
# flow_embedding = FlowEmbedding(embed_dim, max_flow_length, dropout, vocab)
# flow_encoder = FlowLevelEncoder(
#     embed_dim, num_layers, num_heads, dropout, max_flow_length, mask_prob)


# training criterion initialize
optimizer = optim.Adam(list(packet_embedding.parameters(
)) + list(packet_encoder.parameters()) + list(flow_encoder.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

# data loader


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
                    field_pos = field_pos(line)
                    self.field_pos_sequences.append(field_pos)

            with open(header_pos_file, 'r', encoding='utf-8') as f:
                for line in f:
                    header_pos = header_pos(line)
                    self.header_pos_sequences.append(header_pos)

            with open(direction_file, 'r', encoding='utf-8') as f:
                for line in f:
                    direction = direction(line)
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


# Create the dataset and data loader
dataset = PacketSequenceDataset(
    packet_seq_dir='path/to/data/dir', field_pos_dir='path/to/field_pos/dir', header_pos_dir='path/to/header_pos/dir', tokenizer=tokenizer)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Training loop
for epoch in range(num_epochs):
    for packet_sequences, field_pos, header_pos, direction in train_loader:
        # Forward pass
        packet_embeddings = packet_embedding.forward(
            token_ids=packet_sequences, field_pos=field_pos, header_pos=header_pos)
        mlm_loss, sfbo_loss, packet_encodings = packet_encoder(
            packet_embeddings)

        # forwrd pass to flow encoder
        # adds clsf token at beginning
        cls_packet_embedding = tokenizer.encode_flow(packet_encodings)
        flow_embeddings = flow_embedding.forward(
            cls_packet_embedding, direction)
        flow_encodings, mpm_losses = flow_encoder(flow_embeddings)

        # Compute total loss
        total_loss = mlm_loss + sfbo_loss + sum(mpm_losses)

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
