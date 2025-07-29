import torch
import torch.nn as nn
import torch.optim as optim
from Input_Tokenizer import Tokenizer
from embedding_new import PacketEmbedding, FlowEmbedding
from packet_encoder import PacketLevelEncoder, apply_mlm_sfbo_masking
# from flow_encoder import FlowLevelEncoder
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from field_header_pos_encoding import field_pos, header_pos# Hyperparameters
vocab_size = 30000
embed_dim = 768
num_heads = 12
num_layers = 6
dropout = 0.1
max_flow_length = 510
mask_prob = 0.15
num_epochs = 10
max_len=512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# loading vocabulary
vocab = {}
custom_vocab_path = r"/home/satvik/spark/spark/vocab_1.txt"
with open(custom_vocab_path, 'r', encoding='utf-8') as f:
    for line in f:
        token, token_id = line.strip().split('\t')
        vocab[token] = int(token_id)

# initialise the tokenizer
tokenizer = Tokenizer(vocab_file=custom_vocab_path)

# Initialize packet embedding and encoding modules
packet_embedding = PacketEmbedding(
    vocab_size, max_len=512, embed_dim=embed_dim, dropout=dropout).to(device)
packet_encoder = PacketLevelEncoder(   
    vocab_size, embed_dim, max_len, num_heads, num_layers, dropout).to(device)

optimizer = optim.Adam(list(packet_embedding.parameters(
)) + list(packet_encoder.parameters()) , lr=0.001)
criterion = nn.CrossEntropyLoss()

# data loader


class PacketSequenceDataset(Dataset):
    def __init__(self, packet_seq_dir, field_pos_dir, header_pos_dir, tokenizer):
        self.packet_seq_dir = packet_seq_dir
        self.field_pos_dir = field_pos_dir
        self.header_pos_dir = header_pos_dir
        self.tokenizer = tokenizer
        self.packet_sequences = []
        self.field_pos_sequences = []
        self.header_pos_sequences = []

        # Preprocess data
        packet_seq_files = [os.path.join(packet_seq_dir, file) for file in os.listdir(
            packet_seq_dir) if file.endswith('.txt')]
        field_pos_files = [os.path.join(field_pos_dir, file) for file in os.listdir(
            field_pos_dir) if file.endswith('.txt')]
        header_pos_files = [os.path.join(header_pos_dir, file) for file in os.listdir(
            header_pos_dir) if file.endswith('.txt')]

        for packet_seq_file, field_pos_file, header_pos_file in zip(packet_seq_files, field_pos_files, header_pos_files):
            with open(packet_seq_file, 'r', encoding='utf-8') as f:
                hex_dumps = f.readlines()
                padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(
                    hex_dumps)
                self.packet_sequences.append(token_ids.to(device))
                print("td :", token_ids.shape)

            '''
            with open(packet_seq_file, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = self.tokenizer.encode_packet(
                        line.strip(), add_special_tokens=True, truncation=True, padding='max_length')
                    self.packet_sequences.append(tokens)
                '''
            field_posn = field_pos(field_pos_file).to(device).long()
            print("fp: ", field_posn.shape) 
            self.field_pos_sequences.append(field_posn)

            header_posn = header_pos(header_pos_file).to(device).long()
            print("hp :", header_posn.shape)
            self.header_pos_sequences.append(header_posn)

    def __len__(self):
        return len(self.packet_sequences)


    def __getitem__(self, idx):
        return (
        self.packet_sequences[idx],
        self.field_pos_sequences[idx],
        self.header_pos_sequences[idx]
        )
    
# Create the dataset and data loader
dataset = PacketSequenceDataset(
    packet_seq_dir=r'/home/satvik/spark/spark/packets', field_pos_dir='/home/satvik/spark/spark/fields', header_pos_dir='/home/satvik/spark/spark/headers', tokenizer=tokenizer)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    print("epoch 1")
    for packet_sequences, field_pos, header_pos in train_loader:
        
        # Forward pass
        print("1")
        field_pos = field_pos.squeeze(0)
        header_pos = header_pos.squeeze(0)
        print(field_pos.shape)
        print(header_pos.shape)
        # print("packet seq :", packet_sequences.shape)
        # print(packet_sequences)
        print("packet seq shape: ", packet_sequences.shape)
        
        # Generate masked tokens
        masked_packets, span_masks = apply_mlm_sfbo_masking(packet_sequences)
        print("masked packets shape: ", masked_packets.shape)
        print("span masks: ", span_masks)

        # moving to same device
        packet_sequences = packet_sequences.to(device)
        fiels_pos = field_pos.to(device)
        header_pos = header_pos.to(device)
        masked_packets = masked_packets.to(device)
        
        # compute packet embeddings
        packet_embeddings = packet_embedding(
            token_ids=masked_packets, field_pos=field_pos, header_pos=header_pos)
        print("pe shape: ", packet_embeddings.shape)
        
        # calucate loss
        mlm_loss = packet_encoder.compute_mlm_loss(packet_embeddings, masked_packets)
        sfbo_loss = packet_encoder.compute_sfbo_loss(packet_embeddings, span_masks, packet_sequences)
        # flow_encodings, mpm_losses = flow_encoder(flow_sequences)
        print("mlm loss: ", mlm_loss.item())
        print("sfbo_loss: ", sfbo_loss.item())
        total_loss = mlm_loss  + sfbo_loss
        print("total loss: ", total_loss.item())
        
        # Compute total loss
        # total_loss = mlm_loss + sfbo_lossD
        ''' + sum(mpm_losses)'''
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()