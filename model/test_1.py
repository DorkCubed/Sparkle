import torch
import torch.nn as nn
import torch.optim as optim
from Input_Tokenizer import Tokenizer
from embedding_new import PacketEmbedding, FlowEmbedding
from packet_encoder import PacketLevelEncoder
from flow_encoder import FlowLevelEncoder
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from field_header_pos_encoding import field_pos, header_pos

# Hyperparameters
vocab_size = 30000
embed_dim = 768
num_heads = 12
num_layers = 6
dropout = 0.1
max_flow_length = 510
mask_prob = 0.15
num_epochs = 1
max_len=512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# loading vocabulary
vocab = {}
custom_vocab_path = r"/home/satvik/spark/data/vocab_1.txt"
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


# Initialize flow embedding and encoding module
flow_embedding = FlowEmbedding(embed_dim, max_flow_length, dropout, vocab).to(device)
flow_encoder = FlowLevelEncoder(
    embed_dim, num_layers, num_heads, dropout, vocab, max_flow_length, mask_prob)


# training criterion initialize
# optimizer = optim.Adam(list(packet_embedding.parameters(
# )) + list(packet_encoder.parameters()) + list(flow_encoder.parameters()), lr=0.001)
# criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(list(packet_embedding.parameters(
)) + list(packet_encoder.parameters()) , lr=0.001)
criterion = nn.CrossEntropyLoss()

# data loader


class PacketSequenceDataset(Dataset):
    def __init__(self, packet_seq_dir, field_pos_dir, header_pos_dir, tokenizer, batch_size_1):
        self.packet_seq_dir = packet_seq_dir
        self.field_pos_dir = field_pos_dir
        self.header_pos_dir = header_pos_dir
        self.tokenizer = tokenizer
        self.batch_size_1 = batch_size_1
        self.samples = []

        # Load file paths
        packet_seq_files = [os.path.join(packet_seq_dir, file) for file in os.listdir(packet_seq_dir) if file.endswith('.txt')]
        field_pos_files = [os.path.join(field_pos_dir, file) for file in os.listdir(field_pos_dir) if file.endswith('.txt')]
        header_pos_files = [os.path.join(header_pos_dir, file) for file in os.listdir(header_pos_dir) if file.endswith('.txt')]

        # Process each file
        for packet_seq_file, field_pos_file, header_pos_file in zip(packet_seq_files, field_pos_files, header_pos_files):
            with open(packet_seq_file, 'r', encoding='utf-8') as f:
                hex_dumps = f.readlines()

            # Tokenize and pad all lines in the file
            padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(hex_dumps)

            # Split the tokenized data into chunks of size `batch_size_1`
            for i in range(0, token_ids.size(0), self.batch_size_1):
                chunk = token_ids[i:i + self.batch_size_1]
                if chunk.size(0) < self.batch_size_1:
                    continue  # Skip chunks that are smaller than `batch_size_1`

                # Load and prepare field and header position encodings
                field_posn = field_pos(field_pos_file).to(device).long()
                header_posn = header_pos(header_pos_file).to(device).long()

                # Ensure that field_posn and header_posn are split in the same way
                field_pos_chunk = field_posn[i:i + self.batch_size_1]
                header_pos_chunk = header_posn[i:i + self.batch_size_1]

                # Store the chunks as separate samples
                self.samples.append((chunk.to(device), field_pos_chunk, header_pos_chunk))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Create the dataset and data loader
dataset = PacketSequenceDataset(
    packet_seq_dir='/home/satvik/spark/data/packet', field_pos_dir='/home/satvik/spark/data/field', header_pos_dir='/home/satvik/spark/data/header', tokenizer=tokenizer, batch_size_1=2)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# for packets, fields, headers in train_loader:
#     print(packets.shape)
#     print(fields.shape)
#     print(headers.shape)
#     break

# Training loop
for epoch in range(1):
    print("run ", epoch)
    i=0
    all_packet_encodings = []
    for packet_sequences, field_pos, header_pos in train_loader:
        # Forward pass
        i+=1 
        print("i ", i)
        print(packet_sequences.shape)
        # break
        print("1")
        field_pos = field_pos.squeeze(0)
        header_pos = header_pos.squeeze(0)
        # print(field_pos.shape)
        # print(header_pos.shape)
        # print("packet seq :", packet_sequences.shape)
        # print(packet_sequences)
        print("packet seq shape: ", type(packet_sequences))
        print(field_pos.shape)
        packet_embeddings = packet_embedding(
            token_ids=packet_sequences, field_pos=field_pos, header_pos=header_pos)
        # print("pe shape: ", packet_embeddings.unsqueeze(0)[0][:][0])
        print("pe shape: ", packet_embeddings.shape)
        mlm_loss, sfbo_loss, encoded_packets, masked_packets, encoded_packets_mean = packet_encoder(packet_sequences, field_pos=field_pos, header_pos=header_pos)
        # flow_encodings, mpm_losses = flow_encoder(flow_sequences)
        # if (i%510 == 0): 
        print("final shape packet: ", encoded_packets_mean.shape)
        enc_p = []
        enc_p.append(encoded_packets_mean)
        all_packet_encodings.append(encoded_packets_mean)
        # print("length :", len(enc_p[0]))
        print("mlm loss: ", mlm_loss.item())
        print("sfbo loss: ", sfbo_loss.item())
        # print(masked_packets)
        # print("sfbo_loss: ", sfbo_loss.item())
        # total_loss = mlm_loss + sfbo_loss
        # print("total loss: ", total_loss.item())
        # # Compute total loss
        # total_loss = mlm_loss + sfbo_lossD
        ''' + sum(mpm_losses)'''
        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()

        # # Backward pass and optimization
        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()
    final_packet_encodings = torch.cat(all_packet_encodings, dim=0).to(device)
    print("Final concatenated shape: ", final_packet_encodings.shape)
    # flow_embeddings = flow_embedding(final_packet_encodings, final_packet_encodings)

    # Load the corresponding direction data
    direction = load_direction_data(direction_file).to(device)
    print("Direction tensor shape: ", direction.shape)

    # Forward pass through FlowEmbedding
    flow_embeddings = flow_embedding(encoded_packets_mean, direction)
    print("Flow embeddings shape: ", flow_embeddings.shape)