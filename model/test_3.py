import torch
import torch.nn as nn
import torch.optim as optim
from Input_Tokenizer import Tokenizer
from embedding_new import PacketEmbedding, FlowEmbedding
from packet_encoder import PacketLevelEncoder
from flow_encoder import FlowLevelEncoder
import os
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from field_header_pos_encoding import field_pos, header_pos

# Hyperparameters
vocab_size = 262
embed_dim = 256 #768
num_heads = 8 #12
num_layers = 4 #6
dropout = 0.2 #0.1
max_flow_length = 384 #512
mask_prob = 0.15
num_epochs = 1
max_len = 578 #512
batch_size_1 = 2

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
    vocab_size, max_len=578 , embed_dim=embed_dim, dropout=dropout).to(device)
packet_encoder = PacketLevelEncoder(
    vocab_size, embed_dim, max_len, num_heads, num_layers, dropout).to(device)

# Initialize flow embedding and encoding module
flow_embedding = FlowEmbedding(embed_dim, max_flow_length, dropout, vocab).to(device)
flow_encoder = FlowLevelEncoder(
    embed_dim, num_layers, num_heads, dropout, vocab, max_flow_length, mask_prob)

# training criterion initialize
optimizer = optim.Adam(list(packet_embedding.parameters()) + list(flow_embedding.parameters()) + list(packet_encoder.parameters()) + list(flow_encoder.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

from torch.utils.data import Dataset, DataLoader

# class PacketSequenceDataset(Dataset):
#     def __init__(self, packet_seq_dir, field_pos_dir, header_pos_dir, tokenizer, batch_size_1):
#         self.packet_seq_files = sorted([os.path.join(packet_seq_dir, file) for file in os.listdir(packet_seq_dir) if file.endswith('.txt')])
#         self.field_pos_files = sorted([os.path.join(field_pos_dir, file) for file in os.listdir(field_pos_dir) if file.endswith('.txt')])
#         self.header_pos_files = sorted([os.path.join(header_pos_dir, file) for file in os.listdir(header_pos_dir) if file.endswith('.txt')])
#         self.tokenizer = tokenizer
#         self.batch_size_1 = batch_size_1

#     def __len__(self):
#         return sum((len(open(file, 'r').readlines()) + self.batch_size_1 - 1) // self.batch_size_1 for file in self.packet_seq_files)

#     def __getitem__(self, idx):
#         # Determine which file and chunk index this sample corresponds to
#         file_idx, line_idx = divmod(idx, len(self.packet_seq_files))
#         packet_seq_file = self.packet_seq_files[file_idx]
#         field_pos_file = self.field_pos_files[file_idx]
#         header_pos_file = self.header_pos_files[file_idx]

#         with open(packet_seq_file, 'r', encoding='utf-8') as f:
#             hex_dumps = f.readlines()
        
#         padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(hex_dumps)
        
#         chunk_start = line_idx * self.batch_size_1
#         chunk_end = min((line_idx + 1) * self.batch_size_1, token_ids.size(0))
#         chunk = token_ids[chunk_start:chunk_end]
        
#         field_posn = field_pos(field_pos_file, chunk_start, chunk_end)
#         header_posn = header_pos(header_pos_file, chunk_start, chunk_end)
        
#         return chunk, field_posn, header_posn, packet_seq_file

class PacketSequenceDataset(Dataset):
    def __init__(self, packet_seq_dir, field_pos_dir, header_pos_dir, tokenizer, batch_size_1):
        self.packet_seq_files = sorted([os.path.join(packet_seq_dir, file) for file in os.listdir(packet_seq_dir) if file.endswith('.txt')])
        self.field_pos_files = sorted([os.path.join(field_pos_dir, file) for file in os.listdir(field_pos_dir) if file.endswith('.txt')])
        self.header_pos_files = sorted([os.path.join(header_pos_dir, file) for file in os.listdir(header_pos_dir) if file.endswith('.txt')])
        self.tokenizer = tokenizer
        self.batch_size_1 = batch_size_1

        # Calculate total length by summing up chunks across files
        self.total_chunks = []
        for file in self.packet_seq_files:
            num_lines = len(open(file, 'r').readlines())
            num_chunks = (num_lines + self.batch_size_1 - 1) // self.batch_size_1
            self.total_chunks.append(num_chunks)
        self.total_len = sum(self.total_chunks)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Find which file and chunk the index maps to
        cumulative_chunks = 0
        for file_idx, num_chunks in enumerate(self.total_chunks):
            if cumulative_chunks + num_chunks > idx:
                line_idx = idx - cumulative_chunks
                break
            cumulative_chunks += num_chunks
        else:
            raise IndexError("Index out of range")

        packet_seq_file = self.packet_seq_files[file_idx]
        field_pos_file = self.field_pos_files[file_idx]
        header_pos_file = self.header_pos_files[file_idx]

        with open(packet_seq_file, 'r', encoding='utf-8') as f:
            hex_dumps = f.readlines()
        
        padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(hex_dumps)
        
        # Slice out the chunk from token_ids
        chunk_start = line_idx * self.batch_size_1
        chunk_end = min((line_idx + 1) * self.batch_size_1, token_ids.size(0))
        chunk = token_ids[chunk_start:chunk_end]
        
        field_posn = field_pos(field_pos_file, chunk_start, chunk_end)
        header_posn = header_pos(header_pos_file, chunk_start, chunk_end)
        
        return chunk, field_posn, header_posn, packet_seq_file


# Create the dataset and data loader without shuffling
dataset = PacketSequenceDataset(
    packet_seq_dir='/home/satvik/spark/spark2/packets', 
    field_pos_dir='/home/satvik/spark/spark2/fields', 
    header_pos_dir='/home/satvik/spark/spark2/headers', 
    tokenizer=tokenizer, 
    batch_size_1=batch_size_1
)
print("1")
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

direction_dir = "/home/satvik/spark/spark2/direction"
print("2")
def process_encodings(encodings, packet_number):
    """ Process accumulated encodings and load corresponding direction data. """
    final_packet_encodings = torch.cat(encodings, dim=0).to(device)
    print("Final concatenated shape:", final_packet_encodings.shape)

    # Load the corresponding direction data from the text file
    direction_file_name = f'direction_{packet_number}.txt'
    direction_file_path = os.path.join(direction_dir, direction_file_name)

    with open(direction_file_path, 'r') as file:
        direction_data = [int(line.strip()) for line in file.readlines()]

    # Convert direction data to a tensor
    direction_tensor = torch.tensor(direction_data, device=device)

    # Call FlowEmbedding with the accumulated packet encodings and direction data
    flow_embeddings, pad_indices = flow_embedding(final_packet_encodings, direction_tensor)
    print("Flow embeddings computed for packet:", packet_number)
    print("flow embeddings: ", flow_embeddings.shape)
    print("-------------------------------------------------------------")

    # Compute flow encoding and MPM loss
    flow_encoding, mpm_loss = flow_encoder(flow_embeddings, pad_indices)
    print("flow encoding: ", flow_encoding.shape)
    print("mpm loss: ", mpm_loss)
    print(type(mpm_loss))

    # Ensure mpm_loss is a tensor and on the same device
    mpm_loss_tensor = mpm_loss[0].to(device)
    print(mpm_loss_tensor)
    return mpm_loss_tensor

def backward(total_loss):
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# Check contents of train_loader
# for i, (packet_sequences, field_position, header_position, file_name) in enumerate(train_loader):
#     print(f"Batch {i+1}")
#     print("Packet Sequences:", packet_sequences.shape)
#     print("Field Position:", field_position.shape)
#     print("Header Position:", header_position.shape)
#     print("File Name:", file_name) 


for epoch in range(1):
    print("Run", epoch)
    previous_packet_file = None
    all_packet_encodings = []
    previous_packet_number = None
    total_packet_enc_loss = 0

    # Ensure train_loader is sorted by file
    for i, (packet_sequences, field_position, header_position, file_name) in enumerate(train_loader):
        # Move data to the device
        packet_sequences = packet_sequences.to(device)
        field_position = field_position.to(device)
        header_position = header_position.to(device)
        current_file_name = file_name[0]  # File name for the current packet batch

        # If startwing a new file, finish processing for the previous file
        if previous_packet_file is not None and current_file_name != previous_packet_file:
            if all_packet_encodings:
                print(f"Completed processing file: {previous_packet_file}")
                mpm_loss = process_encodings(all_packet_encodings, previous_packet_number)
                total_length = len(all_packet_encodings)
                avg_packet_enc_loss = total_packet_enc_loss/total_length
                backward(mpm_loss + avg_packet_enc_loss)
                del mpm_loss, mlm_loss, sfbo_loss, encoded_packets_mean

            # Reset for the new file
            all_packet_encodings = []
            total_packet_enc_loss = 0
            previous_packet_number = None
            print(f"Started processing new file: {current_file_name}")

        # Update previous_packet_file to the current file name
        previous_packet_file = current_file_name

        # Generate packet embeddings
        # packet_embeddings = packet_embedding(token_ids=packet_sequences, field_pos=field_position, 
        # header_pos=header_position)

        mlm_loss, sfbo_loss, encoded_packets_mean = packet_encoder(
            packet_sequences, field_pos=field_position, header_pos=header_position
            )
        total_packet_enc_loss += (mlm_loss + sfbo_loss)
        all_packet_encodings.append(encoded_packets_mean)
        

        # del packet_sequences, field_position, header_position
        # torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.memory_reserved())


        print(f"Batch {i+1}")
        print(file_name)
        print("shape: ", len(all_packet_encodings))

        # Check for packet number from the file name
        match = re.search(r'packet_(\d+)\.txt', current_file_name)
        if match:
            current_packet_number = match.group(1)

            # Handle packet number change
            if previous_packet_number is not None and current_packet_number != previous_packet_number:
                print(f"Flow completed for packet: {previous_packet_number}")
                all_packet_encodings = []  # Reset encodings for each packet flow

            previous_packet_number = current_packet_number
        else:
            print("No packet number found in file name:", current_file_name)

    # After all batches, handle remaining encodings for the last file
    if all_packet_encodings and previous_packet_number is not None:
        print(f"Final processing for last flow packet: {previous_packet_number} in file: {previous_packet_file}")
        mpm_loss = process_encodings(all_packet_encodings, previous_packet_number)
        total_length = len(all_packet_encodings)
        avg_packet_enc_loss = total_packet_enc_loss/total_length
        total_packet_enc_loss = 0
        backward(mpm_loss + avg_packet_enc_loss)


# Only process packet encodings when switching files or at the end
# for epoch in range(1):
#     print("Run", epoch)
#     previous_packet_file = None
#     all_packet_encodings = []
#     previous_packet_number = None

#     for i, (packet_sequences, field_position, header_position, file_name) in enumerate(train_loader):
#         # Move data to the device in the training loop
#         packet_sequences = packet_sequences.to(device)
#         field_position = field_position.to(device)
#         header_position = header_position.to(device)
#         current_file_name = file_name[0]
#         print(current_file_name)
#         # Generate packet embeddings
#         packet_embeddings = packet_embedding(token_ids=packet_sequences, field_pos=field_position, 
#         header_pos=header_position
#         )
#         print("1", packet_sequences.shape, field_position.shape)
#         mlm_loss, sfbo_loss, encoded_packets, masked_packets, encoded_packets_mean = packet_encoder(
#             packet_sequences, field_pos=field_position, header_pos=header_position
#         )
#         print(packet_sequences.shape)
#         print("mlm loss: ", mlm_loss)
#         print("sfbo loss: ", sfbo_loss)

#         # print(current_file_name)    
#         # print(previous_packet_file)
#         # print(previous_packet_file is not None and all_packet_encodings)
#         # Handle file switch
#         if previous_packet_file is None or current_file_name != previous_packet_file:
#             if previous_packet_file is not None and all_packet_encodings:
#                 print(f"Completed processing file: {previous_packet_file}")
#                 mpm_loss = process_encodings(all_packet_encodings, previous_packet_number)
#                 print("1")
#                 backward(mpm_loss + mlm_loss + sfbo_loss)

#             previous_packet_file = current_file_name
#             all_packet_encodings = []
#             print(f"Started processing new file: {current_file_name}")

#         match = re.search(r'packet_(\d+)\.txt', current_file_name)
#         if match:
#             current_packet_number = match.group(1)
#             all_packet_encodings.append(encoded_packets_mean)

#             # Handle packet number change
#             if previous_packet_number is not None and current_packet_number != previous_packet_number:
#                 print(f"Flow completed for packet: {previous_packet_number}")
#                 all_packet_encodings = []

#             previous_packet_number = current_packet_number
#         else:
#             print("No packet number found in file name:", current_file_name)

#     # Handle remaining encodings
#     if all_packet_encodings and previous_packet_number is not None:
#         print("Final processing for last flow packet:", previous_packet_number)
#         mpm_loss = process_encodings(all_packet_encodings, previous_packet_number)
#         backward(mpm_loss + mlm_loss + sfbo_loss)

