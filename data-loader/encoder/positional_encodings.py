import torch
import torch.nn as nn

def parse_line_to_list(line):
    return [int(x) for x in line.strip().split()]

def pad_sequences(sequences, max_len, padding_value=0):
    padded_sequences = []
    for seq in sequences:
        seq = seq + [padding_value] * (max_len - len(seq))
        padded_sequences.append(seq)
    return padded_sequences

def field_pos(filename, start_idx, end_idx):
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error 2: {filename} (field) not found.")
        exit()

    len_list = []
    for line in lines:
        indices = parse_line_to_list(line)
        len_list.append(indices)

    token_field_pos_emb_list = [parse_line_to_list(lines[i]) for i in range(start_idx, end_idx)]
    max_len = max(len(seq) for seq in len_list) + 2
    padded_sequences = pad_sequences(token_field_pos_emb_list, max_len)
    
    return torch.tensor(padded_sequences, dtype=torch.long)

def header_pos(filename, start_idx, end_idx):
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error 1: {filename} (header) not found.")
        exit()

    len_list = []
    for line in lines:
        indices = parse_line_to_list(line)
        len_list.append(indices)

    token_header_pos_emb_list = [parse_line_to_list(lines[i]) for i in range(start_idx, end_idx)]
    max_len = max(len(seq) for seq in len_list) + 2
    padded_sequences = pad_sequences(token_header_pos_emb_list, max_len)
    
    return torch.tensor(padded_sequences, dtype=torch.long)

# def header_pos(filename, chunk_start, chunk_end):
#     try:
#         with open(filename, "r") as file:
#             lines = file.readlines()
#     except FileNotFoundError:
#         print(f"Error 1 : {filename} (header) not found.")
#         exit()

#     token_header_pos_emb_list = []
    
#     for line in lines:
#         indices = parse_line_to_list(line)
#         token_header_pos_emb_list.append(indices)
    
#     max_len = max(len(seq) for seq in token_header_pos_emb_list)
#     max_len += 2
#     padded_sequences = pad_sequences(token_header_pos_emb_list, max_len)

#     indices_tensor = torch.tensor(padded_sequences, dtype=torch.long)

#     # Slice the tensor based on chunk_start and chunk_end
#     return indices_tensor[chunk_start:chunk_end]

# def field_pos(filename, chunk_start, chunk_end):
#     try:
#         with open(filename, "r") as file:
#             lines = file.readlines()
#     except FileNotFoundError:
#         print(f"Error 2 : {filename} (field) not found.")
#         exit()

#     token_field_pos_emb_list = []
    
#     for line in lines:
#         indices = parse_line_to_list(line)
#         token_field_pos_emb_list.append(indices)
    
#     max_len = max(len(seq) for seq in token_field_pos_emb_list)
#     max_len += 2
#     padded_sequences = pad_sequences(token_field_pos_emb_list, max_len)

#     indices_tensor = torch.tensor(padded_sequences, dtype=torch.long)

#     # Slice the tensor based on chunk_start and chunk_end
#     return indices_tensor[chunk_start:chunk_end]


