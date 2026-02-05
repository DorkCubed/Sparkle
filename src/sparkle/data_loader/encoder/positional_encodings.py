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


def field_pos_safe(filename, start_idx, end_idx, device='cpu'):
    """
    Safe version of field_pos. Returns a padded tensor even if the file is missing
    or indices are out of range.
    """
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Warning: {filename} (field) not found. Returning zero tensor.")
        return torch.zeros((1, 1), dtype=torch.long, device=device)

    n = len(lines)
    if start_idx >= n:
        print(f"Warning: start_idx {start_idx} >= number of lines {n} in file {filename}")
        return torch.zeros((1, 1), dtype=torch.long, device=device)

    end_idx = min(end_idx, n)

    parsed_lines = [parse_line_to_list(line) for line in lines]
    token_field_pos_emb_list = parsed_lines[start_idx:end_idx]

    if not token_field_pos_emb_list:
        print(f"Warning: empty chunk from {start_idx} to {end_idx} in file {filename}")
        return torch.zeros((1, 1), dtype=torch.long, device=device)

    max_len = max(len(seq) for seq in parsed_lines) + 2
    padded_sequences = pad_sequences(token_field_pos_emb_list, max_len)

    return torch.tensor(padded_sequences, dtype=torch.long, device=device)


def header_pos_safe(filename, start_idx, end_idx, device='cpu'):
    """
    Safe version of header_pos. Returns a padded tensor even if the file is missing
    or indices are out of range.
    """
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Warning: {filename} (header) not found. Returning zero tensor.")
        return torch.zeros((1, 1), dtype=torch.long, device=device)

    n = len(lines)
    if start_idx >= n:
        print(f"Warning: start_idx {start_idx} >= number of lines {n} in file {filename}")
        return torch.zeros((1, 1), dtype=torch.long, device=device)

    end_idx = min(end_idx, n)

    parsed_lines = [parse_line_to_list(line) for line in lines]
    token_header_pos_emb_list = parsed_lines[start_idx:end_idx]

    if not token_header_pos_emb_list:
        print(f"Warning: empty chunk from {start_idx} to {end_idx} in file {filename}")
        return torch.zeros((1, 1), dtype=torch.long, device=device)

    max_len = max(len(seq) for seq in parsed_lines) + 2
    padded_sequences = pad_sequences(token_header_pos_emb_list, max_len)

    return torch.tensor(padded_sequences, dtype=torch.long, device=device)

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


