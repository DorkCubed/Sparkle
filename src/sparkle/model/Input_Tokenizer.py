import torch
import torch.nn as nn


class Tokenizer:

    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file):
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                token, token_id = line.strip().split('\t')
                vocab[token] = int(token_id)
            return vocab

    def encode_packet(self, input_file):
        all_tokens = []
        for hex_dump in input_file:
            tokens = []
            byte_data = list(bytes.fromhex(hex_dump))
            tokens = [str(token) for token in byte_data]
            tokens = ['[CLSp]'] + tokens + ['[SEP]']
            all_tokens.append(tokens)

        max_length = max(len(sublist) for sublist in all_tokens)
        padded_all_tokens = [sublist + ['[PAD]'] *
                             (max_length - len(sublist)) for sublist in all_tokens]

        token_ids = torch.tensor([[self.vocab.get(token, self.vocab.get(
            '[UNK]', 0)) for token in sublist] for sublist in padded_all_tokens], dtype=torch.long)

        mask = ~torch.isin(token_ids, torch.tensor([0]))

        return padded_all_tokens, token_ids, mask, max_length  # save the max len value

    def decode(self, token_ids):
        tokens = [[token for token, id_ in self.vocab.items() if id_ in sublist]
                  for sublist in token_ids]
        return tokens
    
    def encode_flow(self, cls_packet_embeddings):
        # Ensure embeddings are on the same device as cls_packet_embeddings
        device = cls_packet_embeddings.device

        # Extract [CLSp] token embeddings (first token of all packets)
        cls_packet_embeddings = cls_packet_embeddings[:, 0, :]  # Shape: [num_packets, embed_dim]

        # Define the [CLSf] token embedding and move it to the correct device
        clsf_token_index = torch.tensor(self.vocab['[CLSf]'], device=device)
        clsf_embedding = self.token_embed(clsf_token_index).unsqueeze(0)  # Shape: [1, embed_dim]

        # Define the [SEP] token embedding and move it to the correct device
        sep_token_index = torch.tensor(self.vocab['[SEP]'], device=device)
        sep_embedding = self.token_embed(sep_token_index).unsqueeze(0)  # Shape: [1, embed_dim]

        # Define the [PAD] token embedding and move it to the correct device
        pad_token_index = torch.tensor(self.vocab['[PAD]'], device=device)
        pad_embedding = self.token_embed(pad_token_index).unsqueeze(0)  # Shape: [1, embed_dim]

        # Divide into chunks of 510, padding if necessary
        fraction_size = 510
        num_chunks = (cls_packet_embeddings.size(0) + fraction_size - 1) // fraction_size
        padded_embeddings = []

        for i in range(num_chunks):
            # Slice the current chunk
            start = i * fraction_size
            end = min(start + fraction_size, cls_packet_embeddings.size(0))
            chunk = cls_packet_embeddings[start:end]

            # If the chunk is smaller than 510, pad it
            if chunk.size(0) < fraction_size:
                pad_size = fraction_size - chunk.size(0)
                padding = pad_embedding.expand(pad_size, -1)  # Shape: [pad_size, embed_dim]
                chunk = torch.cat([chunk, padding], dim=0)  # Shape: [510, embed_dim]

            padded_embeddings.append(chunk)

        # Add [CLSf] at the beginning and [SEP] at the end of each chunk
        encoded_flow = []
        for chunk in padded_embeddings:
            encoded_chunk = torch.cat([clsf_embedding, chunk, sep_embedding], dim=0)  # Shape: [512, embed_dim]
            encoded_flow.append(encoded_chunk)

        # Concatenate all chunks along the token sequence dimension
        encoded_flow = torch.stack(encoded_flow)  # Shape: [num_chunks, 512, embed_dim]

        return encoded_flow




    # def encode_flow(self, cls_packet_embeddings):
    #     # defining the cls token with its embedding
    #     clsf_embedding = nn.Embedding(1, embedding_dim=128)
    #     clsf_token_embedding = clsf_embedding(torch.zeros(
    #         cls_packet_embeddings.size(0), dtype=torch.long))

    #     # Extract [CLS_p] token embeddings
    #     cls_packet_embeddings = cls_packet_embeddings[:, 0, :]

    #     # then adding it to cls_packet_embedding
    #     cls_packet_embeddings = torch.cat(
    #         [clsf_token_embedding, cls_packet_embeddings], dim=1)

    #     # Split the sequence into fractions of 510 tokens
    #     fraction_size = 510
    #     fractions = [cls_packet_embeddings[:, i:i+fraction_size]
    #                  for i in range(0, cls_packet_embeddings.size(1), fraction_size)]

    #     # Add [SEP] token embedding after every 510 tokens
    #     sep_embedding = nn.Embedding(1, embedding_dim=128)
    #     sep_token_embedding = sep_embedding(torch.zeros(
    #         cls_packet_embeddings.size(0), dtype=torch.long))

    #     encoded_flow = []
    #     for fraction in fractions:
    #         encoded_flow.append(
    #             torch.cat([fraction, sep_token_embedding.unsqueeze(1)], dim=1))

    #     # Concatenate all chunks
    #     encoded_flow = torch.cat(encoded_flow, dim=1)

    #     return encoded_flow


# custom_vocab_path = r"C:\Users\Kavish\OneDrive\Desktop\SPARKLE\Custom-Vocab.txt"

# input_file = []
# with open(r"C:\Users\Kavish\OneDrive\Desktop\SPARKLE\Input-Hex-Dump.txt", 'r') as file:
#     for line in file:
#         # Assuming each line in the file contains a hex dump
#         input_file.append(line.strip())

# tokenizer = Tokenizer(vocab_file=custom_vocab_path)
# encoded_values, token_ids, mask = tokenizer.encode_packet(input_file)

# print(encoded_values)
# print(token_ids)
# print(mask)
