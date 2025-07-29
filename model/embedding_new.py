import torch
import torch.nn as nn

class PacketEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.token_pos_embed = nn.Embedding(max_len, embed_dim)
        self.field_pos_embed = nn.Embedding(max_len, embed_dim)
        # check for dropout and embed in pytorch
        self.header_pos_embed = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, token_ids, field_pos, header_pos):
        # print("token ids :", token_ids.shape)
        # print("TOKEEENN", token_ids.shape)
        # token_ids = token_ids.squeeze(0)
        # print("token ids :", token_ids.shape)
        num_packets, seq_len = token_ids.size()
        # print(field_pos.shape)
        # print(header_pos.shape)
        # token_pos = torch.tensor([i for i in range(num_packets)])
        # token_pos = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).repeat(num_packets, 1)
        token_emb = self.token_embed(token_ids)
        token_pos_emb = self.token_pos_embed(torch.arange(seq_len, device=token_ids.device).unsqueeze(0).repeat(num_packets, 1))
        field_pos_emb = self.field_pos_embed(field_pos)
        header_pos_emb = self.header_pos_embed(header_pos)
        # print(token_emb.shape)
        # print("device: ", token_emb.device, token_pos_emb.device, field_pos_emb.device, 
        #       token_pos.device)
        # print(token_pos_emb.shape)
        # print(field_pos_emb.shape)
        # print(header_pos_emb.shape)
        # embed_val = token_emb + token_pos_emb + field_pos_emb + header_pos_emb
        embed_val = self.drop(token_emb + token_pos_emb + field_pos_emb + header_pos_emb)
        # del token_emb, token_pos_emb, field_pos_emb, header_pos_emb
        torch.cuda.empty_cache()
        # return embed_val, token_emb, token_pos_emb, field_pos_emb, header_pos_emb
        return embed_val
    

class FlowEmbedding(nn.Module):
    def __init__(self, embed_dim, max_packets, dropout, vocab):
        super().__init__()
        self.packet_pos_embed = nn.Embedding(max_packets, embed_dim)
        self.direction_embed = nn.Embedding(3, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.vocab = vocab
        self.token_embed = nn.Embedding(len(vocab), embed_dim)

    def forward(self, cls_packet_embeddings, direction):
        device = cls_packet_embeddings.device

        # Extract [CLS_p] token embeddings from the beginning of each packet
        cls_packet_embeddings = cls_packet_embeddings[:, 0, :]  # Shape: [num_packets, embed_dim]

        # Define [CLSf], [SEP], and [PAD] token embeddings using vocab indices
        clsf_token_index = torch.tensor(self.vocab['[CLSf]'], device=device)
        sep_token_index = torch.tensor(self.vocab['[SEP]'], device=device)
        pad_token_index = torch.tensor(self.vocab['[PAD]'], device=device)

        # Get the embeddings for special tokens
        clsf_token_embedding = self.token_embed(clsf_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
        sep_token_embedding = self.token_embed(sep_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
        pad_token_embedding = self.token_embed(pad_token_index).unsqueeze(0)  # Shape: [1, embed_dim]

        # Divide packet embeddings into chunks of size 510
        num_packets = cls_packet_embeddings.size(0)
        chunk_size = 510
        chunks = []
        direction_chunks = []
        pad_indices = []

        for i in range(0, num_packets, chunk_size):
            # Process cls_packet_embeddings
            chunk = cls_packet_embeddings[i:i + chunk_size]
            padding_mask = torch.zeros(chunk_size, device=device, dtype=torch.bool)  # Mask for this chunk

            if chunk.size(0) < chunk_size:  # Pad if the chunk is less than 510
                pad_size = chunk_size - chunk.size(0)
                padding = pad_token_embedding.expand(pad_size, -1)
                chunk = torch.cat([chunk, padding], dim=0)
                # Set padding indices to True in the mask
                padding_mask[-pad_size:] = True

            # Add [CLSf] and [SEP] tokens
            chunk = torch.cat([clsf_token_embedding, chunk, sep_token_embedding], dim=0)  # Shape: [512, embed_dim]
            chunks.append(chunk)

            # Process direction
            direction_chunk = direction[i:i + chunk_size]
            direction_chunk = self.direction_embed(direction_chunk)  # Embed direction
            if direction_chunk.size(0) < chunk_size:  # Pad if the chunk is less than 510
                pad_size = chunk_size - direction_chunk.size(0)
                direction_padding = pad_token_embedding.expand(pad_size, -1)
                direction_chunk = torch.cat([direction_chunk, direction_padding], dim=0)

            # Add [CLSf] and [SEP] tokens
            direction_chunk = torch.cat([clsf_token_embedding, direction_chunk, sep_token_embedding], dim=0)  # Shape: [512, embed_dim]
            direction_chunks.append(direction_chunk)

            # Update padding mask with [CLSf] and [SEP] tokens
            pad_mask = torch.cat(
                [torch.tensor([True], device=device), padding_mask, torch.tensor([True], device=device)]
            )
            pad_indices.append(pad_mask)

        # Stack all chunks into tensors of shape [num_chunks, 512, embed_dim]
        cls_packet_embeddings = torch.stack(chunks, dim=0)
        direction_emb = torch.stack(direction_chunks, dim=0)

        # Stack padding masks into a tensor of shape [num_chunks, 512]
        pad_indices = torch.stack(pad_indices, dim=0)

        # Add position embeddings and sum all embeddings
        num_chunks = cls_packet_embeddings.size(0)
        packet_pos = torch.arange(512, device=device).unsqueeze(0).expand(num_chunks, -1)  # Shape: [num_chunks, 512]
        packet_pos_emb = self.packet_pos_embed(packet_pos)

        # Compute the final embeddings
        embed_val = cls_packet_embeddings + packet_pos_emb + direction_emb
        embed_val = self.drop(embed_val)
        
        return embed_val, pad_indices


    # def forward(self, cls_packet_embeddings, direction):
    #     device = cls_packet_embeddings.device

    #     cls_packet_embeddings = cls_packet_embeddings[:, 0, :]
    #     clsf_token_index = torch.tensor(self.vocab['[CLSf]'], device=device)
    #     sep_token_index = torch.tensor(self.vocab['[SEP]'], device=device)
    #     pad_token_index = torch.tensor(self.vocab['[PAD]'], device=device)

    #     clsf_token_embedding = self.token_embed(clsf_token_index).unsqueeze(0)
    #     sep_token_embedding = self.token_embed(sep_token_index).unsqueeze(0)
    #     pad_token_embedding = self.token_embed(pad_token_index).unsqueeze(0)

    #     num_packets = cls_packet_embeddings.size(0)
    #     chunk_size = 510
    #     num_chunks = (num_packets + chunk_size - 1) // chunk_size

    #     chunks, direction_chunks, pad_indices = [], [], [None] * num_chunks

    #     for idx, i in enumerate(range(0, num_packets, chunk_size)):
    #         chunk = cls_packet_embeddings[i:i + chunk_size]
    #         padding_mask = torch.zeros(chunk_size, device=device, dtype=torch.bool)

    #         if chunk.size(0) < chunk_size:
    #             pad_size = chunk_size - chunk.size(0)
    #             padding = pad_token_embedding.expand(pad_size, -1)
    #             chunk = torch.cat([chunk, padding], dim=0)
    #             padding_mask[-pad_size:] = True

    #         chunks.append(chunk)
    #         pad_indices[idx] = torch.cat(
    #             [torch.tensor([True], device=device), padding_mask, torch.tensor([True], device=device)]
    #         )

    #     for idx, i in enumerate(range(0, num_packets, chunk_size)):
    #         direction_chunk = direction[i:i + chunk_size]  # Shape: [sequence length]

    #         # Embed the direction_chunk to match embedding dimension
    #         direction_chunk = self.direction_embed(direction_chunk)  # Shape: [sequence length, embed_dim]

    #         padding_mask = torch.zeros(chunk_size, device=device, dtype=torch.bool)

    #         if direction_chunk.size(0) < chunk_size:
    #             pad_size = chunk_size - direction_chunk.size(0)
    #             direction_padding = pad_token_embedding.expand(pad_size, -1)  # Shape: [pad_size, embed_dim]
    #             print(direction_chunk.shape, direction_padding.shape)
    #             direction_chunk = torch.cat([direction_chunk, direction_padding], dim=0)  # Shape: [chunk_size, embed_dim]
    #             padding_mask[-pad_size:] = True

    #         direction_chunks.append(direction_chunk)
    #         pad_indices[idx] = torch.cat(
    #             [torch.tensor([True], device=device), padding_mask, torch.tensor([True], device=device)]
    #         )

    #     # for idx, i in enumerate(range(0, num_packets, chunk_size)):
    #     #     direction_chunk = direction[i:i + chunk_size]
    #     #     padding_mask = torch.zeros(chunk_size, device=device, dtype=torch.bool)

    #     #     if direction_chunk.size(0) < chunk_size:
    #     #         pad_size = chunk_size - direction_chunk.size(0)
    #     #         direction_padding = pad_token_embedding.expand(pad_size, -1)
    #     #         print(direction_chunk.shape, direction_padding.shape)
    #     #         direction_chunk = torch.cat([direction_chunk, direction_padding], dim=0)
    #     #         padding_mask[-pad_size:] = True

    #     #     direction_chunks.append(direction_chunk)
    #     #     pad_indices[idx] = torch.cat(
    #     #         [torch.tensor([True], device=device), padding_mask, torch.tensor([True], device=device)]
    #     #     )

    #     cls_packet_embeddings = torch.stack(chunks, dim=0)
    #     direction_emb = torch.stack(direction_chunks, dim=0)
    #     pad_indices = torch.stack(pad_indices, dim=0)

    #     packet_pos = torch.arange(512, device=device).unsqueeze(0).expand(num_chunks, -1)
    #     packet_pos_emb = self.packet_pos_embed(packet_pos)
    #     print(cls_packet_embeddings.shape, packet_pos_emb.shape, direction_emb.shape)
    #     embed_val = cls_packet_embeddings + packet_pos_emb + direction_emb
    #     embed_val = self.drop(embed_val)
        
    #     return embed_val, pad_indices

# class FlowEmbedding(nn.Module):
#     def __init__(self, embed_dim, max_packets, dropout, vocab):
#         super().__init__()
#         self.packet_pos_embed = nn.Embedding(max_packets, embed_dim)
#         self.direction_embed = nn.Embedding(3, embed_dim)
#         self.drop = nn.Dropout(dropout)
#         self.vocab = vocab

#         # Embedding for special tokens
#         self.token_embed = nn.Embedding(len(vocab), embed_dim)
    
#     def forward(self, cls_packet_embeddings, direction):
#         device = cls_packet_embeddings.device

#         # Extract [CLS_p] token embeddings from the beginning of each packet
#         cls_packet_embeddings = cls_packet_embeddings[:, 0, :]  # Shape: [num_packets, embed_dim]

#         # Define [CLSf], [SEP], and [PAD] token embeddings using vocab indices
#         clsf_token_index = torch.tensor(self.vocab['[CLSf]'], device=device)
#         sep_token_index = torch.tensor(self.vocab['[SEP]'], device=device)
#         pad_token_index = torch.tensor(self.vocab['[PAD]'], device=device)

#         # Get the embeddings for special tokens
#         clsf_token_embedding = self.token_embed(clsf_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
#         sep_token_embedding = self.token_embed(sep_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
#         pad_token_embedding = self.token_embed(pad_token_index).unsqueeze(0)  # Shape: [1, embed_dim]

#         # Divide packet embeddings into chunks of size 510
#         num_packets = cls_packet_embeddings.size(0)
#         chunk_size = 510
#         chunks = []
#         pad_indices = []  # List to store padding indices

#         for i in range(0, num_packets, chunk_size):
#             chunk = cls_packet_embeddings[i:i + chunk_size]
#             padding_mask = torch.zeros(chunk_size, device=device, dtype=torch.bool)  # Mask for this chunk

#             if chunk.size(0) < chunk_size:  # Pad if the chunk is less than 510
#                 pad_size = chunk_size - chunk.size(0)
#                 padding = pad_token_embedding.expand(pad_size, -1)
#                 chunk = torch.cat([chunk, padding], dim=0)
#                 # Set padding indices to True in the mask
#                 padding_mask[-pad_size:] = True

#             chunks.append(chunk)
#             pad_indices.append(padding_mask)  # Store padding mask for the chunk

#         # Add [CLSf] and [SEP] tokens to each chunk
#         for i in range(len(chunks)):
#             # Add [CLSf] token at the beginning and [SEP] token at the end
#             chunks[i] = torch.cat([clsf_token_embedding, chunks[i], sep_token_embedding], dim=0)  # Shape: [512, embed_dim]

#             # Update padding mask with [CLSf] and [SEP] tokens
#             pad_indices[i] = torch.cat([torch.tensor([True], device=device), pad_indices[i], torch.tensor([True], device=device)])

#         # Stack all chunks into a tensor of shape [num_chunks, 512, embed_dim]
#         cls_packet_embeddings = torch.stack(chunks, dim=0)

#         # Process the direction embeddings similarly
#         direction_emb = self.direction_embed(direction)  # Embed direction
#         direction_chunks = []

#         for i in range(0, num_packets, chunk_size):
#             direction_chunk = direction_emb[i:i + chunk_size]
#             padding_mask = torch.zeros(chunk_size, device=device, dtype=torch.bool)  # Mask for this direction chunk

#             if direction_chunk.size(0) < chunk_size:  # Pad if the chunk is less than 510
#                 pad_size = chunk_size - direction_chunk.size(0)
#                 direction_padding = pad_token_embedding.expand(pad_size, -1)
#                 direction_chunk = torch.cat([direction_chunk, direction_padding], dim=0)
#                 # Set padding indices to True in the mask
#                 padding_mask[-pad_size:] = True

#             direction_chunks.append(direction_chunk)
#             pad_indices[i] = torch.cat([torch.tensor([True], device=device), padding_mask, torch.tensor([True], device=device)])

#         # Add [CLSf] and [SEP] tokens to each direction chunk
#         for i in range(len(direction_chunks)):
#             direction_chunks[i] = torch.cat([clsf_token_embedding, direction_chunks[i], sep_token_embedding], dim=0)  # Shape: [512, embed_dim]

#         # Stack all direction chunks into a tensor of shape [num_chunks, 512, embed_dim]
#         direction_emb = torch.stack(direction_chunks, dim=0)

#         # Stack padding masks into a tensor of shape [num_chunks, 512]
#         pad_indices = torch.stack(pad_indices, dim=0)  # Shape: [num_chunks, 512]

#         # Add position embeddings and sum all embeddings
#         num_chunks = cls_packet_embeddings.size(0)
#         packet_pos = torch.arange(512, device=device).unsqueeze(0).expand(num_chunks, -1)  # Shape: [num_chunks, 512]
#         packet_pos_emb = self.packet_pos_embed(packet_pos)

#         # Compute the final embeddings
#         embed_val = cls_packet_embeddings + packet_pos_emb + direction_emb
#         embed_val = self.drop(embed_val)
        
#         return embed_val, pad_indices

    
    # def forward(self, cls_packet_embeddings, direction):
    #     device = cls_packet_embeddings.device

    #     # Extract [CLS_p] token embeddings from the beginning of each packet
    #     cls_packet_embeddings = cls_packet_embeddings[:, 0, :]  # Shape: [num_packets, embed_dim]

    #     # Define [CLSf], [SEP], and [PAD] token embeddings using vocab indices
    #     clsf_token_index = torch.tensor(self.vocab['[CLSf]'], device=device)
    #     sep_token_index = torch.tensor(self.vocab['[SEP]'], device=device)
    #     pad_token_index = torch.tensor(self.vocab['[PAD]'], device=device)

    #     # Get the embeddings for special tokens
    #     clsf_token_embedding = self.token_embed(clsf_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
    #     sep_token_embedding = self.token_embed(sep_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
    #     pad_token_embedding = self.token_embed(pad_token_index).unsqueeze(0)  # Shape: [1, embed_dim]

    #     # Divide packet embeddings into chunks of size 510
    #     num_packets = cls_packet_embeddings.size(0)
    #     chunk_size = 510
    #     chunks = []

    #     for i in range(0, num_packets, chunk_size):
    #         chunk = cls_packet_embeddings[i:i + chunk_size]
    #         if chunk.size(0) < chunk_size:  # Pad if the chunk is less than 510
    #             pad_size = chunk_size - chunk.size(0)
    #             padding = pad_token_embedding.expand(pad_size, -1)
    #             chunk = torch.cat([chunk, padding], dim=0)
    #         chunks.append(chunk)

    #     # Add [CLSf] and [SEP] tokens to each chunk
    #     for i in range(len(chunks)):
    #         chunks[i] = torch.cat([clsf_token_embedding, chunks[i], sep_token_embedding], dim=0)  # Shape: [512, embed_dim]

    #     # Stack all chunks into a tensor of shape [num_chunks, 512, embed_dim]
    #     cls_packet_embeddings = torch.stack(chunks, dim=0)

    #     # Process the direction embeddings similarly
    #     direction_emb = self.direction_embed(direction)  # Embed direction
    #     direction_chunks = []

    #     for i in range(0, num_packets, chunk_size):
    #         direction_chunk = direction_emb[i:i + chunk_size]
    #         if direction_chunk.size(0) < chunk_size:  # Pad if the chunk is less than 510
    #             pad_size = chunk_size - direction_chunk.size(0)
    #             direction_padding = pad_token_embedding.expand(pad_size, -1)
    #             direction_chunk = torch.cat([direction_chunk, direction_padding], dim=0)
    #         direction_chunks.append(direction_chunk)

    #     # Add [CLSf] and [SEP] tokens to each direction chunk
    #     for i in range(len(direction_chunks)):
    #         direction_chunks[i] = torch.cat([clsf_token_embedding, direction_chunks[i], sep_token_embedding], dim=0)  # Shape: [512, embed_dim]

    #     # Stack all direction chunks into a tensor of shape [num_chunks, 512, embed_dim]
    #     direction_emb = torch.stack(direction_chunks, dim=0)

    #     # Add position embeddings and sum all embeddings
    #     num_chunks = cls_packet_embeddings.size(0)
    #     packet_pos = torch.arange(512, device=device).unsqueeze(0).expand(num_chunks, -1)  # Shape: [num_chunks, 512]
    #     packet_pos_emb = self.packet_pos_embed(packet_pos)

    #     # Compute the final embeddings
    #     embed_val = cls_packet_embeddings + packet_pos_emb + direction_emb
    #     embed_val = self.drop(embed_val)

    #     return embed_val


    # def forward(self, cls_packet_embeddings, direction):
    #     device = cls_packet_embeddings.device

    #     # Extract [CLS_p] token embeddings from the beginning of each packet
    #     cls_packet_embeddings = cls_packet_embeddings[:, 0, :]  # Shape: [num_packets, embed_dim]
    #     # print(cls_packet_embeddings.shape)

    #     # Define [CLSf], [SEP], and [PAD] token embeddings using vocab indices
    #     clsf_token_index = torch.tensor(self.vocab['[CLSf]'], device=device)
    #     sep_token_index = torch.tensor(self.vocab['[SEP]'], device=device)
    #     pad_token_index = torch.tensor(self.vocab['[PAD]'], device=device)

    #     # Get the embeddings for special tokens
    #     clsf_token_embedding = self.token_embed(clsf_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
    #     sep_token_embedding = self.token_embed(sep_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
    #     pad_token_embedding = self.token_embed(pad_token_index).unsqueeze(0)  # Shape: [1, embed_dim]

    #     # Pad cls_packet_embeddings to make it [510, embed_dim]
    #     if cls_packet_embeddings.size(0) < 510:
    #         pad_size = 510 - cls_packet_embeddings.size(0)
    #         padding = pad_token_embedding.expand(pad_size, -1)
    #         cls_packet_embeddings = torch.cat([cls_packet_embeddings, padding], dim=0)

    #     # Concatenate the [CLSf] token at the beginning
    #     cls_packet_embeddings = torch.cat([clsf_token_embedding, cls_packet_embeddings], dim=0)  # Shape: [1 + 510, embed_dim]
    #     # print("cls: ", cls_packet_embeddings.shape)

    #     # Add [SEP] token
    #     cls_packet_embeddings = torch.cat([cls_packet_embeddings, sep_token_embedding], dim=0)  # Shape: [2 + 510, embed_dim]

    #     # Ensure the shape is correct
    #     assert cls_packet_embeddings.size(0) == 512, "Expected shape [512, embed_dim] after adding [CLSf] and [SEP] tokens"

    #     # Handle direction embedding before padding
    #     # print(direction)
    #     direction_emb = self.direction_embed(direction)  # Embed direction before padding
    #     if direction_emb.size(0) < 510:
    #         pad_size = 510 - direction_emb.size(0)
    #         direction_padding = pad_token_embedding.expand(pad_size, -1)
    #         direction_emb = torch.cat([direction_emb, direction_padding], dim=0)

    #     # Concatenate [CLSf] and [SEP] embeddings to the direction embeddings
    #     direction_emb = torch.cat([clsf_token_embedding, direction_emb, sep_token_embedding], dim=0)  # Shape: [512, embed_dim]

    #     # print("Direction embedding shape: ", direction_emb.shape)

    #     # Position embeddings
    #     num_packets = cls_packet_embeddings.size(0)
    #     packet_pos = torch.arange(num_packets, device=device).unsqueeze(0)
    #     packet_pos_emb = self.packet_pos_embed(packet_pos)

    #     # Add position and direction embeddings
    #     print("----- FLOW EMBED -----")
    #     # print("1. ", cls_packet_embeddings.shape)
    #     # print("2. ", packet_pos_emb.shape)
    #     # print("3. ", direction_emb.shape)

    #     embed_val = cls_packet_embeddings + packet_pos_emb + direction_emb
    #     embed_val = self.drop(embed_val)

    #     return embed_val

    # def forward(self, cls_packet_embeddings, direction):
    #     device = cls_packet_embeddings.device

    #     # Extract [CLS_p] token embeddings from the beginning of each packet
    #     cls_packet_embeddings = cls_packet_embeddings[:, 0, :]  # Shape: [num_packets, embed_dim]
    #     print(cls_packet_embeddings.shape)

    #     # Define [CLSf] and [SEP] token embeddings using vocab indices
    #     clsf_token_index = torch.tensor(self.vocab['[CLSf]'], device=device)
    #     sep_token_index = torch.tensor(self.vocab['[SEP]'], device=device)
    #     pad_token_index = torch.tensor(self.vocab['[PAD]'], device=device)

    #     # Get the embeddings for special tokens
    #     clsf_token_embedding = self.token_embed(clsf_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
    #     sep_token_embedding = self.token_embed(sep_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
    #     pad_token_embedding = self.token_embed(pad_token_index).unsqueeze(0)  # Shape: [1, embed_dim]

    #     # Pad cls_packet_embeddings to make it [510, 768]
    #     if cls_packet_embeddings.size(0) < 510:
    #         pad_size = 510 - cls_packet_embeddings.size(0)
    #         padding = pad_token_embedding.expand(pad_size, -1)
    #         cls_packet_embeddings = torch.cat([cls_packet_embeddings, padding], dim=0)

    #     # Concatenate the [CLSf] token at the beginning
    #     cls_packet_embeddings = torch.cat([clsf_token_embedding, cls_packet_embeddings], dim=0)  # Shape: [1 + 510, 768]
    #     print("cls: ", cls_packet_embeddings.shape)

    #     # Add [SEP] token
    #     cls_packet_embeddings = torch.cat([cls_packet_embeddings, sep_token_embedding], dim=0)  # Shape: [2 + 510, 768]

    #     # Ensure the shape is correct
    #     assert cls_packet_embeddings.size(0) == 512, "Expected shape [512, 768] after adding [CLSf] and [SEP] tokens"

    #     # Handle direction embedding
    #     if direction.size(0) < 510:
    #         pad_size = 510 - direction.size(0)
    #         direction_padding = torch.zeros(pad_size, device=device, dtype=direction.dtype)
    #         direction = torch.cat([direction, direction_padding], dim=0)

    #     # Concatenate [CLSf] and [SEP] to the direction embedding
    #     direction = torch.cat([torch.tensor([0], device=device, dtype=direction.dtype), direction], dim=0)  # Add [CLSf]
    #     direction = torch.cat([direction, torch.tensor([0], device=device, dtype=direction.dtype)], dim=0)  # Add [SEP]

    #     print("dir shape: ", direction.shape, direction.size(0))
    #     # Ensure the shape is correct
    #     # assert direction.size(0) == 512, "Expected shape [512] after adding [CLSf] and [SEP] tokens"
    #     print(direction)
    #     # Expand direction tensor for embedding lookup
    #     # direction_emb = self.direction_embed(direction).unsqueeze(0)  # Shape: [1, 512, embed_dim]
    #     direction_emb = self.direction_embed(direction)
    #     # Position embeddings
    #     num_packets = cls_packet_embeddings.size(0)
    #     packet_pos = torch.arange(num_packets, device=device).unsqueeze(0)
    #     packet_pos_emb = self.packet_pos_embed(packet_pos)

    #     # Add position and direction embeddings
    #     print("1. ", cls_packet_embeddings.shape)
    #     print("1. ", packet_pos_emb.shape)
    #     print("1. ", direction_emb.shape)

    #     embed_val = cls_packet_embeddings + packet_pos_emb + direction_emb
    #     embed_val = self.drop(embed_val)

    #     return embed_val


    # def forward(self, cls_packet_embeddings, direction):
    #     device = cls_packet_embeddings.device

    #     # Extract [CLS_p] token embeddings from the beginning of each packet
    #     cls_packet_embeddings = cls_packet_embeddings[:, 0, :]  # Shape: [num_packets, embed_dim]
    #     print(cls_packet_embeddings.shape)
    #     # Define [CLSf] and [SEP] token embeddings using vocab indices
    #     clsf_token_index = torch.tensor(self.vocab['[CLSf]'], device=device)
    #     sep_token_index = torch.tensor(self.vocab['[SEP]'], device=device)
    #     pad_token_index = torch.tensor(self.vocab['[PAD]'], device=device)

    #     # Get the embeddings for special tokens
    #     clsf_token_embedding = self.token_embed(clsf_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
    #     sep_token_embedding = self.token_embed(sep_token_index).unsqueeze(0)  # Shape: [1, embed_dim]
    #     pad_token_embedding = self.token_embed(pad_token_index).unsqueeze(0)  # Shape: [1, embed_dim]

    #     # Concatenate the [CLSf] token at the beginning
    #     cls_packet_embeddings = torch.cat([clsf_token_embedding, cls_packet_embeddings], dim=0)
    #     print("cls: ", cls_packet_embeddings.shape)
    #     # Split into chunks of 510 and pad if necessary
    #     fraction_size = 510
    #     chunks = [cls_packet_embeddings[i:i+fraction_size] for i in range(0, cls_packet_embeddings.size(0), fraction_size)]

    #     # Pad chunks and add [SEP] token
    #     padded_chunks = []
    #     for chunk in chunks:
    #         if chunk.size(0) < fraction_size:  # Pad if the chunk is less than 510
    #             pad_size = fraction_size - chunk.size(0)
    #             padding = pad_token_embedding.expand(pad_size, -1)
    #             chunk = torch.cat([chunk, padding], dim=0)
    #         # Add [SEP] token
    #         chunk = torch.cat([chunk, sep_token_embedding], dim=0)
    #         padded_chunks.append(chunk)

    #     # Concatenate all chunks into one tensor
    #     encoded_flow = torch.cat(padded_chunks, dim=0).unsqueeze(0)  # Add batch dimension

    #     # Position embeddings and direction embeddings
    #     num_packets = encoded_flow.size(1)
    #     packet_pos = torch.arange(num_packets, device=device).unsqueeze(0)
    #     packet_pos_emb = self.packet_pos_embed(packet_pos)
    #     direction_emb = self.direction_embed(direction).unsqueeze(0)

    #     # Add position and direction embeddings
    #     print("1. ", encoded_flow.shape)
    #     print("1. ", packet_pos_emb.shape)
    #     print("1. ", direction_emb.shape)

    #     embed_val = encoded_flow + packet_pos_emb + direction_emb
    #     embed_val = self.drop(embed_val)

    #     return embed_val


# class FlowEmbedding(nn.Module):
#     def __init__(self, embed_dim, max_packets, dropout, vocab):
#         super().__init__()
#         self.packet_pos_embed = nn.Embedding(max_packets, embed_dim)
#         self.direction_embed = nn.Embedding(2, embed_dim)
#         self.drop = nn.Dropout(dropout)
#         self.vocab = vocab

#     def forward(self, cls_packet_embeddings, direction):
#         ''' Args : 
#                    cls_packet_embedding : 3-D tenosr with first dim being batch size (total no. of packets)
#                    direction : a tensor which will have direction of each packet 1 or 2 (size: num_packets X 1)

#                    it will take the input from encode_flow function in the tokenizer file'''

#         # defining the cls token with its embedding
#         '''clsf_embedding = nn.Embedding(1, embedding_dim=128)
#         clsf_token_embedding = clsf_embedding(torch.zeros(
#             cls_packet_embeddings.size(0), dtype=torch.long))

#         # Extract [CLS_p] token embeddings
#         cls_packet_embeddings = cls_packet_embeddings[:, 0, :]

#         # then adding it to cls_packet_embedding
#         cls_packet_embeddings = torch.cat(
#             [clsf_token_embedding, cls_packet_embeddings], dim=1)'''
#         # print(self.vocab.keys())
#         clsf_token_index = self.vocab['[CLSf]']
#         print("indcies: ", clsf_token_index)
#         num_packets,  = cls_packet_embeddings.size()
#         packet_pos = torch.tensor([i for i in range(num_packets)])

#         packet_pos_emb = self.packet_pos_embed(
#             packet_pos).unsqueeze(0).repeat(num_packets, 1)
#         direction_emb = self.direction_embed(
#             direction).unsqueeze(0).repeat(num_packets, 1)

#         # clsf token does not have the 1x128 embedding
#         # cls_packet_embeddings = torch.cat(
#         #   [torch.tensor([self.vocab['[CLSF]']]), cls_packet_embeddings], dim=0).unsqueeze(0)

#         embed_val = cls_packet_embeddings + packet_pos_emb + direction_emb
#         embed_val = self.drop(embed_val)

#         return embed_val.squeeze(0)


# vocab = {}
# with open(r"C:\Users\Kavish\OneDrive\Desktop\SPARKLE\Custom-Vocab.txt", 'r', encoding='utf-8') as f:
#     for line in f:
#         token, token_id = line.strip().split('\t')
#         vocab[token] = int(token_id)


# vocab = {'[CLSF]': 1, '[PAD]': 0, '[UNK]': 2, '[CLSp]': 3, '[SEP]': 4}

# # Example usage
# max_len = 10  # Maximum packet length
# max_packets = 5  # Maximum number of packets in a flow

# # Calculate relative positions for token, field, and header
# token_pos = torch.tensor([i for i in range(max_len)])
# # Assuming 4 fields per packet
# field_pos = torch.tensor([i // 4 for i in range(max_len)])
# # Assuming 2 headers per packet
# header_pos = torch.tensor([i // 2 for i in range(max_len)])

# # Calculate relative positions for packets in a flow
# packet_pos = torch.tensor([i for i in range(max_packets)])

# # Create instances of the embedding layers
# packet_embedding = PacketEmbedding(
#     vocab_size=256, max_len=max_len, embed_dim=128, dropout=0.1)
# flow_embedding = FlowEmbedding(
#     embed_dim=128, max_packets=max_packets, dropout=0.1, vocab=vocab)

# # Example input tensors
# token_ids = torch.randint(0, 256, (max_len,))
# direction = torch.tensor([2, 1, 2, 1, 2])  # Example direction tensor

# # Get packet embeddings
# packet_embeddings = packet_embedding(
#     token_ids, token_pos, field_pos, header_pos)

# print(packet_embeddings.shape)

# Get flow embeddings
# flow_embeddings = flow_embedding(packet_embeddings, direction)
