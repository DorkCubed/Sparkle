import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from embedding_new import PacketEmbedding

class PacketLevelEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, num_heads, num_layers, dropout):
        super(PacketLevelEncoder, self).__init__()

        # initialise the embedding layer
        self.embedding = PacketEmbedding(
            vocab_size, max_len, embed_dim, dropout)
        # self.embedding_span = PacketEmbedding(
        #     vocab_size, max_len, embed_dim*3, dropout) # for sfbo 
        # initialsise the encoder from PyTorch
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim * 4, dropout)
        
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        # for sfbo 
        # self.encoder_layer_span = nn.TransformerEncoderLayer(
        #     embed_dim*3, num_heads, embed_dim * 6, dropout)
        # self.encoder_span = nn.TransformerEncoder(self.encoder_layer_span, num_layers)

        # initialise the mlm and sfbo predictor
        self.mlm_predictor = nn.Linear(embed_dim, vocab_size)
        self.sfbo_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, vocab_size)
        )
        # self.sfbo_predictor = nn.Sequential(
        #     nn.Linear(embed_dim*3, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, vocab_size)
        # )
    def forward(self, packet_sequences, field_pos, header_pos):

        # splits the seq into masked seqs
        # packet_sequences = packet_sequences.squeeze(0)
        # print("packet: ", packet_sequences.shape)
        # packet_sequences = packet_sequences.float()
        device = packet_sequences.device
        masked_packets, span_masks = apply_mlm_sfbo_masking(packet_sequences, field_pos)
        # print("masked: ", masked_packets.device, span_masks.device)
        # masked_packets = masked_packets.long() # can we change the long to float here? 
        # mask_packet_embeddings = self.embedding(masked_packets, field_pos, header_pos)
        # print(mask_packet_embeddings.shape)
        # masked_packets_val, token_emb, token_pos_emb, field_pos_emb, header_pos_emb = self.embedding(masked_packets.to(device))
        # mask_encoded_packets = self.encoder(masked_packets_val, field_pos, header_pos).squeeze(0)
        mask_encoded_packets = self.encoder(self.embedding(masked_packets.to(device), field_pos, header_pos).squeeze(0))

        # span_masks = span_masks.long()
        # span_packet_embeddings = self.embedding(span_masks, field_pos, header_pos)
        # span_packets_val, token_emb, token_pos_emb, field_pos_emb, header_pos_emb = self.embedding(span_masks.to(device))
        # span_encoded_packets = self.encoder(span_packets_val, field_pos, header_pos).squeeze(0)

        span_encoded_packets = self.encoder(self.embedding(span_masks.to(device), field_pos, header_pos).squeeze(0))
        mlm_loss = self.compute_mlm_loss(mask_encoded_packets, masked_packets.to(device))
        sfbo_loss = self.compute_sfbo_loss(span_encoded_packets, span_masks.to(device), packet_sequences)
        # stack_encoded_packets = torch.stack((mask_encoded_packets, span_encoded_packets))
        mean_encoded_packets = torch.mean(torch.stack((mask_encoded_packets, span_encoded_packets)), dim=0)
        # print("packet enc devices: ", mlm_loss.device, sfbo_loss.device, mask_encoded_packets.device,
        #       masked_packets.device, mean_encoded_packets.device)
        # del masked_packets, span_masks
        torch.cuda.empty_cache()
        return mlm_loss, sfbo_loss, mean_encoded_packets
    
    # def forward(self, packet_sequences, field_pos, header_pos):

    #     # splits the seq into masked seqs
    #     # packet_sequences = packet_sequences.squeeze(0)
    #     # print("packet: ", packet_sequences.shape)
    #     # packet_sequences = packet_sequences.float()
    #     masked_packets, span_masks = apply_mlm_sfbo_masking(packet_sequences, field_pos)
    #     # print("masked: ", masked_packets.device, span_masks.device)
    #     # masked_packets = masked_packets.long() # can we change the long to float here? 
    #     # mask_packet_embeddings = self.embedding(masked_packets, field_pos, header_pos)
    #     # print(mask_packet_embeddings.shape)
    #     mask_encoded_packets = self.encoder(self.embedding(masked_packets, field_pos, header_pos).squeeze(0))

    #     # span_masks = span_masks.long()
    #     # span_packet_embeddings = self.embedding(span_masks, field_pos, header_pos)
    #     span_encoded_packets = self.encoder(self.embedding(span_masks, field_pos, header_pos).squeeze(0))
    #     mlm_loss = self.compute_mlm_loss(mask_encoded_packets, masked_packets)
    #     sfbo_loss = self.compute_sfbo_loss(span_encoded_packets, span_masks, packet_sequences)
    #     # stack_encoded_packets = torch.stack((mask_encoded_packets, span_encoded_packets))
    #     mean_encoded_packets = torch.mean(torch.stack((mask_encoded_packets, span_encoded_packets)), dim=0)
    #     # print("packet enc devices: ", mlm_loss.device, sfbo_loss.device, mask_encoded_packets.device,
    #     #       masked_packets.device, mean_encoded_packets.device)
    #     return mlm_loss, sfbo_loss, mean_encoded_packets

    def compute_mlm_loss(self, encoded_packets, masked_packets):
        mlm_logits = self.mlm_predictor(encoded_packets)
        # print("mlm logits: ", mlm_logits.shape)
        # print(mlm_logits.view(-1, mlm_logits.size(-1)).shape, (masked_packets.view(-1)).shape)
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)), masked_packets.view(-1))
        return mlm_loss

    def compute_sfbo_loss(self, span_encoded_packets, span_masks, packet_sequences):
        sfbo_loss = 0
        sfbo_logits = self.sfbo_predictor(span_encoded_packets)
        # print("sfbo log: ", sfbo_logits.shape)
        flat_logits = sfbo_logits.view(-1, sfbo_logits.size(-1))
        # print("flat logits: ", flat_logits.shape)
        flat_targets = span_masks.view(-1) 
        sfbo_loss += F.cross_entropy(flat_logits, flat_targets)

        return sfbo_loss

def apply_mlm_sfbo_masking(packet_sequences, field_pos, mlm_prob=0.15, sfbo_prob=0.15, max_span_length=6):
    span_masks = []
    masked_sequences = []

    # Unbind field_pos for processing
    field_pos = field_pos.unbind(0)

    for packet_seq, field_pos_seq in zip(packet_sequences, field_pos):
        # Perform masking on CPU
        masked_seq = apply_mlm_masking(packet_seq, mlm_prob).cpu()
        span_mask = apply_sfbo_masking(packet_seq, field_pos_seq, max_span_length=max_span_length, padding_value=4).cpu()
        
        # Collect results
        masked_sequences.append(masked_seq)
        span_masks.append(span_mask)

    # After the loop, move lists to GPU and stack tensors
    # masked_sequences_tensor = torch.stack(masked_sequences).to(packet_sequences.device)
    # span_masks_tensor = torch.stack(span_masks).to(packet_sequences.device)

    masked_sequences_tensor = torch.stack(masked_sequences)
    span_masks_tensor = torch.stack(span_masks)

    return masked_sequences_tensor, span_masks_tensor


def apply_mlm_masking(packet_seq, mlm_prob):
    # Perform operations on the CPU
    packet_seq_cpu = packet_seq.cpu()
    masked_sequences = packet_seq_cpu.clone()
    valid_mask = packet_seq_cpu != 0
    valid_len = len(valid_mask)
    mlm_mask = (torch.rand(valid_len) < mlm_prob) & valid_mask
    masked_sequences = masked_sequences.masked_fill(mlm_mask, 4)

    # Return the masked sequence on the CPU
    return masked_sequences


def apply_sfbo_masking(packet_seq, field_pos, max_span_length, padding_value, sfbo_prob=0.15):
    # Move tensors to CPU for processing
    packet_seq_cpu = packet_seq.cpu()
    field_pos_cpu = field_pos.cpu()

    # Clone to avoid in-place modification
    masked_packet_seq = packet_seq_cpu.clone()

    # Get unique fields (excluding zeros)
    unique_fields = torch.unique(field_pos_cpu[field_pos_cpu != 0]).tolist()

    # Randomly select number of spans and validate bounds
    num_spans = random.randint(1, max_span_length)
    if num_spans > len(unique_fields):
        raise ValueError("num_spans exceeds the length of unique_fields list")

    # Select a random span of unique fields
    start_index = random.randint(0, len(unique_fields) - num_spans)
    span_selection = unique_fields[start_index:start_index + num_spans]

    # Flatten `field_pos` to make it 1-dimensional if needed
    field_pos_flat = field_pos_cpu.view(-1) if field_pos_cpu.dim() > 1 else field_pos_cpu

    # Identify positions of the selected spans
    span_positions = [i for i, value in enumerate(field_pos_flat) if value.item() in span_selection]

    # Filter span_positions to ensure they are within bounds of `packet_seq`
    span_positions = [idx for idx in span_positions if idx < len(packet_seq_cpu)]

    # Apply masking with random probability
    random_mask = torch.rand(len(span_positions))
    sfbo_mask = random_mask < sfbo_prob

    for idx, mask in zip(span_positions, sfbo_mask):
        if mask:  
            masked_packet_seq[idx] = padding_value  # Apply the padding value

    # Return the masked sequence on the CPU
    return masked_packet_seq


# def apply_mlm_sfbo_masking(packet_sequences, field_pos, mlm_prob=0.15, sfbo_prob=0.15, max_span_length=6):
#     # Implementation of MLM and SFBO masking
#     """
#     Applies MLM and SFBO masking to the input packet sequences.

#     Args:
#         packet_sequences (List[torch.Tensor]): List of packet sequences represented as tensors.
#         mlm_prob (float, optional): Probability of masking a token for MLM. Default is 0.15.
#         sfbo_prob (float, optional): Probability of masking a span for SFBO. Default is 0.15.
#         max_span_length (int, optional): Maximum length of a span for SFBO. Default is 6.

#     Returns:
#         Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
#             List of masked packet sequences and a list of tuples containing
#             (start token indices, end token indices, span token indices) for SFBO.
#     """

#     span_masks = []
#     masked_sequences = []
#     # print(packet_sequences.shape)
#     # print(field_pos.shape)
#     # field_pos = field_pos.squeeze(0).unbind(0)  
#     field_pos = field_pos.unbind(0)  
#     # print("packet_sequences:", len(packet_sequences), [p.shape for p in packet_sequences])
#     # print("field_pos:", len(field_pos), [f.shape for f in field_pos])

#     for packet_seq, field_pos_seq in zip(packet_sequences, field_pos):
#         # print(packet_seq.shape) 
#         masked_seq = apply_mlm_masking(packet_seq, mlm_prob)
#         masked_sequences.append(masked_seq)
#         span_mask = apply_sfbo_masking(packet_seq, field_pos_seq, max_span_length=6, padding_value=4)
#         span_masks.append(span_mask)
    
#     # print(len(span_masks), len(masked_sequences))
#     span_masks_tensor = torch.stack(span_masks)
#     masked_sequences_tensor = torch.stack(masked_sequences)
#     # print("span masks: ", span_masks_tensor.shape)
#     # print("type: ", masked_sequences_tensor.shape)
#     return masked_sequences_tensor , span_masks_tensor # torch.tensor(span_masks)

# # def apply_mlm_masking(packet_seq, mlm_prob):
# #     masked_sequences = packet_seq.clone()
# #     valid_mask = packet_seq != 0 
# #     valid_len = len(valid_mask)
# #     device = valid_mask.device
# #     mlm_mask = (torch.rand(valid_len, device=device) < mlm_prob) & valid_mask
# #     # masked_sequences[mlm_mask] = torch.tensor(4, dtype=packet_seq.dtype, device=device)
# #     masked_sequences = masked_sequences.masked_fill(mlm_mask, torch.tensor(4, dtype=packet_seq.dtype, device=device))

# #     return masked_sequences

# def apply_mlm_masking(packet_seq, mlm_prob):
#     # Move the input sequence to the CPU
#     packet_seq_cpu = packet_seq.cpu()

#     # Perform operations on the CPU
#     masked_sequences = packet_seq_cpu.clone()
#     valid_mask = packet_seq_cpu != 0 
#     valid_len = len(valid_mask)
#     mlm_mask = (torch.rand(valid_len) < mlm_prob) & valid_mask
#     masked_sequences = masked_sequences.masked_fill(mlm_mask, 4)

#     # Return the final masked sequence back to the GPU
#     return masked_sequences.to(packet_seq.device)

# def apply_sfbo_masking(packet_seq, field_pos, max_span_length, padding_value, sfbo_prob=0.15):
#     # Move tensors to CPU for processing
#     packet_seq_cpu = packet_seq.cpu()
#     field_pos_cpu = field_pos.cpu()

#     # Clone to avoid in-place modification
#     masked_packet_seq = packet_seq_cpu.clone()

#     # Get unique fields (excluding zeros)
#     unique_fields = torch.unique(field_pos_cpu[field_pos_cpu != 0]).tolist()

#     # Randomly select number of spans and validate bounds
#     num_spans = random.randint(1, max_span_length)
#     if num_spans > len(unique_fields):
#         raise ValueError("num_spans exceeds the length of unique_fields list")

#     # Select a random span of unique fields
#     start_index = random.randint(0, len(unique_fields) - num_spans)
#     span_selection = unique_fields[start_index:start_index + num_spans]

#     # Flatten `field_pos` to make it 1-dimensional if needed
#     field_pos_flat = field_pos_cpu.view(-1) if field_pos_cpu.dim() > 1 else field_pos_cpu

#     # Identify positions of the selected spans
#     span_positions = [i for i, value in enumerate(field_pos_flat) if value.item() in span_selection]

#     # Filter span_positions to ensure they are within bounds of `packet_seq`
#     span_positions = [idx for idx in span_positions if idx < len(packet_seq_cpu)]

#     # Apply masking with random probability
#     random_mask = torch.rand(len(span_positions))
#     sfbo_mask = random_mask < sfbo_prob

#     for idx, mask in zip(span_positions, sfbo_mask):
#         if mask:  
#             masked_packet_seq[idx] = padding_value  # Apply the padding value

#     # Return the masked sequence back on the original device
#     return masked_packet_seq.to(packet_seq.device)


# def apply_sfbo_masking(packet_seq, field_pos, max_span_length, padding_value, sfbo_prob=0.15):
#     masked_packet_seq = packet_seq.clone()  # Clone to avoid in-place modification
#     # valid_indices_field = field_pos[field_pos != 0]
#     unique_fields = torch.unique(field_pos[field_pos != 0]).tolist()
#     num_spans = random.randint(1, max_span_length)
    
#     if num_spans > len(unique_fields):
#         raise ValueError("num_spans exceeds the length of unique_elements list")
    
#     start_index = random.randint(0, len(unique_fields) - num_spans)
#     span_selection = unique_fields[start_index:start_index + num_spans]

#     # Flatten `field_pos` to make it 1-dimensional if needed
#     field_pos_flat = field_pos.view(-1) if field_pos.dim() > 1 else field_pos
#     span_positions = [i for i, value in enumerate(field_pos_flat) if value.item() in span_selection]

#     # Filter span_positions to ensure they are within bounds of `packet_seq`
#     span_positions = [idx for idx in span_positions if idx < len(packet_seq)]
    
#     random_mask = torch.rand(len(span_positions))
#     sfbo_mask = random_mask < sfbo_prob
    
#     for idx, mask in zip(span_positions, sfbo_mask):
#         if mask:  
#             masked_packet_seq[idx] = padding_value  # Apply the padding value where required

#     return masked_packet_seq




# def apply_sfbo_masking(packet_seq, field_pos, max_span_length, padding_value, sfbo_prob=0.15):
#     masked_packet_seq = packet_seq.clone()  # Clone to avoid in-place modification
#     valid_indices_field = field_pos[field_pos != 0]
#     unique_fields = torch.unique(valid_indices_field).tolist()
#     num_spans = random.randint(1, max_span_length)
    
#     if num_spans > len(unique_fields):
#         raise ValueError("num_spans exceeds the length of unique_elements list")
    
#     start_index = random.randint(0, len(unique_fields) - num_spans)
#     span_selection = unique_fields[start_index:start_index + num_spans]
#     span_positions = [i for i, value in enumerate(field_pos) if value in span_selection]
#     random_mask = torch.rand(len(span_positions))
#     sfbo_mask = random_mask < sfbo_prob
    
#     for idx, mask in zip(span_positions, sfbo_mask):
#         if mask:  
#             masked_packet_seq[idx] = padding_value  # This needs to be carefully handled if it remains in-place

#     return masked_packet_seq.clone().detach()

