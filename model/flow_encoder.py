import torch
import torch.nn as nn
import torch.nn.functional as F
# from hierarchial_transformer import BERT
import random
from embedding_new import FlowEmbedding


class FlowLevelEncoder(nn.Module):
    def __init__(self, embed_dim, n_layers, attn_heads, dropout, vocab, max_flow_length=510, mask_prob=0.15):
        super(FlowLevelEncoder, self).__init__()

        # Initialize class variables
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.max_flow_length = max_flow_length
        self.mask_prob = mask_prob
        self.vocab = vocab

        # Initialize the flow encoder block with TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=attn_heads, dim_feedforward=embed_dim * 4, dropout=dropout
        )
        self.flow_encoder_block = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Initialize the MPM predictor and similarity calculation
        self.mpm_predictor = nn.Linear(embed_dim, embed_dim)
        self.mpm_similarity = nn.CosineSimilarity(dim=-1)

        # Initialize self-attention pooling
        self.self_attn_pooling = SelfAttentionPooling(embed_dim)

    def forward(self, flow_sequences, pad_indices_list):
        flow_encodings = []
        mpm_losses = []

        # Get the device from the first encoder layer
        device = next(self.flow_encoder_block.parameters()).device
        for flow_seq, pad_indices in zip(flow_sequences, pad_indices_list):
            # Move flow_seq to the same device as the model
            # flow_seq = flow_seq.to(device)
            # Divide the flow into fractions of max_flow_length
            flow_fractions = [
                flow_seq[i:i + self.max_flow_length] 
                for i in range(0, len(flow_seq), self.max_flow_length)
            ]
            pad_fractions = [
                pad_indices[i:i + self.max_flow_length] 
                for i in range(0, len(flow_seq), self.max_flow_length)
            ]

            fraction_encodings = []
            for fraction, pad_indices_fraction in zip(flow_fractions, pad_fractions):
                # Move fraction and pad_indices_fraction to the same device
                print(len(fraction))
                print("paddings: ", len(pad_indices_fraction))
                # fraction = fraction.to(device)
                # pad_indices_fraction = pad_indices_fraction.to(device)

                # Apply MPM masking and calculate MPM loss, excluding padding tokens
                masked_fraction, mask_indices = apply_mpm_masking(
                    fraction, self.mask_prob, pad_indices_fraction
                )
                masked_fraction = masked_fraction.to(device)  # Ensure masked_fraction is on the correct device
                fraction_encoding = self.flow_encoder_block(masked_fraction)
                # masked_fraction = masked_fraction.to(fraction.device)
                print("device: ", fraction_encoding.device, fraction.device)
                mpm_loss = self.calculate_mpm_loss(
                    fraction_encoding, fraction, mask_indices
                )
                mpm_losses.append(mpm_loss)
                fraction_encodings.append(fraction_encoding)

            # Aggregate the fraction encodings using self-attention pooling
            flow_encoding = self.self_attn_pooling(fraction_encodings)
            flow_encodings.append(flow_encoding)
            flow_encodings_tensor = torch.stack(flow_encodings)  # Convert list to tensor
        return flow_encodings_tensor, mpm_losses
        # return flow_encodings, mpm_losses

    def calculate_mpm_loss(self, fraction_encoding, original_fraction, mask_indices):
        # Move original_fraction to the same device as fraction_encoding
        original_fraction = original_fraction.to(fraction_encoding.device)

        # Compute the similarity matrix
        similarity_matrix = self.mpm_similarity(fraction_encoding, original_fraction)

        # Print shape for debugging
        print(f"Similarity Matrix Shape: {similarity_matrix.shape}")

        # Check if similarity_matrix is empty or 1D
        if similarity_matrix.dim() == 0:
            print("Warning: Similarity matrix is empty.")
            return torch.tensor(0.0, device=similarity_matrix.device)  # Return a zero loss
        elif similarity_matrix.dim() == 1:
            print("Warning: Similarity matrix is 1D, reshaping to 2D.")
            similarity_matrix = similarity_matrix.unsqueeze(0)  # Reshape to 2D

        # Generate targets
        targets = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)

        # Ensure targets have the correct size for cross-entropy
        if similarity_matrix.size(0) == 0:
            return torch.tensor(0.0, device=similarity_matrix.device)  # Prevent further error

        # Compute the cross-entropy loss
        mpm_loss = F.cross_entropy(similarity_matrix, targets)
        return mpm_loss


    # def calculate_mpm_loss(self, fraction_encoding, original_fraction, mask_indices):
    #     # Compute the similarity matrix
    #     similarity_matrix = self.mpm_similarity(fraction_encoding, original_fraction)

    #     # Print shape for debugging
    #     print(f"Similarity Matrix Shape: {similarity_matrix.shape}")

    #     # Check if similarity_matrix is empty or 1D
    #     if similarity_matrix.dim() == 0:
    #         print("Warning: Similarity matrix is empty.")
    #         return torch.tensor(0.0, device=similarity_matrix.device)  # Return a zero loss
    #     elif similarity_matrix.dim() == 1:
    #         print("Warning: Similarity matrix is 1D, reshaping to 2D.")
    #         similarity_matrix = similarity_matrix.unsqueeze(0)  # Reshape to 2D

    #     # Generate targets
    #     targets = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)

    #     # Ensure targets have the correct size for cross-entropy
    #     if similarity_matrix.size(0) == 0:
    #         return torch.tensor(0.0, device=similarity_matrix.device)  # Prevent further error

    #     # Compute the cross-entropy loss
    #     mpm_loss = F.cross_entropy(similarity_matrix, targets)
    #     return mpm_loss





class SelfAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attention = nn.Linear(embed_dim, 1)
        self.weight_m = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.bias_m = nn.Parameter(torch.randn(embed_dim))
        self.weight_u = nn.Parameter(torch.randn(embed_dim))

    def forward(self, fraction_encodings):
        attention_scores = []
        for fraction_encoding in fraction_encodings:
            m = torch.tanh(torch.mm(fraction_encoding, self.weight_m) + self.bias_m)
            attention_score = torch.mm(m, self.weight_u.unsqueeze(1))  # Changed to unsqueeze(1) for correct shape
            attention_scores.append(attention_score)

        attention_scores = torch.cat(attention_scores, dim=0).squeeze(-1)  # Concatenate along the first dimension
        attention_weights = F.softmax(attention_scores, dim=0)

        flow_encoding = torch.sum(torch.stack(fraction_encodings, dim=0) * attention_weights.unsqueeze(-1), dim=0)
        return flow_encoding

def apply_mpm_masking(packet_encodings, mask_prob, pad_indices):
    """
    Applies MPM masking to packet encodings.
    Args:
        packet_encodings (Tensor): Encodings of shape [seq_length, embed_dim].
        mask_prob (float): Probability of masking eligible positions.
        pad_indices (Tensor): Boolean tensor of shape [seq_length] indicating padding positions.
    Returns:
        Tensor, Tensor: Masked encodings and indices of masked positions.
    """
    # Ensure pad_indices is boolean and on the same device as packet_encodings
    pad_indices = pad_indices.bool().to(packet_encodings.device)

    # Ensure the sizes match
    assert pad_indices.size(0) == packet_encodings.size(0), \
        "pad_indices and packet_encodings must have the same length."

    # Identify eligible indices (non-padding positions)
    eligible_indices = (~pad_indices).nonzero(as_tuple=True)[0]

    if eligible_indices.numel() == 0:
        # Return original encodings if no eligible positions
        return packet_encodings, torch.tensor([], device=packet_encodings.device)

    # Randomly decide which eligible indices to mask
    mask_decision = torch.bernoulli(torch.full((eligible_indices.size(0),), mask_prob, device=packet_encodings.device))
    mask_indices = eligible_indices[mask_decision.bool()]

    if mask_indices.numel() == 0:
        # Return original encodings if no positions are selected for masking
        return packet_encodings, mask_indices

    # Clone encodings and apply masking
    masked_packets = packet_encodings.clone()
    masked_packets[mask_indices] = torch.randn_like(masked_packets[mask_indices])

    return masked_packets, mask_indices


# def apply_mpm_masking(packet_encodings, mask_prob, pad_indices):
#     # Ensure we only consider non-padding positions for masking
#     eligible_indices = (~pad_indices).nonzero().squeeze()

#     # Generate a mask for eligible positions based on the mask probability
#     mask_decision = torch.bernoulli(torch.full((eligible_indices.size(0),), mask_prob))
#     mask_indices = eligible_indices[mask_decision.bool()].squeeze()

#     # Clone packet encodings and apply masking
#     masked_packets = packet_encodings.clone()
#     masked_packets[mask_indices] = torch.randn_like(masked_packets[mask_indices])

#     return masked_packets, mask_indices



# VOCAB_SIZE = 256
# N_SEGMENTS = 2
# MAX_LEN = 512
# EMBED_DIM = 768
# N_LAYERS = 12
# ATTN_HEADS = 12
# DROPOUT = 0.1

# vocab = {}
# with open(r"C:\Users\Kavish\OneDrive\Desktop\SPARKLE\Custom-Vocab.txt", 'r', encoding='utf-8') as f:
#     for line in f:
#         token, token_id = line.strip().split('\t')
#         vocab[token] = int(token_id)


# packet_encoder = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN,
#                       EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)
# flow_sequences = [torch.randn(800, 768), torch.randn(1200, 768)]

# # Initialize the flow level encoder
# flow_encoder = FlowLevelEncoder(
#     packet_encoder, embed_dim=768, n_layers=2, attn_heads=8, dropout=0.1)

# # Forward pass through the flow level encoder
# flow_encodings, mpm_losses = flow_encoder(flow_sequences)

# # Compute the final loss
# final_loss = sum(mpm_losses)
