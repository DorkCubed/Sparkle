import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sparkle.model.embedding import PacketEmbedding


class PacketLevelEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, num_heads, num_layers, dropout):
        super(PacketLevelEncoder, self).__init__()

        # initialise the embedding layer
        self.embedding = PacketEmbedding(vocab_size, max_len, embed_dim, dropout)
        # self.embedding_span = PacketEmbedding(
        #     vocab_size, max_len, embed_dim*3, dropout) # for sfbo
        # initialsise the encoder from PyTorch
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim * 4, dropout
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        # for sfbo
        # self.encoder_layer_span = nn.TransformerEncoderLayer(
        #     embed_dim*3, num_heads, embed_dim * 6, dropout)
        # self.encoder_span = nn.TransformerEncoder(self.encoder_layer_span, num_layers)

        # initialise the mlm and sfbo predictor
        self.mlm_predictor = nn.Linear(embed_dim, vocab_size)
        self.sfbo_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, vocab_size)
        )
        # self.sfbo_predictor = nn.Sequential(
        #     nn.Linear(embed_dim*3, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, vocab_size)
        # )

    def forward(self, packet_sequences, field_pos, header_pos, return_metrics=False):
        device = packet_sequences.device

        # Ensure tensors have at least 2 dimensions
        if packet_sequences.dim() == 0:
            raise ValueError(f"packet_sequences is 0-d tensor: {packet_sequences}")
        if packet_sequences.dim() == 1:
            packet_sequences = packet_sequences.unsqueeze(0)
        if field_pos.dim() == 0:
            field_pos = field_pos.unsqueeze(0)
        if field_pos.dim() == 1:
            field_pos = field_pos.unsqueeze(0)
        if header_pos.dim() == 0:
            header_pos = header_pos.unsqueeze(0)
        if header_pos.dim() == 1:
            header_pos = header_pos.unsqueeze(0)

        # ideally move this to DataLoader for speed
        masked_packets, span_masks = apply_mlm_sfbo_masking(packet_sequences, field_pos)
        masked_packets = masked_packets.to(device, non_blocking=True)
        span_masks = span_masks.to(device, non_blocking=True)

        # Track which positions were masked for accuracy calculation
        mlm_masked_positions = (masked_packets != packet_sequences) & (
            packet_sequences != 0
        )
        sfbo_masked_positions = (span_masks != packet_sequences) & (
            packet_sequences != 0
        )

        # We embed and encode for MLM
        mask_emb = self.embedding(masked_packets, field_pos, header_pos).squeeze(0)
        mask_encoded_packets = self.encoder(mask_emb)
        mlm_loss = self.compute_mlm_loss(
            mask_encoded_packets, packet_sequences, mlm_masked_positions
        )

        # We embed and encode for SFBO
        span_emb = self.embedding(span_masks, field_pos, header_pos).squeeze(0)
        span_encoded_packets = self.encoder(span_emb)
        sfbo_loss = self.compute_sfbo_loss(
            span_encoded_packets, packet_sequences, sfbo_masked_positions
        )

        # We DETACH here. This ensures this purely informational tensor
        # does not keep the computation graph alive.
        mean_encoded_packets = torch.mean(
            torch.stack((mask_encoded_packets.detach(), span_encoded_packets.detach())),
            dim=0,
        )

        if not return_metrics:
            return mlm_loss, sfbo_loss, mean_encoded_packets

        # Get predictions for accuracy calculation (eval only)
        with torch.no_grad():
            mlm_logits = self.mlm_predictor(mask_encoded_packets)
            mlm_preds = torch.argmax(mlm_logits, dim=-1)

            sfbo_logits = self.sfbo_predictor(span_encoded_packets)
            sfbo_preds = torch.argmax(sfbo_logits, dim=-1)

        return (
            mlm_loss,
            sfbo_loss,
            mean_encoded_packets,
            mlm_preds,
            packet_sequences,
            mlm_masked_positions,
            sfbo_preds,
            packet_sequences,
            sfbo_masked_positions,
        )

    def compute_mlm_loss(self, encoded_packets, packet_sequences, mlm_mask):
        mlm_logits = self.mlm_predictor(encoded_packets)

        flat_logits = mlm_logits.reshape(-1, mlm_logits.size(-1))
        flat_targets = packet_sequences.reshape(-1)
        flat_mask = mlm_mask.reshape(-1)

        masked_logits = flat_logits[flat_mask]
        masked_targets = flat_targets[flat_mask]

        loss = F.cross_entropy(masked_logits, masked_targets)
        return loss

    def compute_sfbo_loss(self, span_encoded_packets, packet_sequences, sfbo_mask):
        sfbo_logits = self.sfbo_predictor(span_encoded_packets)
        flat_logits = sfbo_logits.reshape(-1, sfbo_logits.size(-1))
        flat_targets = packet_sequences.reshape(-1)
        flat_mask = sfbo_mask.reshape(-1)

        masked_logits = flat_logits[flat_mask]
        masked_targets = flat_targets[flat_mask]

        loss = F.cross_entropy(masked_logits, masked_targets)
        return loss


def apply_mlm_sfbo_masking(
    packet_sequences, field_pos, mlm_prob=0.15, max_span_length=6
):
    span_masks = []
    masked_sequences = []

    # Handle 0-dimensional tensors by adding a batch dimension
    if packet_sequences.dim() == 0:
        packet_sequences = packet_sequences.unsqueeze(0)
    if packet_sequences.dim() == 1:
        packet_sequences = packet_sequences.unsqueeze(0)
    if field_pos.dim() == 0:
        field_pos = field_pos.unsqueeze(0)
    if field_pos.dim() == 1:
        field_pos = field_pos.unsqueeze(0)

    # Unbind field_pos for processing
    field_pos = field_pos.unbind(0)

    for packet_seq, field_pos_seq in zip(packet_sequences, field_pos):
        # Perform masking on CPU
        masked_seq = apply_mlm_masking(packet_seq, mlm_prob).cpu()
        span_mask = apply_sfbo_masking(
            packet_seq, field_pos_seq, max_span_length=max_span_length, padding_value=4
        ).cpu()

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
    # Create mask with same shape as valid_mask
    mlm_mask = (torch.rand_like(packet_seq_cpu.float()) < mlm_prob) & valid_mask
    masked_sequences = masked_sequences.masked_fill(mlm_mask, 4)

    # Return the masked sequence on the CPU
    return masked_sequences


def apply_sfbo_masking(packet_seq, field_pos, max_span_length, padding_value):
    # Move tensors to CPU for processing
    packet_seq_cpu = packet_seq.cpu()
    field_pos_cpu = field_pos.cpu()

    # Clone to avoid in-place modification
    masked_packet_seq = packet_seq_cpu.clone()

    # Get unique fields (excluding zeros)
    unique_fields = torch.unique(field_pos_cpu[field_pos_cpu != 0]).tolist()

    # Handle edge case: no valid fields
    if len(unique_fields) == 0:
        return masked_packet_seq

    # Randomly select number of spans and validate bounds
    num_spans = random.randint(1, min(max_span_length, len(unique_fields)))

    # Select a random span of unique fields
    start_index = random.randint(0, len(unique_fields) - num_spans)
    span_selection = unique_fields[start_index : start_index + num_spans]

    # Flatten `field_pos` to make it 1-dimensional if needed
    field_pos_flat = (
        field_pos_cpu.view(-1) if field_pos_cpu.dim() > 1 else field_pos_cpu
    )

    # Identify positions of the selected spans
    span_positions = [
        i for i, value in enumerate(field_pos_flat) if value.item() in span_selection
    ]

    # Filter span_positions to ensure they are within bounds of `packet_seq`
    span_positions = [idx for idx in span_positions if idx < len(packet_seq_cpu)]

    # Mask all positions in the selected field spans
    for idx in span_positions:
        masked_packet_seq[idx] = padding_value

    # Return the masked sequence on the CPU
    return masked_packet_seq
