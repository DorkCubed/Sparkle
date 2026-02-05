import torch.nn as nn
from embedding import PacketEmbedding, FlowEmbedding
from packet_encoder import PacketLevelEncoder
from flow_encoder import FlowLevelEncoder
from Input_Tokenizer import Tokenizer


class BERTModel(nn.Module):
    def __init__(self, vocab_size, vocab, embed_dim, num_heads, num_layers, dropout, max_flow_length, mask_prob):
        super().__init__()
        self.packet_embedding = PacketEmbedding(
            vocab_size, max_len=512, embed_dim=embed_dim, dropout=dropout)
        self.packet_encoder = PacketLevelEncoder(
            vocab_size, embed_dim, num_heads, num_layers, dropout)
        self.flow_embedding = FlowEmbedding(
            embed_dim, max_flow_length, dropout, vocab)
        self.flow_encoder = FlowLevelEncoder(
            embed_dim, num_layers, num_heads, dropout, max_flow_length, mask_prob)
        self.tokenizer = Tokenizer(vocab)

    def forward(self, packet_sequences, field_pos, header_pos, direction):
        packet_embeddings = self.packet_embedding(
            token_ids=packet_sequences, field_pos=field_pos, header_pos=header_pos)
        mlm_loss, sfbo_loss, packet_encodings = self.packet_encoder(
            packet_embeddings)

        cls_packet_embedding = self.tokenizer.encode_flow(packet_encodings)
        flow_embeddings = self.flow_embedding(
            cls_packet_embedding, direction)
        flow_encodings, mpm_losses = self.flow_encoder(flow_embeddings)

        total_loss = mlm_loss + sfbo_loss + sum(mpm_losses)

        return total_loss
