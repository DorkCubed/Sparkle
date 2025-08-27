import torch
import torch.nn as nn
import torch.optim as optim
from model.embedding import PacketEmbedding, FlowEmbedding
from model.flow_encoder import FlowLevelEncoder
from model.packet_encoder import PacketLevelEncoder
from data_loader.data_loader import DataModule
from torch.cuda.amp import GradScaler
from configs.config import Config
from model.scripts.test import flow_embedding


class HierarchicalBERTTrainer:
    def __init__(self):
        data_module = DataModule()


class PacketLevelTrainer:
    def __init__(self):
        self.config = Config()
        self.vocab = self._init_vocab()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.packet_embedding = PacketEmbedding(self.config.vocab_size, max_len=self.config.max_len, embed_dim=self.config.embed_dim, dropout=self.config.dropout).to(self.device)
        self.packet_encoder = PacketLevelEncoder(self.config.vocab_size, self.config.embed_dim, self.config.max_len, self.config.num_heads, self.config.num_layers, self.config.dropout).to(self.device)

        self.flow_embedding = FlowEmbedding(self.config.embed_dim, self.config.max_flow_length, self.config.dropout, self.vocab).to(self.device)
        self.flow_encoder = FlowLevelEncoder(self.config.embed_dim, self.config.num_layers, self.config.num_heads, self.config.dropout, self.config.max_flow_length, self.config.mask_prob).to(self.device)

        self.optimizer = optim.Adam(list(self.packet_embedding.parameters()) + list(self.flow_embedding.parameters()) + list(self.packet_encoder.parameters()) + list(self.flow_encoder.parameters()), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

    def _init_vocab(self):
        vocab = {}
        with open(self.config.tokenizer_path, 'r', encoding='utf-8') as f:
            for line in f:
                token, token_id = line.strip().split('\t')
                vocab[token] = int(token_id)

        return vocab

    def process_encodings(self, encodings, direction_file_path):
        final_packet_encodings = torch.cat(encodings, dim=0).to(self.device)
        print("Final concatenated shape:", final_packet_encodings.shape)

        with open(direction_file_path, 'r') as file:
            direction_data = [int(line.strip()) for line in file.readlines()]

        # Convert direction data to a tensor
        direction_tensor = torch.tensor(direction_data, device=self.device)
        print(final_packet_encodings.device, direction_tensor.device)
        # Call FlowEmbedding with the accumulated packet encodings and direction data
        flow_embeddings, pad_indices = flow_embedding(final_packet_encodings, direction_tensor)
        print("Flow embeddings computed for packet:", {direction_file_path})
        print("flow embeddings: ", flow_embeddings.shape)
        print("-------------------------------------------------------------")

        # Compute flow encoding and MPM loss
        flow_encoding, mpm_loss = self.flow_encoder(flow_embeddings, pad_indices)
        print("flow encoding: ", flow_encoding.shape)
        print("mpm loss: ", mpm_loss)
        print(type(mpm_loss))

        # Ensure mpm_loss is a tensor and on the same device
        mpm_loss_tensor = mpm_loss[0].to(self.device)
        print(mpm_loss_tensor)
        return mpm_loss_tensor

