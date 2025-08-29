import torch
import torch.nn as nn
import torch.optim as optim
from sparkle.model.embedding import PacketEmbedding, FlowEmbedding
from sparkle.model.flow_encoder import FlowLevelEncoder
from sparkle.model.packet_encoder import PacketLevelEncoder
from sparkle.data_loader.data_loader import DataModule
from sparkle.configs.config import Config

class PacketLevelTrainer:
    def __init__(self, packet_embedding, packet_encoder, flow_embedding, flow_encoder):
        data_module = DataModule()
        self.config = Config()
        self.vocab = self._init_vocab()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.packet_embedding = packet_embedding.to(self.device)
        self.packet_encoder = packet_encoder.to(self.device)
        self.flow_embedding = flow_embedding.to(self.device)
        self.flow_encoder = flow_encoder.to(self.device)

        self.train_loader = data_module.get_loader()


        self.optimizer = optim.Adam(list(self.packet_embedding.parameters()) + list(self.flow_embedding.parameters()) + list(self.packet_encoder.parameters()) + list(self.flow_encoder.parameters()), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.accumulation_steps = 5
        self.accumulated_mlm_loss = 0.0
        self.accumulated_sfbo_loss = 0.0
        self.batch_counter = 0

        # bookkeeping logic
        self.previous_packet_file = None
        self.all_packet_encodings = []
        self.previous_packet_id = None
        self.previous_entry = None
        self.total_packet_enc_loss = 0

    def _init_vocab(self):
        vocab = {}
        with open(self.config.tokenizer_path, 'r', encoding='utf-8') as f:
            for line in f:
                token, token_id = line.strip().split('\t')
                vocab[token] = int(token_id)

        return vocab

    # TODO (done): fix this to be compatible with manifest.json logic
    # TODO (test)
    def process_encodings(self, encodings, entry):
        final_packet_encodings = torch.cat(encodings, dim=0).to(self.device)
        print("Final concatenated shape:", final_packet_encodings.shape)

        direction_file_path = entry["direction"]
        with open(direction_file_path, 'r', encoding="utf-8") as file:
            direction_data = [int(line.strip()) for line in file.readlines()]

        # Convert direction data to a tensor
        direction_tensor = torch.tensor(direction_data, device=self.device)
        print(final_packet_encodings.device, direction_tensor.device)
        # Call FlowEmbedding with the accumulated packet encodings and direction data
        flow_embeddings, pad_indices = FlowEmbedding(final_packet_encodings, direction_tensor)
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

    def backward_and_optimize(self, accumulated_mlm_loss, accumulated_sfbo_loss):
        total_accumulated_loss = accumulated_mlm_loss + accumulated_sfbo_loss
        self.optimizer.zero_grad()
        total_accumulated_loss.backward()
        self.optimizer.step()
        self.accumulated_mlm_loss = 0.0
        self.accumulated_sfbo_loss = 0.0
        self.batch_counter = 0

    # TODO: test this, especially loss logic
    def train_epoch(self, epoch):
        print(f"Training epoch {epoch}")

        for i, (packet_sequences, field_position, header_position, entry) in enumerate(self.train_loader):
            packet_sequences = packet_sequences.squeeze(0).to(self.device)
            field_position = field_position.squeeze(0).to(self.device)
            header_position = header_position.squeeze(0).to(self.device)

            current_packet_file = entry["packet"]

            # finalize previous file, then switch to new file
            if self.previous_entry is not None and current_packet_file != self.previous_packet_file:
                if self.all_packet_encodings:
                    print(f"Completed file: {self.previous_entry["packet"]}")
                    mpm_loss = self.process_encodings(self.all_packet_encodings, self.previous_entry["direction"])
                    self.optimizer.zero_grad()
                    mpm_loss.backward()
                    self.optimizer.step()
                # reset
                self.all_packet_encodings, self.total_packet_enc_loss = [], 0
                self.previous_packet_id = None
                print(f"Started new file: {current_packet_file}")

            self.previous_packet_file = current_packet_file

            # forward pass (packet level)
            mlm_loss, sfbo_loss, encoded_packets_mean = self.packet_encoder(
                packet_sequences, field_pos=field_position, header_pos=header_position
            )

            # accumulate
            self.accumulated_mlm_loss += mlm_loss
            self.accumulated_sfbo_loss += sfbo_loss
            self.batch_counter += 1

            if self.batch_counter == self.accumulation_steps:
                self.backward_and_optimize(self.accumulated_mlm_loss, self.accumulated_sfbo_loss)

            self.all_packet_encodings.append(encoded_packets_mean.detach())

            # TODO: test this, does it work with manifest.json?
            # detect packet number
            if self.previous_entry and self.previous_entry["packet"] != current_packet_file:
                print(f"Flow completed for packet {self.previous_packet_id}")
                self.all_packet_encodings = []

            self.previous_entry = entry
        # handle last file after loop
        if self.all_packet_encodings and self.previous_entry["packet"]:
            print(
                f"Final processing for last flow packet {self.previous_packet_id} in file {self.previous_packet_file}")
            mpm_loss = self.process_encodings(self.all_packet_encodings, self.previous_entry)
            self.optimizer.zero_grad()
            mpm_loss.backward()
            self.optimizer.step()

class ExperimentRunner:
    def __init__(self):
        self.config = Config()
        data_module = DataModule()
        self.tokenizer = data_module.get_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_vocab(self):
        vocab = {}
        with open(self.config.tokenizer_path, 'r', encoding='utf-8') as f:
            for line in f:
                token, token_id = line.strip().split('\t')
                vocab[token] = int(token_id)
        return vocab

    def run(self):
        vocab = self.load_vocab()
        tokenizer = self.tokenizer


        packet_embedding = PacketEmbedding(self.config.vocab_size, max_len=self.config.max_len, embed_dim=self.config.embed_dim, dropout=self.config.dropout).to(self.device)
        packet_encoder = PacketLevelEncoder(self.config.vocab_size, self.config.embed_dim, self.config.max_len, self.config.num_heads, self.config.num_layers, self.config.dropout).to(self.device)
        flow_embedding = FlowEmbedding(self.config.embed_dim, self.config.max_flow_length, self.config.dropout, vocab).to(self.device)
        flow_encoder = FlowLevelEncoder(self.config.embed_dim, self.config.num_layers, self.config.num_heads, self.config.dropout, self.config.max_flow_length, self.config.mask_prob).to(self.device)

        trainer = PacketLevelTrainer(packet_embedding, packet_encoder, flow_embedding, flow_encoder)

        for epoch in range(self.config.num_epochs):
            trainer.train_epoch(epoch)

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()
