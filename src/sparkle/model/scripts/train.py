import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sparkle.configs.config import Config
from sparkle.data_loader.data_loader import DataModule
from sparkle.model.embedding import PacketEmbedding, FlowEmbedding
from sparkle.model.flow_encoder import FlowLevelEncoder
from sparkle.model.packet_encoder import PacketLevelEncoder

# Configure logging
os.makedirs('logs', exist_ok=True)
log_file = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PacketLevelTrainer:
    def __init__(self, packet_embedding, packet_encoder, flow_embedding, flow_encoder):
        data_module = DataModule()
        self.config = Config()
        self.vocab = self._init_vocab()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skipped = 0

        self.packet_embedding = packet_embedding.to(self.device)
        self.packet_encoder = packet_encoder.to(self.device)
        self.flow_embedding = flow_embedding.to(self.device)
        self.flow_encoder = flow_encoder.to(self.device)

        self.train_loader = data_module.get_loader()

        self.optimizer = optim.Adam(
            list(self.packet_embedding.parameters()) + list(self.flow_embedding.parameters()) + list(
                self.packet_encoder.parameters()) + list(self.flow_encoder.parameters()), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.step_successful = False

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

    @staticmethod
    def safe_prepare(tensor, name, device):
        if tensor is None:
            raise ValueError(f"{name} is None")

        if not hasattr(tensor, "to"):
            raise TypeError(f"{name} is not a tensor-like object")

        if tensor.dim() > 1 and tensor.size(0) == 1:
            tensor = tensor[0]

        try:
            tensor = tensor.squeeze(0)
        except Exception as e:
            raise RuntimeError(f"Failed to squeeze {name}: {e}") from e

        try:
            tensor = tensor.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to move {name} to device {device}: {e}") from e

        return tensor



    # TODO (done): fix this to be compatible with manifest.json logic
    # TODO (test)
    def process_encodings(self, encodings, entry):
        try:
            logger.info(f"Starting process encoding for {entry}")

            if not encodings:
                logger.error("process_encodings called with empty encodings list.")
                return None

            try:
                final_packet_encodings = torch.cat(encodings, dim=0).to(self.device)
            except Exception as e:
                logger.exception(f"Failed concatenating encodings: {e}")
                return None

            logger.info(f"Final concatenated shape: {final_packet_encodings.shape}")

            direction_file_path = entry.get("direction")
            if direction_file_path is None:
                logger.error("Entry missing required key 'direction'.")
                return None

            try:
                with open(direction_file_path, 'r', encoding="utf-8") as file:
                    direction_data = [int(line.strip()) for line in file.readlines()]
            except Exception as e:
                logger.exception(f"Failed reading direction file {direction_file_path}: {e}")
                self.skipped += 1
                return None

            try:
                direction_tensor = torch.tensor(direction_data, device=self.device)
            except Exception as e:
                logger.exception(f"Failed creating direction tensor: {e}")
                return None

            try:
                flow_embeddings, pad_indices = self.flow_embedding(final_packet_encodings, direction_tensor)
            except Exception as e:
                logger.exception(f"FlowEmbedding forward pass failed: {e}")
                return None

            logger.info(f"Flow embeddings computed for packet: {direction_file_path}")

            try:
                flow_encoding, mpm_loss = self.flow_encoder(flow_embeddings, pad_indices)
            except Exception as e:
                logger.exception(f"FlowEncoder forward pass failed: {e}")
                return None

            try:
                mpm_loss_tensor = mpm_loss[0].to(self.device)
            except Exception as e:
                logger.exception(f"Failed extracting MPM loss tensor: {e}")
                return None

            return mpm_loss_tensor

        except Exception as e:
            logger.exception(f"Unexpected error inside process_encodings: {e}")
            return None

    def backward_and_optimize(self, accumulated_mlm_loss, accumulated_sfbo_loss):

        total_accumulated_loss = accumulated_mlm_loss + accumulated_sfbo_loss

        self.optimizer.zero_grad()
        total_accumulated_loss.backward()
        self.optimizer.step()
        self.accumulated_mlm_loss = 0.0
        self.accumulated_sfbo_loss = 0.0
        self.batch_counter = 0

    def train_epoch(self, epoch):
        logger.info(f"\n{'=' * 30}")
        logger.info(f"Starting training epoch {epoch + 1}")
        logger.info(f"{'=' * 30}")

        progress_bar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}",
            leave=False
        )

        for i, (packet_sequences, field_position, header_position, entry) in enumerate(progress_bar):
            # try:
            #     entry = {k: (v[0] if isinstance(v, list) else v) for k, v in entry.items()}
            #     packet_sequences = packet_sequences.squeeze(0).to(self.device)
            #     field_position = field_position.squeeze(0).to(self.device)
            #     header_position = header_position.squeeze(0).to(self.device)
            #
            #     current_packet_file = entry.get("packet")
            #     if current_packet_file is None:
            #         logger.error("Missing required entry['packet'] in batch, skipping.")
            #         continue
            # except Exception as e:
            #     logger.exception(f"Unexpected error inside batch {i}: {e}")
            #     continue

            try:
                vocab_limit = self.config.vocab_size
                max_idx = packet_sequences.max().item()
                min_idx = packet_sequences.min().item()

                if max_idx >= vocab_limit or min_idx < 0:
                    logger.error(
                        f"SKIPPING BATCH {i}: Found invalid index {max_idx} (Max allowed: {vocab_limit - 1}) in file {entry.get('packet')}")
                    self.skipped += 1
                    progress_bar.set_postfix({"skipped": self.skipped})
                    continue  # Skip safely, GPU is still healthy
            except Exception as check_e:
                logger.error(f"Error during index validation: {check_e}")
                self.skipped += 1
                continue

            try:
                entry = {k: (v[0] if isinstance(v, list) else v) for k, v in entry.items()}
                packet_sequences = self.safe_prepare(packet_sequences, "packet_sequences", self.device)
                field_position = self.safe_prepare(field_position, "field_position", self.device)
                header_position = self.safe_prepare(header_position, "header_position", self.device)

                current_packet_file = entry.get("packet")
                if current_packet_file is None:
                    logger.error("Missing required entry['packet'] in batch, skipping.")
                    continue
            except Exception as e:
                logger.exception(f"Unexpected error inside batch {i}: {e}")
                self.skipped += 1
                continue
            # Handle file transitions safely
            try:
                if self.previous_entry is not None and current_packet_file != self.previous_packet_file:
                    if self.all_packet_encodings:
                        logger.info(f"Completed processing file: {self.previous_entry['packet']}")
                        mpm_loss = self.process_encodings(self.all_packet_encodings, self.previous_entry)
                        if mpm_loss is not None:
                            self.optimizer.zero_grad()
                            mpm_loss.backward()
                            self.optimizer.step()

                    # reset for new file
                    self.all_packet_encodings = []
                    self.total_packet_enc_loss = 0
                    logger.info(f"Starting new file: {current_packet_file}")

                self.previous_packet_file = current_packet_file
            except Exception as e:
                logger.exception(f"Error during file-boundary logic: {e}")
                self.skipped += 1

            try:
                # Packet-level forward pass
                mlm_loss, sfbo_loss, encoded_packets_mean = self.packet_encoder(
                    packet_sequences, field_pos=field_position, header_pos=header_position
                )

                self.accumulated_mlm_loss += mlm_loss
                self.accumulated_sfbo_loss += sfbo_loss
                self.batch_counter += 1

                if self.batch_counter == self.accumulation_steps:
                    self.backward_and_optimize(self.accumulated_mlm_loss, self.accumulated_sfbo_loss)

                self.all_packet_encodings.append(encoded_packets_mean.detach())
                self.step_successful = True
            except Exception as e:
                logger.exception(f"Loss accumulation error inside batch {i}: {e}")
                self.skipped += 1

            self.previous_entry = entry

            progress_bar.set_postfix({
                "skipped": self.skipped
            })

        logger.info("Reached last file.")
        logger.info(f"self.previous_entry: {self.previous_entry}")

        # Final file after loop
        try:
            if self.all_packet_encodings and self.previous_entry["packet"]:
                logger.info(f"Final processing for last flow packet {self.previous_packet_file}")
                final_loss = self.process_encodings(self.all_packet_encodings, self.previous_entry)
                if final_loss is not None:
                    self.optimizer.zero_grad()
                    final_loss.backward()
                    self.optimizer.step()
        except Exception as e:
            logger.exception(f"Final mpm_loss computation failed: {e}")

    def save_checkpoint(self, epoch, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pth")

        state = {
            "epoch": epoch,
            "packet_embedding": self.packet_embedding.state_dict(),
            "packet_encoder": self.packet_encoder.state_dict(),
            "flow_embedding": self.flow_embedding.state_dict(),
            "flow_encoder": self.flow_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "accumulated_mlm_loss": self.accumulated_mlm_loss,
            "accumulated_sfbo_loss": self.accumulated_sfbo_loss,
        }
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.packet_embedding.load_state_dict(checkpoint["packet_embedding"])
        self.packet_encoder.load_state_dict(checkpoint["packet_encoder"])
        self.flow_embedding.load_state_dict(checkpoint["flow_embedding"])
        self.flow_encoder.load_state_dict(checkpoint["flow_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.accumulated_mlm_loss = checkpoint.get("accumulated_mlm_loss", 0.0)
        self.accumulated_sfbo_loss = checkpoint.get("accumulated_sfbo_loss", 0.0)

        logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {checkpoint['epoch'] + 1})")
        return checkpoint["epoch"]


class ExperimentRunner:
    def __init__(self):
        logger.info("Initializing ExperimentRunner...")
        self.config = Config()
        data_module = DataModule()
        self.tokenizer = data_module.get_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")

    def load_vocab(self):
        vocab = {}
        with open(self.config.tokenizer_path, 'r', encoding='utf-8') as f:
            for line in f:
                token, token_id = line.strip().split('\t')
                vocab[token] = int(token_id)
        return vocab

    def run(self):
        # print("-" * 30)
        # print(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', 'Not Set')}")
        # print(f"TORCH_USE_CUDA_DSA:   {os.environ.get('TORCH_USE_CUDA_DSA', 'Not Set')}")
        # print("-" * 30)

        logger.info("Starting model training...")
        vocab = self.load_vocab()
        logger.info(f"Vocabulary size: {len(vocab)}")

        # print("Loading embeddings.")
        packet_embedding = PacketEmbedding(self.config.vocab_size, max_len=self.config.max_len,
                                           embed_dim=self.config.embed_dim, dropout=self.config.dropout).to(self.device)
        # print("Loaded packet embeddings.")
        packet_encoder = PacketLevelEncoder(self.config.vocab_size, self.config.embed_dim, self.config.max_len,
                                            self.config.num_heads, self.config.num_layers, self.config.dropout).to(
            self.device)
        # print("Loaded packet encoder.")
        flow_embedding = FlowEmbedding(self.config.embed_dim, self.config.max_flow_length, self.config.dropout,
                                       vocab).to(self.device)
        # print("Loaded flow embeddings.")
        flow_encoder = FlowLevelEncoder(self.config.embed_dim, self.config.num_layers, self.config.num_heads,
                                        self.config.dropout, vocab, self.config.max_flow_length,
                                        self.config.mask_prob).to(self.device)
        # print("Loaded flow encoder.")
        # print("Loaded embeddings.")

        trainer = PacketLevelTrainer(packet_embedding, packet_encoder, flow_embedding, flow_encoder)

        print("Loaded trainer.")

        for epoch in range(self.config.num_epochs):
            trainer.train_epoch(epoch)
            trainer.save_checkpoint(epoch)


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()
