import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ignite.engine import Engine, Events

from sparkle.configs.config import Config
from sparkle.data_loader.data_loader import DataModule
from sparkle.utils import get_project_root
from sparkle.model.embedding import FlowEmbedding
from sparkle.model.flow_encoder import FlowLevelEncoder
from sparkle.model.packet_encoder import PacketLevelEncoder

os.makedirs("logs", exist_ok=True)
log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

TRAIN_BASE_PATH = "~/sparkle-datavol/local"


class PacketLevelTrainer:
    def __init__(self, packet_encoder, flow_embedding, flow_encoder, manifest_path=None, num_epochs=None):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if manifest_path:
            self.config.manifest_path = manifest_path
        else:
            self.config.manifest_path = os.path.join(get_project_root(), "manifest", "manifest_10.json")

        if num_epochs:
            self.config.num_epochs = num_epochs

        self.packet_encoder = packet_encoder.to(self.device)
        self.flow_embedding = flow_embedding.to(self.device)
        self.flow_encoder = flow_encoder.to(self.device)

        self.data_module = DataModule(device=self.device, config=self.config, base_path=TRAIN_BASE_PATH)
        self.train_loader_len = len(self.data_module.get_loader())

        self.optimizer = optim.Adam(
            list(self.packet_encoder.parameters()) +
            list(self.flow_embedding.parameters()) +
            list(self.flow_encoder.parameters()),
            lr=self.config.learning_rate,
        )

        self.accumulation_steps = 2
        self.accumulated_mlm_loss = 0.0
        self.accumulated_sfbo_loss = 0.0
        self.batch_counter = 0

        self.previous_packet_file = None
        self.previous_entry = None
        self.all_packet_encodings = []
        self.all_packet_counts = []
        self.FLOW_CHUNK_SIZE = 64

        self.trainer = Engine(self.train_step)

        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.post_step)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.end_epoch)

    def safe_prepare(self, tensor):
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() > 1 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def train_step(self, engine, batch):
        packet_sequences, field_position, header_position, entry = batch

        entry = {k: (v[0] if isinstance(v, list) else v) for k, v in entry.items()}

        packet_sequences = self.safe_prepare(packet_sequences)
        field_position = self.safe_prepare(field_position)
        header_position = self.safe_prepare(header_position)

        current_packet_file = entry.get("packet")

        mlm_loss, sfbo_loss, encoded_packets_mean = self.packet_encoder(
            packet_sequences,
            field_pos=field_position,
            header_pos=header_position,
        )

        self.accumulated_mlm_loss += mlm_loss
        self.accumulated_sfbo_loss += sfbo_loss
        self.batch_counter += 1

        self.all_packet_encodings.append(encoded_packets_mean.detach().cpu())
        self.all_packet_counts.append(encoded_packets_mean.size(0))

        return {
            "entry": entry,
            "packet_file": current_packet_file,
        }

    def post_step(self, engine):
        output = engine.state.output
        entry = output["entry"]
        current_packet_file = output["packet_file"]

        if self.batch_counter == self.accumulation_steps:
            total_loss = self.accumulated_mlm_loss + self.accumulated_sfbo_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.accumulated_mlm_loss = 0.0
            self.accumulated_sfbo_loss = 0.0
            self.batch_counter = 0

        if self.previous_entry is not None and current_packet_file != self.previous_packet_file:
            self.flush_flow(self.previous_entry)
            self.all_packet_encodings = []
            self.all_packet_counts = []

        if len(self.all_packet_encodings) >= self.FLOW_CHUNK_SIZE:
            self.process_chunk(entry)

        self.previous_packet_file = current_packet_file
        self.previous_entry = entry

    def process_chunk(self, entry):
        chunk = self.all_packet_encodings[: self.FLOW_CHUNK_SIZE]
        counts = self.all_packet_counts[: self.FLOW_CHUNK_SIZE]

        loss = self.process_encodings(chunk, counts, entry)

        if loss is not None and hasattr(loss, "backward"):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.all_packet_encodings = self.all_packet_encodings[self.FLOW_CHUNK_SIZE:]
        self.all_packet_counts = self.all_packet_counts[self.FLOW_CHUNK_SIZE:]

    def flush_flow(self, entry):
        if not self.all_packet_encodings:
            return

        loss = self.process_encodings(self.all_packet_encodings, self.all_packet_counts, entry)

        if loss is not None and hasattr(loss, "backward"):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def end_epoch(self, engine):
        if self.all_packet_encodings and self.previous_entry:
            self.flush_flow(self.previous_entry)
            self.all_packet_encodings = []
            self.all_packet_counts = []

    def process_encodings(self, encodings, packet_counts, entry):
        try:
            direction_file_path = entry.get("direction")
            with open(direction_file_path, "r", encoding="utf-8") as f:
                direction_data = [int(line.strip()) for line in f]
        except Exception:
            return None

        cumsum = [0]
        for count in packet_counts:
            cumsum.append(cumsum[-1] + count)

        total_loss = 0.0
        total_chunks = 0

        start = 0
        SUB_CHUNK_SIZE = 32

        while start < len(encodings):
            chunk = encodings[start:start + self.FLOW_CHUNK_SIZE]
            chunk_counts = packet_counts[start:start + self.FLOW_CHUNK_SIZE]

            dir_start = cumsum[start]
            dir_end = cumsum[start + len(chunk)]
            dir_chunk = direction_data[dir_start:dir_end]

            sub_start = 0
            sub_dir_idx = 0

            while sub_start < len(chunk):
                sub_chunk = chunk[sub_start:sub_start + SUB_CHUNK_SIZE]
                sub_counts = chunk_counts[sub_start:sub_start + SUB_CHUNK_SIZE]

                sub_dir_chunk = dir_chunk[sub_dir_idx: sub_dir_idx + sum(sub_counts)]
                sub_dir_idx += sum(sub_counts)

                try:
                    max_len = max(t.size(0) for t in sub_chunk)
                    embed_dim = sub_chunk[0].size(1)

                    padded = []
                    for t in sub_chunk:
                        if t.size(0) < max_len:
                            pad = torch.zeros(max_len - t.size(0), embed_dim)
                            t = torch.cat([t, pad], dim=0)
                        padded.append(t)

                    packet_chunk = torch.cat(padded, dim=0).to(self.device)

                    padded_dir = []
                    idx = 0
                    for i, t in enumerate(padded):
                        pkt_count = sub_counts[i]
                        d = sub_dir_chunk[idx: idx + pkt_count]
                        idx += pkt_count
                        d_tensor = torch.tensor(d)
                        if d_tensor.size(0) < max_len:
                            pad = torch.zeros(max_len - d_tensor.size(0), dtype=d_tensor.dtype)
                            d_tensor = torch.cat([d_tensor, pad], dim=0)
                        padded_dir.append(d_tensor)

                    direction_tensor = torch.cat(padded_dir, dim=0).to(self.device)

                    flow_embeddings, pad_indices = self.flow_embedding(packet_chunk, direction_tensor)
                    _, mpm_loss = self.flow_encoder(flow_embeddings, pad_indices)

                    total_loss += mpm_loss[0]
                    total_chunks += 1

                except Exception:
                    return None

                sub_start += SUB_CHUNK_SIZE

            start += self.FLOW_CHUNK_SIZE

        return total_loss / max(total_chunks, 1)

    def run(self):
        loader = self.data_module.get_batches()
        self.trainer.run(loader, max_epochs=self.config.num_epochs)


class ExperimentRunner:
    def __init__(self, manifest_path=None, num_epochs=None):
        self.config = Config()

        if manifest_path:
            self.config.manifest_path = manifest_path
        else:
            self.config.manifest_path = os.path.join(get_project_root(), "manifest", "manifest_10.json")

        if num_epochs:
            self.config.num_epochs = num_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        packet_encoder = PacketLevelEncoder(
            self.config.vocab_size,
            self.config.embed_dim,
            self.config.max_len,
            self.config.num_heads,
            self.config.num_layers,
            self.config.dropout,
        ).to(self.device)

        flow_embedding = FlowEmbedding(
            self.config.embed_dim,
            self.config.max_flow_length,
            self.config.dropout,
            {},
        ).to(self.device)

        flow_encoder = FlowLevelEncoder(
            self.config.embed_dim,
            self.config.num_layers,
            self.config.num_heads,
            self.config.dropout,
            {},
            self.config.max_flow_length,
            self.config.mask_prob,
        ).to(self.device)

        trainer = PacketLevelTrainer(
            packet_encoder,
            flow_embedding,
            flow_encoder,
            manifest_path=self.config.manifest_path,
            num_epochs=self.config.num_epochs,
        )

        trainer.run()


if __name__ == "__main__":
    manifest_path = os.path.join(get_project_root(), "manifest", "manifest_10.json")
    runner = ExperimentRunner(manifest_path=manifest_path, num_epochs=1)
    runner.run()