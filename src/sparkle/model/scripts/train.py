import logging
import os
from datetime import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sparkle.configs.config import Config
from sparkle.data_loader.data_loader import DataModule
from sparkle.utils import get_project_root
from sparkle.model.embedding import FlowEmbedding
from sparkle.model.flow_encoder import FlowLevelEncoder
from sparkle.model.packet_encoder import PacketLevelEncoder

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Base path for S3 remapping - training mode uses sparkle-datavol
TRAIN_BASE_PATH = "~/sparkle-datavol/local"


class PacketLevelTrainer:
    def __init__(
        self,
        packet_encoder,
        flow_embedding,
        flow_encoder,
        manifest_path=None,
        num_epochs=None,
    ):
        self.config = Config()
        self.vocab = self._init_vocab()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Override manifest path if provided, otherwise use default
        if manifest_path:
            self.config.manifest_path = manifest_path
        else:
            self.config.manifest_path = os.path.join(
                get_project_root(), "manifest", "manifest_10.json"
            )

        # Override num_epochs if provided
        if num_epochs:
            self.config.num_epochs = num_epochs

        self.skipped = 0

        self.packet_encoder = packet_encoder.to(self.device)
        self.flow_embedding = flow_embedding.to(self.device)
        self.flow_encoder = flow_encoder.to(self.device)

        self.data_module = DataModule(
            device=self.device, config=self.config, base_path=TRAIN_BASE_PATH
        )
        self.train_loader_len = len(self.data_module.get_loader())

        self.optimizer = optim.Adam(
            list(self.flow_embedding.parameters())
            + list(self.packet_encoder.parameters())
            + list(self.flow_encoder.parameters()),
            lr=self.config.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        self.step_successful = False

        self.accumulation_steps = 2
        self.accumulated_mlm_loss = 0.0
        self.accumulated_sfbo_loss = 0.0
        self.batch_counter = 0

        # bookkeeping logic
        self.previous_packet_file = None
        self.all_packet_encodings = []
        self.all_packet_counts = []  # Track packet counts per encoding
        self.previous_packet_id = None
        self.previous_entry = None
        self.total_packet_enc_loss = 0

        self.FLOW_CHUNK_SIZE = 64  # Process every 64 encodings to prevent accumulation

    def _init_vocab(self):
        vocab = {}
        with open(self.config.tokenizer_path, "r", encoding="utf-8") as f:
            for line in f:
                token, token_id = line.strip().split("\t")
                vocab[token] = int(token_id)

        return vocab

    def safe_prepare(self, tensor, name, device):
        if tensor is None:
            self.skipped += 1
            raise ValueError(f"{name} is None")

        if not hasattr(tensor, "to"):
            self.skipped += 1
            raise TypeError(f"{name} is not a tensor-like object")

        # Handle 0-d tensor first - add batch dimension
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)

        # Handle 1-d tensor - add batch dimension
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        # Handle 2-d tensor with batch size 1 - remove batch dimension
        if tensor.dim() > 1 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)

        # Now ensure it's at least 2-d
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        try:
            tensor = tensor.to(device)
        except Exception as e:
            self.skipped += 1
            raise RuntimeError(f"Failed to move {name} to device {device}: {e}") from e

        return tensor

    @staticmethod
    def validate_indices(tensors, names, limits):
        for t, name, limit in zip(tensors, names, limits):
            if t.min() < 0 or t.max() >= limit:
                raise ValueError(
                    f"{name} index out of bounds: min={t.min()}, max={t.max()}, limit={limit}"
                )

    @staticmethod
    def is_cuda_oom(e: Exception) -> bool:
        return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()

    # TODO (done): fix this to not cause OOMs
    # TODO test
    def process_encodings(self, encodings, packet_counts, entry):
        FLOW_CHUNK_SIZE = self.FLOW_CHUNK_SIZE

        if not encodings:
            logger.error("process_encodings called with empty encodings list.")
            return None

        direction_file_path = entry.get("direction")
        if direction_file_path is None:
            logger.error("Entry missing required key 'direction'.")
            return None

        try:
            with open(direction_file_path, "r", encoding="utf-8") as f:
                direction_data = [int(line.strip()) for line in f]
        except Exception:
            return None

        # Compute cumulative packet counts for proper direction indexing
        cumsum = [0]
        for count in packet_counts:
            cumsum.append(cumsum[-1] + count)

        total_loss = 0.0
        total_chunks = 0

        start = 0
        SUB_CHUNK_SIZE = 32  # Process in smaller chunks to avoid OOM

        while start < len(encodings):
            chunk = encodings[start : start + FLOW_CHUNK_SIZE]
            chunk_counts = packet_counts[start : start + FLOW_CHUNK_SIZE]

            # Get direction slice based on cumulative packet counts
            dir_start = cumsum[start]
            dir_end = cumsum[start + len(chunk)]
            dir_chunk = direction_data[dir_start:dir_end]

            # Process large chunks in smaller sub-chunks
            sub_start = 0
            sub_dir_idx = 0  # Track position in dir_chunk by packet count
            while sub_start < len(chunk):
                sub_chunk = chunk[sub_start : sub_start + SUB_CHUNK_SIZE]
                sub_counts = chunk_counts[sub_start : sub_start + SUB_CHUNK_SIZE]

                # Get direction values based on packet counts
                sub_dir_chunk = dir_chunk[sub_dir_idx : sub_dir_idx + sum(sub_counts)]
                sub_dir_idx += sum(sub_counts)

                packet_chunk = None
                direction_tensor = None
                flow_embeddings = None

                try:
                    if len(sub_chunk) == 0:
                        logger.warning("Empty sub_chunk encountered, skipping")
                        sub_start += SUB_CHUNK_SIZE
                        continue

                    # Pad all tensors to the same length before concatenation
                    max_len = max(t.size(0) for t in sub_chunk)
                    embed_dim = sub_chunk[0].size(1)
                    padded_sub_chunk = []
                    for t in sub_chunk:
                        if t.size(0) < max_len:
                            pad_size = max_len - t.size(0)
                            padding = torch.zeros(pad_size, embed_dim)
                            t = torch.cat([t, padding], dim=0)
                        padded_sub_chunk.append(t)

                    packet_chunk = torch.cat(padded_sub_chunk, dim=0).to(self.device)

                    # Pad direction values to match padded packet lengths
                    padded_dir = []
                    dir_idx = 0
                    for i, t in enumerate(padded_sub_chunk):
                        pkt_count = sub_counts[i]
                        d = sub_dir_chunk[dir_idx : dir_idx + pkt_count]
                        dir_idx += pkt_count
                        d_tensor = torch.tensor(d, dtype=torch.long, device="cpu")
                        if d_tensor.size(0) < max_len:
                            pad_size = max_len - d_tensor.size(0)
                            d_padding = torch.zeros(pad_size, dtype=d_tensor.dtype)
                            d_tensor = torch.cat([d_tensor, d_padding], dim=0)
                        padded_dir.append(d_tensor)
                    direction_tensor = torch.cat(padded_dir, dim=0).to(self.device)

                    if packet_chunk.dim() == 0 or direction_tensor.dim() == 0:
                        sub_start += SUB_CHUNK_SIZE
                        continue

                    flow_embeddings, pad_indices = self.flow_embedding(
                        packet_chunk, direction_tensor
                    )

                    _, mpm_loss = self.flow_encoder(flow_embeddings, pad_indices)

                    total_loss += mpm_loss[0]
                    total_chunks += 1

                except Exception as e:
                    logger.exception(f"Unexpected error inside process_encodings: {e}")

                    if self.is_cuda_oom(e):
                        print("Hit CUDA OOM in sub-chunk processing")
                        torch.cuda.empty_cache()
                        return None
                    else:
                        return None

                finally:
                    if packet_chunk is not None:
                        del packet_chunk
                    if direction_tensor is not None:
                        del direction_tensor
                    if flow_embeddings is not None:
                        del flow_embeddings
                    torch.cuda.empty_cache()

                sub_start += SUB_CHUNK_SIZE

            start += FLOW_CHUNK_SIZE

        return total_loss / max(total_chunks, 1)

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

        # Create fresh generator for each epoch
        train_loader = self.data_module.get_batches()

        progress_bar = tqdm(
            train_loader,
            total=self.train_loader_len,
            desc=f"Epoch {epoch + 1}",
            leave=False,
        )

        for i, (packet_sequences, field_position, header_position, entry) in enumerate(
            progress_bar
        ):
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

            # try:
            #     vocab_limit = self.config.vocab_size
            #     max_idx = packet_sequences.max().item()
            #     min_idx = packet_sequences.min().item()

            #     if max_idx >= vocab_limit or min_idx < 0:
            #         logger.error(
            #             f"SKIPPING BATCH {i}: Found invalid index {max_idx} (Max allowed: {vocab_limit - 1}) in file {entry.get('packet')}")
            #         self.skipped += 1
            #         progress_bar.set_postfix({"skipped": self.skipped})
            #         continue  # Skip safely, GPU is still healthy
            # except Exception as check_e:
            #     logger.error(f"Error during index validation: {check_e}")
            #     self.skipped += 1
            #     continue

            try:
                entry = {
                    k: (v[0] if isinstance(v, list) else v) for k, v in entry.items()
                }
                packet_sequences = self.safe_prepare(
                    packet_sequences, "packet_sequences", self.device
                )
                field_position = self.safe_prepare(
                    field_position, "field_position", self.device
                )
                header_position = self.safe_prepare(
                    header_position, "header_position", self.device
                )

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
                if (
                    self.previous_entry is not None
                    and current_packet_file != self.previous_packet_file
                ):
                    if self.all_packet_encodings:
                        mpm_loss = self.process_encodings(
                            self.all_packet_encodings,
                            self.all_packet_counts,
                            self.previous_entry,
                        )
                        if mpm_loss is not None:
                            self.optimizer.zero_grad()
                            mpm_loss.backward()
                            self.optimizer.step()

                    # reset for new file
                    self.all_packet_encodings = []
                    self.all_packet_counts = []
                    self.total_packet_enc_loss = 0

                self.previous_packet_file = current_packet_file
            except Exception as e:
                logger.exception(f"Error during file-boundary logic: {e}")
                self.all_packet_encodings = []
                self.all_packet_counts = []
                self.total_packet_enc_loss = 0
                self.skipped += 1

            try:
                # Packet-level forward pass
                mlm_loss, sfbo_loss, encoded_packets_mean = self.packet_encoder(
                    packet_sequences,
                    field_pos=field_position,
                    header_pos=header_position,
                )

                self.accumulated_mlm_loss += mlm_loss
                self.accumulated_sfbo_loss += sfbo_loss
                self.batch_counter += 1

                if self.batch_counter == self.accumulation_steps:
                    self.backward_and_optimize(
                        self.accumulated_mlm_loss, self.accumulated_sfbo_loss
                    )

                self.all_packet_encodings.append(encoded_packets_mean.detach().cpu())
                self.all_packet_counts.append(encoded_packets_mean.size(0))
                self.step_successful = True

                # Process encodings in chunks to prevent memory buildup
                if len(self.all_packet_encodings) >= self.FLOW_CHUNK_SIZE:
                    try:
                        chunk_loss = self.process_encodings(
                            self.all_packet_encodings[: self.FLOW_CHUNK_SIZE],
                            self.all_packet_counts[: self.FLOW_CHUNK_SIZE],
                            self.previous_entry if self.previous_entry else entry,
                        )
                        if chunk_loss is not None and hasattr(chunk_loss, "backward"):
                            self.optimizer.zero_grad()
                            chunk_loss.backward()
                            self.optimizer.step()
                    except Exception as proc_e:
                        logger.exception(f"Error processing encoding chunk: {proc_e}")
                    finally:
                        # Remove processed encodings to free memory
                        self.all_packet_encodings = self.all_packet_encodings[
                            self.FLOW_CHUNK_SIZE :
                        ]
                        self.all_packet_counts = self.all_packet_counts[
                            self.FLOW_CHUNK_SIZE :
                        ]
            except Exception as e:
                logger.error(
                    f"Exception at batch {i}: {type(e).__name__}: {str(e)[:200]}"
                )
                if self.is_cuda_oom(e):
                    logger.error(f"CUDA OOM at batch {i}, skipping batch")
                    logger.error(
                        f"  packet_sequences shape: {packet_sequences.shape}, device: {packet_sequences.device}"
                    )
                    logger.error(
                        f"  field_position shape: {field_position.shape}, device: {field_position.device}"
                    )
                    logger.error(
                        f"  header_position shape: {header_position.shape}, device: {header_position.device}"
                    )
                    logger.error(
                        f"  all_packet_encodings length: {len(self.all_packet_encodings)}"
                    )
                    current_file = entry.get("packet", "unknown")
                    logger.error(f"  Current file: {current_file}")

                    # critical cleanup
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                    # reset accumulation safely
                    self.accumulated_mlm_loss = 0
                    self.accumulated_sfbo_loss = 0
                    self.batch_counter = 0
                    self.all_packet_encodings = []
                    self.all_packet_counts = []

                    self.skipped += 1
                    continue
                continue

            self.previous_entry = entry

            progress_bar.set_postfix({"skipped": self.skipped})

        # Process any remaining encodings at end of epoch
        try:
            if self.all_packet_encodings and self.previous_entry:
                logger.info(
                    f"Processing {len(self.all_packet_encodings)} remaining encodings at end of epoch"
                )
                final_loss = self.process_encodings(
                    self.all_packet_encodings,
                    self.all_packet_counts,
                    self.previous_entry,
                )
                if final_loss is not None and hasattr(final_loss, "backward"):
                    self.optimizer.zero_grad()
                    final_loss.backward()
                    self.optimizer.step()
                self.all_packet_encodings = []
                self.all_packet_counts = []
        except Exception as e:
            logger.exception(f"Final mpm_loss computation failed: {e}")

    def save_checkpoint(self, epoch, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")

        state = {
            "epoch": epoch,
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

        self.packet_encoder.load_state_dict(checkpoint["packet_encoder"])
        self.flow_embedding.load_state_dict(checkpoint["flow_embedding"])
        self.flow_encoder.load_state_dict(checkpoint["flow_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.accumulated_mlm_loss = checkpoint.get("accumulated_mlm_loss", 0.0)
        self.accumulated_sfbo_loss = checkpoint.get("accumulated_sfbo_loss", 0.0)

        logger.info(
            f"Checkpoint loaded from {checkpoint_path} (epoch {checkpoint['epoch'] + 1})"
        )
        return checkpoint["epoch"]


class ExperimentRunner:
    def __init__(self, manifest_path=None, num_epochs=None):
        logger.info("Initializing ExperimentRunner...")
        self.config = Config()

        # Override manifest path if provided
        if manifest_path:
            self.config.manifest_path = manifest_path
            logger.info(f"Using custom manifest: {self.config.manifest_path}")
        else:
            # Override manifest path to use manifest_10.json
            self.config.manifest_path = os.path.join(
                get_project_root(), "manifest", "manifest_10.json"
            )
            logger.info(f"Using default manifest: {self.config.manifest_path}")

        # Override num_epochs if provided
        if num_epochs:
            self.config.num_epochs = num_epochs
            logger.info(f"Using custom epochs: {self.config.num_epochs}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_module = DataModule(
            device=self.device, config=self.config, base_path=TRAIN_BASE_PATH
        )
        self.tokenizer = data_module.get_tokenizer()
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")

    def load_vocab(self):
        vocab = {}
        with open(self.config.tokenizer_path, "r", encoding="utf-8") as f:
            for line in f:
                token, token_id = line.strip().split("\t")
                vocab[token] = int(token_id)
        return vocab

    def run(self):
        print("-" * 30)
        print(
            f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', 'Not Set')}"
        )
        print(
            f"TORCH_USE_CUDA_DSA:   {os.environ.get('TORCH_USE_CUDA_DSA', 'Not Set')}"
        )
        print("-" * 30)

        logger.info("Starting model training...")
        vocab = self.load_vocab()
        logger.info(f"Vocabulary size: {len(vocab)}")

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
            vocab,
        ).to(self.device)
        flow_encoder = FlowLevelEncoder(
            self.config.embed_dim,
            self.config.num_layers,
            self.config.num_heads,
            self.config.dropout,
            vocab,
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

        print("Loaded trainer.")

        for epoch in range(self.config.num_epochs):
            trainer.train_epoch(epoch)
            trainer.save_checkpoint(epoch)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Sparkle Model Runner")
    print("=" * 50)
    print("\nYou are about to run: train.py")

    while True:
        mode = (
            input(
                "\nDo you want to run in training mode or evaluation mode? (train/eval): "
            )
            .strip()
            .lower()
        )
        if mode in ["train", "eval"]:
            break
        print("Invalid input. Please enter 'train' or 'eval'.")

    if mode == "eval":
        print("\nHint: Run 'python -m src.sparkle.model.scripts.eval' instead\n")
        exit(0)

    print("\nRunning in training mode...")
    print("=" * 50)

    # Ask for manifest selection
    print("\nTraining Configuration:")
    print("-" * 30)

    # List available manifests
    available_manifests = {
        "1": "manifest_10.json",
        "2": "manifest_50.json",
        "3": "manifest_100.json",
        "4": "manifest.json",
        "5": "eval_manifest.json",
    }

    print("\nAvailable manifests:")
    for key, manifest in available_manifests.items():
        print(f"  {key}. {manifest}")

    while True:
        manifest_input = input("\nSelect manifest (1-5, default: 1): ").strip()
        if manifest_input == "":
            manifest_file = available_manifests["1"]
            break
        if manifest_input in available_manifests:
            manifest_file = available_manifests[manifest_input]
            break
        print("Invalid selection. Please enter 1, 2, 3, 4, or 5.")

    print(f"Selected: {manifest_file}")

    # Ask for number of epochs
    while True:
        epochs_input = input("Enter number of epochs (default: 1): ").strip()
        if epochs_input == "":
            num_epochs = 1
            break
        try:
            num_epochs = int(epochs_input)
            if num_epochs > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("-" * 30)
    print(f"Configuration: manifest={manifest_file}, epochs={num_epochs}")

    # Create runner with custom manifest and epochs
    manifest_path = os.path.join(get_project_root(), "manifest", manifest_file)
    runner = ExperimentRunner(manifest_path=manifest_path, num_epochs=num_epochs)
    runner.run()
