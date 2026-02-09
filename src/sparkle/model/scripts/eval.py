import json
import logging
import os
from datetime import datetime

import torch
from tqdm import tqdm

from sparkle.configs.config import Config
from sparkle.utils import get_project_root
from sparkle.data_loader.dataset import PacketSequenceDataset
from sparkle.data_loader.tokenizer.tokenizer import Tokenizer
from sparkle.model.embedding import PacketEmbedding, FlowEmbedding
from sparkle.model.flow_encoder import FlowLevelEncoder
from sparkle.model.packet_encoder import PacketLevelEncoder
from torch.utils.data import DataLoader


def remap_path(s3_path):
    """Remap S3-style paths to local paths"""
    if s3_path and s3_path.startswith("netml-s3-bucket/"):
        return os.path.join(os.path.expanduser("~/sparkle-finetuning"), s3_path)
    return s3_path


def setup_logging(log_dir=None):
    if log_dir is None:
        log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


class PacketLevelEvaluator:
    def __init__(
        self,
        packet_embedding,
        packet_encoder,
        flow_embedding,
        flow_encoder,
        eval_manifest_path,
        max_samples=None,
        max_files=None,
    ):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_samples = max_samples
        self.max_files = max_files

        self.packet_embedding = packet_embedding.to(self.device)
        self.packet_encoder = packet_encoder.to(self.device)
        self.flow_embedding = flow_embedding.to(self.device)
        self.flow_encoder = flow_encoder.to(self.device)

        self.packet_embedding.eval()
        self.packet_encoder.eval()
        self.flow_embedding.eval()
        self.flow_encoder.eval()

        self.vocab = self._init_vocab()
        self.eval_loader = self._init_eval_loader(eval_manifest_path)

        if self.max_files:
            logger = logging.getLogger(__name__)
            logger.info(f"Limited to first {self.max_files} files from manifest")

        self.skipped = 0
        self.FLOW_CHUNK_SIZE = 512

        self.total_mlm_loss = 0.0
        self.total_sfbo_loss = 0.0
        self.total_mpm_loss = 0.0
        self.total_loss = 0.0
        self.num_batches = 0
        self.num_samples = 0

        self.previous_packet_file = None
        self.all_packet_encodings = []
        self.previous_entry = None

    def _init_vocab(self):
        vocab = {}
        with open(self.config.tokenizer_path, "r", encoding="utf-8") as f:
            for line in f:
                token, token_id = line.strip().split("\t")
                vocab[token] = int(token_id)
        return vocab

    def _init_eval_loader(self, eval_manifest_path):
        tokenizer = Tokenizer(vocab_file=self.config.tokenizer_path)

        dataset = PacketSequenceDataset(
            config=self.config,
            manifest_path=eval_manifest_path,
            tokenizer=tokenizer,
            chunk_size=self.config.chunk_size,
            max_files=self.max_files,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def safe_prepare(self, tensor, name):
        if tensor is None:
            self.skipped += 1
            raise ValueError(f"{name} is None")

        if tensor.dim() > 1 and tensor.size(0) == 1:
            tensor = tensor[0]

        try:
            tensor = tensor.squeeze(0)
        except Exception as e:
            self.skipped += 1
            raise RuntimeError(f"Failed to squeeze {name}: {e}") from e

        try:
            tensor = tensor.to(self.device)
        except Exception as e:
            self.skipped += 1
            raise RuntimeError(
                f"Failed to move {name} to device {self.device}: {e}"
            ) from e

        return tensor

    @staticmethod
    def is_cuda_oom(e: Exception) -> bool:
        return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()

    def process_encodings(self, encodings, entry):
        if not encodings:
            return None, 0

        direction_file_path = entry.get("direction")
        if direction_file_path is None:
            return None, 0

        direction_file_path = remap_path(direction_file_path)

        try:
            with open(direction_file_path, "r", encoding="utf-8") as f:
                direction_data = [int(line.strip()) for line in f]
        except Exception:
            return None, 0

        total_mpm_loss = 0.0
        total_chunks = 0
        start = 0

        while start < len(encodings):
            chunk = encodings[start : start + self.FLOW_CHUNK_SIZE]
            dir_chunk = direction_data[start : start + self.FLOW_CHUNK_SIZE]

            packet_chunk = None
            direction_tensor = None
            flow_embeddings = None

            try:
                packet_chunk = torch.cat(chunk, dim=0).to(self.device)
                direction_tensor = torch.tensor(dir_chunk, device=self.device)

                flow_embeddings, pad_indices = self.flow_embedding(
                    packet_chunk, direction_tensor
                )
                _, mpm_losses = self.flow_encoder(flow_embeddings, pad_indices)

                if mpm_losses:
                    total_mpm_loss += sum(mpm_losses)
                    total_chunks += len(mpm_losses)

            except Exception as e:
                if self.is_cuda_oom(e):
                    torch.cuda.empty_cache()
                return None, 0
            finally:
                del packet_chunk, direction_tensor, flow_embeddings
                torch.cuda.empty_cache()

            start += self.FLOW_CHUNK_SIZE

        avg_mpm_loss = total_mpm_loss / max(total_chunks, 1)
        return avg_mpm_loss, total_chunks

    def evaluate(self):
        logger = logging.getLogger(__name__)
        logger.info(f"\n{'=' * 30}")
        logger.info("Starting evaluation")
        logger.info(f"{'=' * 30}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Max samples: {self.max_samples if self.max_samples else 'All'}")

        with torch.no_grad():
            progress_bar = tqdm(
                self.eval_loader,
                total=len(self.eval_loader),
                desc="Evaluating",
                leave=False,
            )

            for i, (
                packet_sequences,
                field_position,
                header_position,
                entry,
            ) in enumerate(progress_bar):
                if self.max_samples and self.num_samples >= self.max_samples:
                    logger.info(f"Reached max samples limit: {self.max_samples}")
                    break

                try:
                    entry = {
                        k: (v[0] if isinstance(v, list) else v)
                        for k, v in entry.items()
                    }
                    packet_sequences = self.safe_prepare(
                        packet_sequences, "packet_sequences"
                    )
                    field_position = self.safe_prepare(field_position, "field_position")
                    header_position = self.safe_prepare(
                        header_position, "header_position"
                    )

                    current_packet_file = entry.get("packet")
                    if current_packet_file is None:
                        logger.error(
                            "Missing required entry['packet'] in batch, skipping."
                        )
                        continue
                except Exception as e:
                    logger.exception(f"Unexpected error inside batch {i}: {e}")
                    self.skipped += 1
                    continue

                try:
                    if (
                        self.previous_entry is not None
                        and current_packet_file != self.previous_packet_file
                    ):
                        if self.all_packet_encodings:
                            mpm_loss, num_chunks = self.process_encodings(
                                self.all_packet_encodings, self.previous_entry
                            )
                            if mpm_loss is not None:
                                self.total_mpm_loss += mpm_loss * num_chunks
                                self.num_batches += num_chunks

                        self.all_packet_encodings = []

                    self.previous_packet_file = current_packet_file
                except Exception as e:
                    logger.exception(f"Error during file-boundary logic: {e}")
                    self.all_packet_encodings = []
                    self.skipped += 1

                try:
                    mlm_loss, sfbo_loss, encoded_packets_mean = self.packet_encoder(
                        packet_sequences,
                        field_pos=field_position,
                        header_pos=header_position,
                    )

                    self.total_mlm_loss += mlm_loss.item()
                    self.total_sfbo_loss += sfbo_loss.item()
                    self.total_loss += mlm_loss.item() + sfbo_loss.item()
                    self.num_samples += 1

                    self.all_packet_encodings.append(
                        encoded_packets_mean.detach().cpu()
                    )

                except Exception as e:
                    if self.is_cuda_oom(e):
                        logger.error(f"CUDA OOM at batch {i}, skipping batch")
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        self.all_packet_encodings = []
                    self.skipped += 1
                    continue

                self.previous_entry = entry
                progress_bar.set_postfix({"skipped": self.skipped})

            try:
                if (
                    self.all_packet_encodings
                    and self.previous_entry
                    and self.previous_entry.get("packet")
                ):
                    mpm_loss, num_chunks = self.process_encodings(
                        self.all_packet_encodings, self.previous_entry
                    )
                    if mpm_loss is not None:
                        self.total_mpm_loss += mpm_loss * num_chunks
                        self.num_batches += num_chunks
            except Exception as e:
                logger.exception(f"Final mpm_loss computation failed: {e}")

        metrics = self._compute_metrics()
        return metrics

    def _compute_metrics(self):
        total_processed = max(self.num_samples, 1)

        metrics = {
            "avg_mlm_loss": self.total_mlm_loss / total_processed,
            "avg_sfbo_loss": self.total_sfbo_loss / total_processed,
            "avg_packet_loss": (self.total_mlm_loss + self.total_sfbo_loss)
            / total_processed,
            "avg_mpm_loss": self.total_mpm_loss / max(self.num_batches, 1)
            if self.num_batches > 0
            else 0.0,
            "total_loss": self.total_loss + self.total_mpm_loss,
            "num_samples": self.num_samples,
            "num_batches": self.num_batches,
            "skipped": self.skipped,
            "perplexity": torch.exp(
                torch.tensor(
                    (self.total_loss + self.total_mpm_loss)
                    / max(total_processed + self.num_batches, 1)
                )
            ).item(),
        }

        return metrics


def load_model_from_checkpoint(checkpoint_path, config, vocab, device):
    packet_embedding = PacketEmbedding(
        config.vocab_size,
        max_len=config.max_len,
        embed_dim=config.embed_dim,
        dropout=config.dropout,
    ).to(device)

    packet_encoder = PacketLevelEncoder(
        config.vocab_size,
        config.embed_dim,
        config.max_len,
        config.num_heads,
        config.num_layers,
        config.dropout,
    ).to(device)

    flow_embedding = FlowEmbedding(
        config.embed_dim, config.max_flow_length, config.dropout, vocab
    ).to(device)

    flow_encoder = FlowLevelEncoder(
        config.embed_dim,
        config.num_layers,
        config.num_heads,
        config.dropout,
        vocab,
        config.max_flow_length,
        config.mask_prob,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    packet_embedding.load_state_dict(checkpoint["packet_embedding"])
    packet_encoder.load_state_dict(checkpoint["packet_encoder"])
    flow_embedding.load_state_dict(checkpoint["flow_embedding"])
    flow_encoder.load_state_dict(checkpoint["flow_encoder"])

    logger = logging.getLogger(__name__)
    logger.info(
        f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown') + 1})"
    )

    return packet_embedding, packet_encoder, flow_embedding, flow_encoder


def save_metrics_to_json(metrics, output_path):
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_serializable[key] = value.item()
        else:
            metrics_serializable[key] = value

    metrics_serializable["eval_timestamp"] = datetime.now().isoformat()

    with open(output_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    return output_path


def run_evaluation(
    checkpoint_path=None,
    eval_manifest_path=None,
    max_samples=None,
    output_json_path=None,
    log_dir=None,
    max_files=None,
):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            get_project_root(), "checkpoints", "checkpoint_1.pth"
        )
    if eval_manifest_path is None:
        eval_manifest_path = os.path.join(
            get_project_root(), "manifest", "eval_manifest.json"
        )
    if log_dir is None:
        log_dir = os.path.join(get_project_root(), "logs")

    logger = setup_logging(log_dir)

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading vocabulary...")
    vocab = {}
    with open(config.tokenizer_path, "r", encoding="utf-8") as f:
        for line in f:
            token, token_id = line.strip().split("\t")
            vocab[token] = int(token_id)

    logger.info(f"Vocabulary size: {len(vocab)}")

    logger.info("Loading model from checkpoint...")
    packet_embedding, packet_encoder, flow_embedding, flow_encoder = (
        load_model_from_checkpoint(checkpoint_path, config, vocab, device)
    )

    logger.info("Initializing evaluator...")
    evaluator = PacketLevelEvaluator(
        packet_embedding=packet_embedding,
        packet_encoder=packet_encoder,
        flow_embedding=flow_embedding,
        flow_encoder=flow_encoder,
        eval_manifest_path=eval_manifest_path,
        max_samples=max_samples,
        max_files=max_files,
    )

    logger.info("Starting evaluation...")
    metrics = evaluator.evaluate()

    logger.info(f"\n{'=' * 30}")
    logger.info("Evaluation Results")
    logger.info(f"{'=' * 30}")
    logger.info(f"Average MLM Loss: {metrics['avg_mlm_loss']:.4f}")
    logger.info(f"Average SFBO Loss: {metrics['avg_sfbo_loss']:.4f}")
    logger.info(f"Average Packet Loss: {metrics['avg_packet_loss']:.4f}")
    logger.info(f"Average MPM Loss: {metrics['avg_mpm_loss']:.4f}")
    logger.info(f"Total Loss: {metrics['total_loss']:.4f}")
    logger.info(f"Perplexity: {metrics['perplexity']:.4f}")
    logger.info(f"Samples Evaluated: {metrics['num_samples']}")
    logger.info(f"Flow Batches: {metrics['num_batches']}")
    logger.info(f"Skipped: {metrics['skipped']}")
    logger.info(f"{'=' * 30}")

    if output_json_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json_path = os.path.join(
            get_project_root(), "logs", f"eval_results_{timestamp}.json"
        )

    output_path = save_metrics_to_json(metrics, output_json_path)
    logger.info(f"Metrics saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    CHECKPOINT_PATH = os.path.join(
        get_project_root(), "checkpoints", "checkpoint_1.pth"
    )
    EVAL_MANIFEST_PATH = os.path.join(
        get_project_root(), "manifest", "eval_manifest.json"
    )
    MAX_SAMPLES = 10
    MAX_FILES = 2  # Limit to first 2 files from manifest
    OUTPUT_JSON_PATH = None
    LOG_DIR = os.path.join(get_project_root(), "logs")

    metrics = run_evaluation(
        checkpoint_path=CHECKPOINT_PATH,
        eval_manifest_path=EVAL_MANIFEST_PATH,
        max_samples=MAX_SAMPLES,
        output_json_path=OUTPUT_JSON_PATH,
        log_dir=LOG_DIR,
        max_files=MAX_FILES,
    )

    print("\nEvaluation complete!")
    print(f"Results: {metrics}")
