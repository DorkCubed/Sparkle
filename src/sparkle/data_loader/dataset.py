import json
import os
from pathlib import Path

import torch
from s3fs import S3FileSystem
from torch.utils.data import Dataset

from sparkle.configs.config import Config
from sparkle.data_loader.encoder.positional_encodings import (
    field_pos_safe,
    header_pos_safe,
)


def remap_path(s3_path):
    """Remap S3-style paths to local paths"""
    if s3_path and s3_path.startswith("netml-s3-bucket/"):
        return os.path.join(os.path.expanduser("~/sparkle-datavol/local"), s3_path)
    return s3_path


class PacketSequenceDataset(Dataset):
    def __init__(
        self,
        config: Config,
        manifest_path,
        tokenizer,
        chunk_size,
        max_files=None,
        device=None,
        base_path=None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.manifest_path = manifest_path
        self.max_files = max_files
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.base_path = base_path  # Custom base path for remapping
        self.files = self._load_manifest()
        self.fs = S3FileSystem()

        # packets per sample returned in the dataset
        self.chunk_size = chunk_size
        self.total_chunks = []

        for file in self.files:
            num_lines = self._count_lines_streaming(file["packet"])
            num_chunks = (num_lines + self.chunk_size - 1) // self.chunk_size
            self.total_chunks.append(num_chunks)

        self.total_len = sum(self.total_chunks)

    def _load_manifest(self):
        with open(self.manifest_path, "r") as f:
            data = json.load(f)

        # Limit the number of files if max_files is specified
        if self.max_files is not None:
            data = data[: self.max_files]

        # Use custom base path if provided, otherwise use default
        if self.base_path:
            files = [
                {
                    k: os.path.join(os.path.expanduser(self.base_path), m[k])
                    if m[k].startswith("netml-s3-bucket/")
                    else m[k]
                    for k in ("packet", "header", "field", "direction")
                }
                for m in data
            ]
        else:
            files = [
                {
                    k: remap_path(m[k])
                    for k in ("packet", "header", "field", "direction")
                }
                for m in data
            ]

        return files

    def _read_s3_file(self, s3_path):
        with self.fs.open(s3_path, "r") as f:
            return f.read()

    def _read_file(self, path):
        path = Path(path)
        return path.read_text(encoding="utf-8")

    def _count_lines_streaming(self, path):
        """Count lines in a file without loading entire file into memory."""
        count = 0
        with open(path, "rb") as f:
            # Read in chunks to avoid loading entire file
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                count += chunk.count(b"\n")
        return count

    def _read_lines_chunk(self, path, start_line, num_lines):
        """Read only specific lines from a file without loading entire file."""
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            # Skip to start line
            for _ in range(start_line):
                next(f, None)
            # Read only needed lines
            for _ in range(num_lines):
                try:
                    line = next(f)
                    lines.append(line.strip())
                except StopIteration:
                    break
        return lines

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        cumulative_chunks = 0
        for file_idx, num_chunks in enumerate(self.total_chunks):
            if cumulative_chunks + num_chunks > idx:
                line_idx = idx - cumulative_chunks
                break
            cumulative_chunks += num_chunks
        else:
            raise IndexError("Index out of range")

        entry = self.files[file_idx]
        packet_path, header_path, field_path, direction_path = (
            entry["packet"],
            entry["header"],
            entry["field"],
            entry["direction"],
        )

        # Stream only the needed chunk instead of loading entire file
        chunk_start = line_idx * self.chunk_size
        # Read chunk_size lines - _read_lines_chunk handles EOF gracefully
        hex_dumps = self._read_lines_chunk(packet_path, chunk_start, self.chunk_size)
        padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(
            hex_dumps
        )

        chunk = token_ids
        chunk_end = chunk_start + len(hex_dumps)

        field_position = field_pos_safe(field_path, 0, len(hex_dumps), device="cpu")
        header_position = header_pos_safe(header_path, 0, len(hex_dumps), device="cpu")

        max_model_len = self.config.max_len
        actual_len = chunk.size(1)

        # Truncate to max_model_len
        if actual_len > max_model_len:
            chunk = chunk[:, :max_model_len]
            actual_len = max_model_len

        # Truncate positional encodings to match chunk length
        if field_position.size(1) > actual_len:
            field_position = field_position[:, :actual_len]
        if header_position.size(1) > actual_len:
            header_position = header_position[:, :actual_len]

        return chunk, field_position, header_position, entry
