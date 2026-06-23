"""FastAPI inference server for SPARKLE flow embeddings.

Usage:
    uv run fastapi dev src/sparkle/model/serve.py --port 8000
    uv run uvicorn sparkle.model.serve:app --host 0.0.0.0 --port 8000
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sparkle.configs.config import Config
from sparkle.data_loader.tokenizer.tokenizer import Tokenizer
from sparkle.model.embedding import FlowEmbedding
from sparkle.model.flow_encoder import FlowLevelEncoder
from sparkle.model.packet_encoder import PacketLevelEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = torch.Generator(device=DEVICE)
RNG.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

app = FastAPI(title="SPARKLE Flow Embedding API", version="0.1.0")

# ---------------------------------------------------------------------------
# Global model components (lazy loaded on first request)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None


def get_project_root():
    return Path(__file__).resolve().parent.parent.parent.parent


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    config = Config()
    project_root = get_project_root()
    vocab_path = os.path.join(project_root, "src", "sparkle", "data_loader", "tokenizer", "vocab.txt")
    checkpoint_path = os.path.join(project_root, "checkpoints", "checkpoint_1.pth")

    if not os.path.exists(vocab_path):
        raise RuntimeError(f"Vocab not found: {vocab_path}")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    _tokenizer = Tokenizer(vocab_path)

    packet_encoder = PacketLevelEncoder(
        config.vocab_size, config.embed_dim, config.max_len,
        config.num_heads, config.num_layers, config.dropout
    )
    flow_embedding = FlowEmbedding(
        config.embed_dim, config.max_flow_length, config.dropout, _tokenizer.vocab
    )
    flow_encoder = FlowLevelEncoder(
        config.embed_dim, config.num_layers, config.num_heads,
        config.dropout, _tokenizer.vocab, config.max_flow_length, config.mask_prob
    )

    state = torch.load(checkpoint_path, map_location=DEVICE)
    packet_encoder.load_state_dict(state["packet_encoder"])
    flow_embedding.load_state_dict(state["flow_embedding"])
    flow_encoder.load_state_dict(state["flow_encoder"])

    packet_encoder.to(DEVICE)
    flow_embedding.to(DEVICE)
    flow_encoder.to(DEVICE)
    packet_encoder.eval()
    flow_embedding.eval()
    flow_encoder.eval()

    _model = {
        "packet_encoder": packet_encoder,
        "flow_embedding": flow_embedding,
        "flow_encoder": flow_encoder,
        "config": config,
    }
    print(f"Model loaded on {DEVICE}")
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    hex_dumps: list[str]
    field_lines: list[str]
    header_lines: list[str]
    direction_lines: list[str]
    max_packets: int = 3000


class EmbedResponse(BaseModel):
    embedding: list[float]
    n_packets: int
    device: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        load_model()
        return HealthResponse(status="ok", device=str(DEVICE), model_loaded=True)
    except RuntimeError as e:
        return HealthResponse(status=f"error: {e}", device=str(DEVICE), model_loaded=False)


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    model, tokenizer = load_model()
    pe = model["packet_encoder"]
    fe = model["flow_embedding"]
    fen = model["flow_encoder"]
    config = model["config"]

    n = min(len(req.hex_dumps), len(req.field_lines), len(req.header_lines), len(req.direction_lines))
    n = min(n, req.max_packets)
    if n == 0:
        raise HTTPException(status_code=400, detail="No valid packets in request")

    valid = [
        i for i in range(n)
        if req.field_lines[i].strip()
        and req.direction_lines[i].strip() in ("1", "2")
    ]
    if not valid:
        raise HTTPException(status_code=400, detail="No packets with valid field/direction data")

    hex_dumps = [req.hex_dumps[i] for i in valid]
    field_lines = [req.field_lines[i] for i in valid]
    header_lines = [req.header_lines[i] for i in valid]
    direction_lines = [req.direction_lines[i] for i in valid]

    all_tokens, all_ids, mask, max_len = tokenizer.encode_packet(hex_dumps)
    all_ids = all_ids.to(DEVICE)

    def parse_positions(lines, device):
        parsed = [[int(x) for x in l.split()] for l in lines]
        max_len = max(len(s) for s in parsed)
        padded = [s + [0] * (max_len - len(s)) for s in parsed]
        return torch.tensor(padded, dtype=torch.long, device=device)

    field_pos = parse_positions(field_lines, DEVICE)
    header_pos = parse_positions(header_lines, DEVICE)

    max_seq_len = max(all_ids.size(1), field_pos.size(1), header_pos.size(1))
    if max_seq_len > config.max_len:
        all_ids = all_ids[:, :config.max_len]
        field_pos = field_pos[:, :config.max_len]
        header_pos = header_pos[:, :config.max_len]

    direction = torch.tensor([int(l.strip()) for l in direction_lines], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        _, _, encoded_packets = pe(all_ids, field_pos=field_pos, header_pos=header_pos)
        flow_emb, pad_indices = fe(encoded_packets, direction)
        flow_enc, _ = fen(flow_emb, pad_indices)

    embeddings = flow_enc.cpu().numpy()
    flow_vec = embeddings.mean(axis=0).tolist()

    return EmbedResponse(
        embedding=flow_vec,
        n_packets=len(valid),
        device=str(DEVICE),
    )
