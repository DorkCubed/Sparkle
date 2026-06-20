import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def load_components(config, vocab):
    packet_encoder = PacketLevelEncoder(
        config.vocab_size, config.embed_dim, config.max_len,
        config.num_heads, config.num_layers, config.dropout
    )
    flow_embedding = FlowEmbedding(
        config.embed_dim, config.max_flow_length, config.dropout, vocab
    )
    flow_encoder = FlowLevelEncoder(
        config.embed_dim, config.num_layers, config.num_heads,
        config.dropout, vocab, config.max_flow_length, config.mask_prob
    )
    return packet_encoder.to(DEVICE), flow_embedding.to(DEVICE), flow_encoder.to(DEVICE)


def load_checkpoint(packet_encoder, flow_embedding, flow_encoder, path):
    state = torch.load(path, map_location=DEVICE)
    packet_encoder.load_state_dict(state["packet_encoder"])
    flow_embedding.load_state_dict(state["flow_embedding"])
    flow_encoder.load_state_dict(state["flow_encoder"])
    print(f"Loaded checkpoint from epoch {state.get('epoch', '?')}")


def extract_flow_embeddings(entry, tokenizer, packet_encoder, flow_embedding, flow_encoder,
                            max_packets=5000, max_model_len=578):
    hex_dumps = open(entry["packet"]).read().splitlines()
    if not hex_dumps:
        return []

    hex_dumps = hex_dumps[:max_packets]

    field_lines = open(entry["field"]).read().splitlines()[:max_packets]
    header_lines = open(entry["header"]).read().splitlines()[:max_packets]
    direction_lines = open(entry["direction"]).read().splitlines()[:max_packets]

    # Filter out packets with empty field lines (cause SFBO masking to crash)
    valid = [i for i, fl in enumerate(field_lines) if fl.strip()]
    if not valid:
        return []

    hex_dumps = [hex_dumps[i] for i in valid]
    field_lines = [field_lines[i] for i in valid]
    header_lines = [header_lines[i] for i in valid]
    direction_lines = [direction_lines[i] for i in valid]

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
    if max_seq_len > max_model_len:
        all_ids = all_ids[:, :max_model_len]
        field_pos = field_pos[:, :max_model_len]
        header_pos = header_pos[:, :max_model_len]

    direction = torch.tensor([int(l.strip()) for l in direction_lines], dtype=torch.long, device=DEVICE)

    packet_encoder.eval()
    flow_embedding.eval()
    flow_encoder.eval()

    for attempt in range(3):
        try:
            random.seed(attempt * 42)
            torch.manual_seed(attempt * 42)
            with torch.no_grad():
                _, _, encoded_packets = packet_encoder(all_ids, field_pos=field_pos, header_pos=header_pos)
                flow_emb, pad_indices = flow_embedding(encoded_packets, direction)
                flow_enc, _ = flow_encoder(flow_emb, pad_indices)
            embeddings = flow_enc.cpu().numpy()
            return [emb for emb in embeddings]
        except ValueError as e:
            if "num_spans exceeds" in str(e) and attempt < 2:
                continue
            return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/ustc_preprocessed/manifest.json")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_1.pth")
    parser.add_argument("--output", default="results/classification")
    parser.add_argument("--max_packets", type=int, default=5000)
    args = parser.parse_args()

    config = Config()
    tokenizer = Tokenizer(vocab_file=config.tokenizer_path)
    vocab = tokenizer.vocab

    print(f"Loading model on {DEVICE}...")
    packet_encoder, flow_embedding, flow_encoder = load_components(config, vocab)
    load_checkpoint(packet_encoder, flow_embedding, flow_encoder, args.checkpoint)

    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest)} entries")

    embeddings = []
    labels = []
    names = []

    for entry in manifest:
        embs = extract_flow_embeddings(entry, tokenizer, packet_encoder, flow_embedding, flow_encoder,
                                       max_packets=args.max_packets)
        for emb in embs:
            embeddings.append(emb)
            labels.append(entry["label"])
            names.append(entry["label_name"])
        print(f"  {entry['flow']:>30}  ->  {len(embs)} chunks")

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    print(f"\nExtracted {len(embeddings)} embeddings, dim={embeddings.shape[1] if embeddings.ndim > 1 else '?'}")

    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, labels, names, test_size=0.3, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=5000, multi_class="multinomial")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nConfusion Matrix:\n{cm}\n")


if __name__ == "__main__":
    main()
