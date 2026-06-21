import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
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

    # Single combined filter: keep only rows where all four files have a valid
    # entry — non-empty field line, direction encoded to 1 or 2, and index
    # in bounds for all lists (files can have slightly different line counts).
    n = min(len(hex_dumps), len(field_lines), len(header_lines), len(direction_lines))
    valid = [
        i for i in range(n)
        if field_lines[i].strip()
        and direction_lines[i].strip() in ("1", "2")
    ]
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
            return [emb.mean(axis=0) for emb in embeddings]
        except ValueError as e:
            if "num_spans exceeds" in str(e) and attempt < 2:
                continue
            return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/ustc_preprocessed_by_flow/manifest.json")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_1.pth")
    parser.add_argument("--output", default="results/classification")
    parser.add_argument("--max_packets", type=int, default=5000)
    parser.add_argument("--test_size", type=float, default=0.3)
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
    groups = []  # source PCAP filename per embedding, for leakage-free splitting

    for entry in manifest:
        # source_pcap is added by preprocess_ustc_by_flow.py. Fall back to
        # the "flow" field (file-level manifests) so this script still runs
        # against old, non-flow-split manifests without crashing.
        group_id = entry.get("source_pcap", entry["flow"])

        embs = extract_flow_embeddings(entry, tokenizer, packet_encoder, flow_embedding, flow_encoder,
                                       max_packets=args.max_packets)
        for emb in embs:
            embeddings.append(emb)
            labels.append(entry["label"])
            names.append(entry["label_name"])
            groups.append(group_id)
        print(f"  {entry['flow']:>30}  ->  {len(embs)} chunks")

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    groups = np.array(groups)
    print(f"\nExtracted {len(embeddings)} embeddings, dim={embeddings.shape[1] if embeddings.ndim > 1 else '?'}")
    print(f"From {len(set(groups))} distinct source PCAPs")

    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    # GROUPED split: every embedding from the same source PCAP goes
    # entirely into train OR entirely into test, never split across both.
    # This prevents the model/classifier from "recognizing the capture
    # session" (shared background noise, same two IPs, same timing) instead
    # of genuinely generalizing to the malware family.
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, labels, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    names_train, names_test = groups[train_idx], groups[test_idx]

    train_pcaps = set(groups[train_idx])
    test_pcaps = set(groups[test_idx])
    overlap = train_pcaps & test_pcaps
    assert not overlap, f"LEAKAGE: PCAPs present in both train and test: {overlap}"

    print(f"\nTrain: {len(X_train)} samples from {len(train_pcaps)} PCAPs")
    print(f"Test:  {len(X_test)} samples from {len(test_pcaps)} PCAPs")
    print(f"Confirmed zero PCAP overlap between train and test.")

    train_class_counts = Counter(y_train.tolist())
    test_class_counts = Counter(y_test.tolist())
    all_classes = sorted(set(labels.tolist()))
    missing_from_train = [c for c in all_classes if train_class_counts.get(c, 0) == 0]
    missing_from_test = [c for c in all_classes if test_class_counts.get(c, 0) == 0]
    if missing_from_train:
        print(f"WARNING: classes with zero TRAIN samples after grouped split: {missing_from_train}")
    if missing_from_test:
        print(f"WARNING: classes with zero TEST samples after grouped split: {missing_from_test}")

    # If a class has only one source PCAP, a leakage-free grouped split can
    # never put any of that class in both train and test -- it is
    # structurally all-or-nothing for that class, on every run, regardless
    # of random_state. This is not a bug in the split; it is the dataset
    # telling you it doesn't support a held-out test set for that class yet.
    pcaps_per_class = {}
    for g, y in zip(groups.tolist(), labels.tolist()):
        pcaps_per_class.setdefault(y, set()).add(g)
    single_pcap_classes = [c for c, pcaps in pcaps_per_class.items() if len(pcaps) == 1]
    if single_pcap_classes:
        print(
            f"\nNOTE: classes backed by only ONE source PCAP: {single_pcap_classes}. "
            "A grouped (leakage-free) split cannot give these classes both train "
            "and test coverage in the same run -- more PCAPs per class are needed "
            "for a meaningful held-out evaluation of them."
        )

    clf = LogisticRegression(max_iter=5000)
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
        f.write(f"Grouped split (by source PCAP) — test_size={args.test_size}\n")
        f.write(f"Train: {len(X_train)} samples / {len(train_pcaps)} PCAPs\n")
        f.write(f"Test:  {len(X_test)} samples / {len(test_pcaps)} PCAPs\n")
        if missing_from_train:
            f.write(f"WARNING: classes missing from train: {missing_from_train}\n")
        if missing_from_test:
            f.write(f"WARNING: classes missing from test: {missing_from_test}\n")
        f.write(f"\nAccuracy: {acc:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nConfusion Matrix:\n{cm}\n")


if __name__ == "__main__":
    main()
