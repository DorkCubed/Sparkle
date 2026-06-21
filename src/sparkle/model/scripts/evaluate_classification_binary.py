import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
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
                            max_packets=3000, max_model_len=578):
    hex_dumps = open(entry["packet"]).read().splitlines()
    if not hex_dumps:
        return []

    hex_dumps = hex_dumps[:max_packets]

    field_lines = open(entry["field"]).read().splitlines()[:max_packets]
    header_lines = open(entry["header"]).read().splitlines()[:max_packets]
    direction_lines = open(entry["direction"]).read().splitlines()[:max_packets]

    # Filter out packets with empty field lines (crashes SFBO masking) and
    # packets whose direction wasn't cleanly encoded to "1"/"2". The latter
    # happens when a flow's direction file still has raw MAC strings in it
    # (e.g. it was never run through process_direction.py's encode_file
    # step, or predates that script existing) -- int(line) on those raises
    # ValueError with no guard, previously.
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
    parser.add_argument("--output", default="results/classification_binary")
    parser.add_argument("--max_packets", type=int, default=3000)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument(
        "--no_balance", action="store_true",
        help="Skip capping the benign class to match malware flow count. "
             "Off by default because raw USTC-TFC2016 benign:malware flow "
             "counts are heavily skewed (~1601:~2000 across 10 families, "
             "but per-PCAP this can be very lopsided), which lets a "
             "classifier hit high accuracy by mostly predicting the "
             "majority class rather than learning real signal."
    )
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
    label_names = []
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
            label_names.append(entry["label_name"])
            groups.append(group_id)
        print(f"  {entry['flow']:>30}  ->  {len(embs)} chunks")

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    label_names = np.array(label_names)
    groups = np.array(groups)
    print(f"\nExtracted {len(embeddings)} embeddings, dim={embeddings.shape[1] if embeddings.ndim > 1 else '?'}")
    print(f"From {len(set(groups))} distinct source PCAPs")

    # Binary collapse: label 0 (Benign, per LABEL_MAP in
    # preprocess_ustc_by_flow.py) stays 0; all 10 malware families collapse
    # to 1. This sidesteps the structural problem with the 11-class grouped
    # split -- each malware family has exactly one source PCAP, so a
    # leakage-free split can never hold out *and* train on a given family.
    # Binary collapse doesn't remove that limitation, it just changes the
    # question being asked: "malware vs benign on an unseen capture
    # session" instead of "which malware family", which the dataset can
    # actually support.
    labels = (labels != 0).astype(int)
    print(f"\nCollapsed to binary: {np.sum(labels == 0)} benign, {np.sum(labels == 1)} malware")

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
    family_test = label_names[test_idx]  # pre-collapse family name, for per-family breakdown

    train_pcaps = set(groups[train_idx])
    test_pcaps = set(groups[test_idx])
    overlap = train_pcaps & test_pcaps
    assert not overlap, f"LEAKAGE: PCAPs present in both train and test: {overlap}"

    print(f"\nTrain: {len(X_train)} samples from {len(train_pcaps)} PCAPs")
    print(f"Test:  {len(X_test)} samples from {len(test_pcaps)} PCAPs")
    print(f"Confirmed zero PCAP overlap between train and test.")

    train_class_counts = Counter(y_train.tolist())
    test_class_counts = Counter(y_test.tolist())
    print(f"Train class balance: benign={train_class_counts.get(0, 0)}  malware={train_class_counts.get(1, 0)}")
    print(f"Test  class balance: benign={test_class_counts.get(0, 0)}  malware={test_class_counts.get(1, 0)}")
    if test_class_counts.get(0, 0) == 0 or test_class_counts.get(1, 0) == 0:
        print("WARNING: test set is missing one of the two classes entirely -- "
              "accuracy/confusion matrix below will be degenerate. Try a "
              "different --test_size or rerun (GroupShuffleSplit uses a fixed "
              "random_state=42 here, so this needs a manual seed change).")

    # Cap the TRAIN benign count to roughly match the TRAIN malware count.
    # We balance only the train split, not test: the test set should still
    # reflect the real population so the reported metrics mean something,
    # but an unbalanced classifier trained on ~1600 benign vs ~2000 malware
    # flows (and worse, lopsided per-PCAP) can hit high accuracy just by
    # leaning on the majority class. Capping happens after the grouped
    # split so it never influences which PCAPs land in train vs test.
    if not args.no_balance:
        pre_balance_n = len(X_train)
        rng = np.random.RandomState(42)
        benign_idx = np.where(y_train == 0)[0]
        malware_idx = np.where(y_train == 1)[0]
        target = min(len(benign_idx), len(malware_idx))
        if len(benign_idx) > target:
            benign_idx = rng.choice(benign_idx, size=target, replace=False)
        if len(malware_idx) > target:
            malware_idx = rng.choice(malware_idx, size=target, replace=False)
        keep = np.sort(np.concatenate([benign_idx, malware_idx]))
        X_train, y_train = X_train[keep], y_train[keep]
        print(f"Balanced train set: {np.sum(y_train == 0)} benign, {np.sum(y_train == 1)} malware "
              f"({len(X_train)} total, down from {pre_balance_n})")

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}  (use this over raw accuracy -- "
          f"test set is not class-balanced, so a model that always predicts "
          f"the majority class can score well on accuracy alone)")
    print(classification_report(y_test, y_pred, target_names=["benign", "malware"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=true, cols=pred, order=[benign, malware]):")
    print(cm)

    # Per-family breakdown: binary accuracy hides whether the model is
    # actually catching every malware family or just a couple of easy ones.
    # This isn't a substitute for real 11-class evaluation (LOPO is the
    # right tool for that), but it's a useful sanity check on what's
    # driving the binary number.
    print("\nPer-family detection rate within test set (malware families only):")
    family_lines = []
    for fam in sorted(set(family_test[family_test != "Benign"])):
        fam_mask = family_test == fam
        fam_total = fam_mask.sum()
        fam_caught = np.sum((y_pred == 1) & fam_mask)
        line = f"  {fam:>10}: {fam_caught}/{fam_total} flows flagged as malware ({fam_caught/fam_total:.1%})"
        print(line)
        family_lines.append(line)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "report.txt"), "w") as f:
        f.write(f"BINARY classification (benign=0 vs malware=1)\n")
        f.write(f"Grouped split (by source PCAP) — test_size={args.test_size}\n")
        f.write(f"Train balancing: {'OFF (--no_balance)' if args.no_balance else 'ON (benign capped to malware count)'}\n")
        f.write(f"Train: {len(X_train)} samples / {len(train_pcaps)} PCAPs\n")
        f.write(f"Test:  {len(X_test)} samples / {len(test_pcaps)} PCAPs "
                f"(benign={test_class_counts.get(0, 0)}, malware={test_class_counts.get(1, 0)})\n")
        f.write(f"\nAccuracy: {acc:.4f}\n")
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=["benign", "malware"]))
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write(f"\nPer-family detection rate (malware families only):\n")
        f.write("\n".join(family_lines) + "\n")


if __name__ == "__main__":
    main()
