"""
End-to-end fine-tuning for flow classification.

Differs from evaluate_classification.py in one structural way: instead of
freezing the encoders and fitting a separate sklearn classifier on their
output, this trains the encoders AND a classification head together, so
gradients from the classification loss flow all the way back into
packet_encoder / flow_embedding / flow_encoder.

Read the comments marked DECISION POINT before running this -- those are
choices specific to your data/compute that I can't make for you.
"""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from sparkle.configs.config import Config
from sparkle.data_loader.tokenizer.tokenizer import Tokenizer
from sparkle.model.embedding import FlowEmbedding
from sparkle.model.flow_encoder import FlowLevelEncoder
from sparkle.model.packet_encoder import PacketLevelEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset: loads one flow per item, prepared the same way
# extract_flow_embeddings() does in evaluate_classification.py, but WITHOUT
# running it through the model here -- the model runs in the training loop,
# not in the dataset, because we need gradients.
# ---------------------------------------------------------------------------
class FlowClassificationDataset(Dataset):
    def __init__(self, manifest_entries, tokenizer, max_packets=5000, max_model_len=578):
        self.entries = manifest_entries
        self.tokenizer = tokenizer
        self.max_packets = max_packets
        self.max_model_len = max_model_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        hex_dumps = open(entry["packet"]).read().splitlines()[: self.max_packets]
        field_lines = open(entry["field"]).read().splitlines()[: self.max_packets]
        header_lines = open(entry["header"]).read().splitlines()[: self.max_packets]
        direction_lines = open(entry["direction"]).read().splitlines()[: self.max_packets]

        # same "drop empty field lines" guard as the eval script -- these
        # rows crash SFBO masking otherwise
        valid = [i for i, fl in enumerate(field_lines) if fl.strip()]
        hex_dumps = [hex_dumps[i] for i in valid]
        field_lines = [field_lines[i] for i in valid]
        header_lines = [header_lines[i] for i in valid]
        direction_lines = [direction_lines[i] for i in valid]

        all_tokens, all_ids, mask, _ = self.tokenizer.encode_packet(hex_dumps)

        def parse_positions(lines):
            parsed = [[int(x) for x in l.split()] for l in lines]
            max_len = max(len(s) for s in parsed)
            padded = [s + [0] * (max_len - len(s)) for s in parsed]
            return torch.tensor(padded, dtype=torch.long)

        field_pos = parse_positions(field_lines)
        header_pos = parse_positions(header_lines)

        max_seq_len = max(all_ids.size(1), field_pos.size(1), header_pos.size(1))
        if max_seq_len > self.max_model_len:
            all_ids = all_ids[:, : self.max_model_len]
            field_pos = field_pos[:, : self.max_model_len]
            header_pos = header_pos[:, : self.max_model_len]

        direction = torch.tensor([int(l.strip()) for l in direction_lines], dtype=torch.long)

        return {
            "packet_ids": all_ids,
            "field_pos": field_pos,
            "header_pos": header_pos,
            "direction": direction,
            "label": entry["label"],
            "flow": entry["flow"],
        }


def collate_single(batch):
    # DECISION POINT: batch_size=1 here, same constraint the rest of the
    # codebase has (variable-length flows don't stack into a regular tensor
    # without padding/bucketing work). One flow's gradient per step.
    # This is the *correct* baseline to get working first -- batching is an
    # optimization to add once this trains correctly, not before.
    assert len(batch) == 1
    return batch[0]


# ---------------------------------------------------------------------------
# Model: same three modules as evaluate_classification.py, plus a head
# ---------------------------------------------------------------------------
class FineTuneClassifier(nn.Module):
    def __init__(self, config, vocab, num_classes):
        super().__init__()
        self.packet_encoder = PacketLevelEncoder(
            config.vocab_size, config.embed_dim, config.max_len,
            config.num_heads, config.num_layers, config.dropout,
        )
        self.flow_embedding = FlowEmbedding(
            config.embed_dim, config.max_flow_length, config.dropout, vocab,
        )
        self.flow_encoder = FlowLevelEncoder(
            config.embed_dim, config.num_layers, config.num_heads,
            config.dropout, vocab, config.max_flow_length, config.mask_prob,
        )
        self.classifier_head = nn.Linear(config.embed_dim, num_classes)

    def load_pretrained(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        self.packet_encoder.load_state_dict(state["packet_encoder"])
        self.flow_embedding.load_state_dict(state["flow_embedding"])
        self.flow_encoder.load_state_dict(state["flow_encoder"])
        print(f"Loaded pretrained weights from epoch {state.get('epoch', '?')}")

    def forward(self, packet_ids, field_pos, header_pos, direction):
        # DECISION POINT: packet_encoder.forward applies MLM/SFBO masking
        # internally and returns those losses too. During fine-tuning you
        # have three options for mlm_loss/sfbo_loss:
        #   (a) ignore them, only backprop classification loss
        #   (b) add them in as regularizers: total = cls_loss + 0.1*(mlm+sfbo)
        #   (c) freeze packet_encoder, only fine-tune flow_encoder + head
        # Option (a) is simplest and what this script does. Try that first;
        # if you overfit fast (likely, given dataset size), (b) or (c) are
        # the next things to reach for, not a deeper/bigger head.
        mlm_loss, sfbo_loss, encoded_packets = self.packet_encoder(
            packet_ids, field_pos=field_pos, header_pos=header_pos
        )

        flow_emb, pad_indices = self.flow_embedding(encoded_packets, direction)

        # Same masking caveat applies here -- flow_encoder also injects MPM
        # masking and returns mpm_losses for it. Ignored here for the same
        # reason (option a above).
        flow_enc, mpm_losses = self.flow_encoder(flow_emb, pad_indices)

        # flow_enc is a list (one tensor per fraction, see flow_encoder.py);
        # mean-pool across packets within each fraction, then across
        # fractions, to get one fixed-size vector per flow -- matches what
        # evaluate_classification.py does with .mean(axis=0)
        fraction_means = [frac.mean(dim=0) for frac in flow_enc]
        flow_vector = torch.stack(fraction_means).mean(dim=0)

        logits = self.classifier_head(flow_vector)
        return logits


def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        packet_ids = batch["packet_ids"].to(DEVICE)
        field_pos = batch["field_pos"].to(DEVICE)
        header_pos = batch["header_pos"].to(DEVICE)
        direction = batch["direction"].to(DEVICE)
        label = torch.tensor([batch["label"]], device=DEVICE)

        with torch.set_grad_enabled(train):
            logits = model(packet_ids, field_pos, header_pos, direction)
            loss = criterion(logits.unsqueeze(0), label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # DECISION POINT: gradient clipping. The pretraining commits
                # you reviewed had NaN-loss issues already (the pad-token-
                # embedding fix). Fine-tuning end-to-end re-opens that risk
                # since gradients now flow through everything. Clip defensively:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        all_preds.append(logits.argmax(dim=-1).item())
        all_labels.append(label.item())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/ustc_preprocessed/manifest.json")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_1.pth")
    parser.add_argument("--output_checkpoint", default="checkpoints/finetuned.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_packets", type=int, default=3000)
    parser.add_argument("--freeze_packet_encoder", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    config = Config()
    tokenizer = Tokenizer(vocab_file=config.tokenizer_path)
    vocab = tokenizer.vocab

    with open(args.manifest) as f:
        manifest = json.load(f)

    labels = [e["label"] for e in manifest]
    num_classes = len(set(labels))
    print(f"Manifest: {len(manifest)} entries, {num_classes} classes")

    # Split at the manifest/file level, BEFORE any chunking happens --
    # this is the leakage point flagged in the linear-probe review.
    # Cannot stratify -- most malware classes have only 1 entry each.
    train_entries, val_entries = train_test_split(
        manifest, test_size=0.3, random_state=42
    )
    print(f"Train: {len(train_entries)}  Val: {len(val_entries)}")

    train_ds = FlowClassificationDataset(train_entries, tokenizer, args.max_packets)
    val_ds = FlowClassificationDataset(val_entries, tokenizer, args.max_packets)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    model = FineTuneClassifier(config, vocab, num_classes).to(DEVICE)
    model.load_pretrained(args.checkpoint)

    if args.freeze_packet_encoder:
        # DECISION POINT: with ~150-250 flows per class typical of
        # USTC-TFC2016, fine-tuning the full packet_encoder (the biggest
        # piece) risks overfitting fast. Freezing it and only fine-tuning
        # flow_encoder + classifier_head is a reasonable middle ground
        # between the linear probe (everything frozen) and full fine-tuning
        # (nothing frozen). Try --freeze_packet_encoder first if you have
        # limited data; drop it once you confirm things are stable.
        for p in model.packet_encoder.parameters():
            p.requires_grad = False
        print("packet_encoder frozen -- only flow_encoder + head will train")

    # DECISION POINT: lr=1e-5 default here, not the 1e-3 used for
    # pretraining in Config. This is the single most important fine-tuning
    # hyperparameter to get right -- pretrained weights already encode
    # useful structure, and a pretraining-scale LR will wreck that in a
    # few steps before the classification signal has a chance to help.
    # If loss explodes or accuracy stays at chance level, lower this first.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, optimizer, criterion, train=False)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "packet_encoder": model.packet_encoder.state_dict(),
                "flow_embedding": model.flow_embedding.state_dict(),
                "flow_encoder": model.flow_encoder.state_dict(),
                "classifier_head": model.classifier_head.state_dict(),
                "val_acc": val_acc,
            }, args.output_checkpoint)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

    print("\nFinal validation report:")
    print(classification_report(val_labels, val_preds))


if __name__ == "__main__":
    main()
