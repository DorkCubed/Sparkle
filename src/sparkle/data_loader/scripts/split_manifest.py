from sparkle.configs.config import Config

import json
from pathlib import Path
import os
import random

def split_manifest(
    manifest_path,
    train_path,
    val_path,
    test_path,
    train_ratio=0.85,
    val_ratio=0.075,
    seed=42
):
    random.seed(seed)

    # Load the manifest (JSON array)
    with open(Path(manifest_path), "r", encoding="utf-8") as f:
        entries = json.load(f)

    random.shuffle(entries)

    n_total = len(entries)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_set = entries[:n_train]
    val_set = entries[n_train:n_train+n_val]
    test_set = entries[n_train+n_val:]

    # Save splits back as JSON arrays
    for path, subset in [(train_path, train_set),
                         (val_path, val_set),
                         (test_path, test_set)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=4, ensure_ascii=False)

    print(f"Total: {n_total}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")


if __name__ == "__main__":
    config = Config()
    manifest_path = Path(config.manifest_path)
    print(manifest_path)
    parent_path = manifest_path.parent
    train_path = os.path.join(parent_path, "train.json")
    val_path = os.path.join(parent_path, "val.json")
    test_path = os.path.join(parent_path, "test.json")
    split_manifest(manifest_path, train_path, val_path, test_path)