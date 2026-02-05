import json
import os
from sparkle.configs.config import Config
from sparkle.utils import get_project_root

config = Config()

INPUT_MANIFEST = config.manifest_path      # your existing manifest
OUTPUT_MANIFEST = os.path.join(get_project_root(), "manifest", "cleaned_local_manifest.json")

def same_tail(*paths):
    tails = [os.path.basename(p) for p in paths]
    return len(set(tails)) == 1

def sanitize_manifest(path_in, path_out):
    with open(path_in, "r") as f:
        data = json.load(f)

    cleaned = []
    for entry in data:
        if same_tail(entry["packet"],
                     entry["header"],
                     entry["field"],
                     entry["direction"]):
            cleaned.append(entry)

    with open(path_out, "w") as f:
        json.dump(cleaned, f, indent=4)

    print(f"Sanitized manifest saved to {path_out}")
    print(f"Kept {len(cleaned)} of {len(data)} entries.")

if __name__ == "__main__":
    sanitize_manifest(INPUT_MANIFEST, OUTPUT_MANIFEST)