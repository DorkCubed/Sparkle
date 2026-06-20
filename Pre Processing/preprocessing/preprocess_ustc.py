import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from test import process_fields
from pcap_to_packet import process_pcap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from sparkle.model.utils.direction_encoder import encode_file

INPUT_BASE = "/home/shashank/sparkle/datavol-finetune/USTC-TFC2016-master/USTC-TFC2016-master"
OUTPUT_BASE = "/home/shashank/sparkle/data/ustc_preprocessed"
LABEL_MAP = {
    "Benign": 0, "Cridex": 1, "Geodo": 2, "Htbot": 3, "Miuref": 4,
    "Neris": 5, "Nsis-ay": 6, "Shifu": 7, "Tinba": 8, "Virut": 9, "Zeus": 10,
}
LABEL_NAMES = {0: "Benign", 1: "Cridex", 2: "Geodo", 3: "Htbot", 4: "Miuref",
               5: "Neris", 6: "Nsis-ay", 7: "Shifu", 8: "Tinba", 9: "Virut", 10: "Zeus"}

for dirname in ("Benign", "Malware"):
    dirpath = os.path.join(INPUT_BASE, dirname)
    outdir = os.path.join(OUTPUT_BASE, dirname)
    os.makedirs(f"{outdir}/packets", exist_ok=True)
    os.makedirs(f"{outdir}/fields", exist_ok=True)
    os.makedirs(f"{outdir}/header", exist_ok=True)
    os.makedirs(f"{outdir}/direction", exist_ok=True)

    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith(".pcap"):
            continue
        pcap_path = os.path.join(dirpath, fname)
        out_stem = fname.replace(".pcap", ".txt")
        out_path = os.path.join(outdir, out_stem)

        print(f"Processing {pcap_path} -> {outdir}...")
        process_fields(pcap_path, out_path)
        process_pcap(pcap_path, out_path)

print("\nEncoding direction files...")
for dirname in ("Benign", "Malware"):
    direction_dir = os.path.join(OUTPUT_BASE, dirname, "direction")
    if not os.path.isdir(direction_dir):
        continue
    for fname in sorted(os.listdir(direction_dir)):
        fpath = os.path.join(direction_dir, fname)
        try:
            encode_file(fpath)
        except ValueError as e:
            print(f"  Skipping {fpath}: {e}")

print("\nBuilding manifest...")
manifest = []
for dirname in ("Benign", "Malware"):
    dirpath = os.path.join(OUTPUT_BASE, dirname)
    for fname in sorted(os.listdir(f"{dirpath}/packets")):
        if not fname.endswith(".txt"):
            continue
        base = dirname
        if dirname == "Malware":
            label_stem = fname.replace(".txt", "")
            label = LABEL_MAP.get(label_stem, len(LABEL_MAP) + 1)
        else:
            label_stem = "Benign"
            label = 0

        manifest.append({
            "flow": f"{dirname}/{fname}",
            "packet": os.path.join(dirpath, "packets", fname),
            "field": os.path.join(dirpath, "fields", fname),
            "header": os.path.join(dirpath, "header", fname),
            "direction": os.path.join(dirpath, "direction", fname),
            "label": label,
            "label_name": label_stem,
        })

manifest_path = os.path.join(OUTPUT_BASE, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Done. {len(manifest)} entries saved to {manifest_path}")
