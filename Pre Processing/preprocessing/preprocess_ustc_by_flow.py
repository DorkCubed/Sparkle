"""
preprocess_ustc_by_flow.py

Drop-in replacement for preprocess_ustc.py that fixes the file = "flow" bug.

WHAT CHANGED vs. the original preprocess_ustc.py:
  - Original: rdpcap() reads a whole PCAP, every packet in the file (mixed
    across all conversations) is written to one packets/<file>.txt, and the
    manifest has exactly one entry per PCAP.
  - This version: each PCAP is first split into its real network flows
    (grouped by 5-tuple: src IP, dst IP, src port, dst port, protocol) using
    flow_split.py. Each flow gets its own packets/fields/header/direction
    files and its own manifest entry, inheriting the PCAP's label.

Everything downstream (process_fields, process_pcap, direction encode_file,
the model, the linear probe) is untouched — they already operate on "a list
of packets" / "a packets file", so they don't need to know whether that list
came from one whole PCAP or one flow within it.

Usage: same as preprocess_ustc.py — edit INPUT_BASE / OUTPUT_BASE below and run.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from test import process_fields_from_packets
from pcap_to_packet import process_pcap_from_packets
from flow_split import split_pcap_into_flows

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from sparkle.model.utils.direction_encoder import encode_file

INPUT_BASE = "/home/shashank/sparkle/datavol-finetune/USTC-TFC2016-master/USTC-TFC2016-master"
OUTPUT_BASE = "/home/shashank/sparkle/data/ustc_preprocessed_by_flow"

LABEL_MAP = {
    "Benign": 0, "Cridex": 1, "Geodo": 2, "Htbot": 3, "Miuref": 4,
    "Neris": 5, "Nsis-ay": 6, "Shifu": 7, "Tinba": 8, "Virut": 9, "Zeus": 10,
}

# Cap on flows extracted per PCAP. USTC-TFC2016 PCAPs can contain thousands
# of short flows (e.g. background noise, scans); this keeps the longest /
# most substantial flows per file so classes stay roughly balanced and
# preprocessing stays tractable. Set to None for no cap.
MAX_FLOWS_PER_PCAP = 200

# Drop flows shorter than this many packets — a 1-2 packet flow (stray SYN,
# ARP-adjacent noise) carries little signal and previously crashed SFBO
# masking downstream.
MIN_PACKETS_PER_FLOW = 4


def process_one_pcap(pcap_path, outdir, label_stem):
    """Split one PCAP into flows and write packets/fields/header/direction
    files for each flow. Returns list of flow_ids successfully written."""
    flows = split_pcap_into_flows(
        pcap_path, min_packets=MIN_PACKETS_PER_FLOW, max_flows=MAX_FLOWS_PER_PCAP
    )

    pcap_stem = os.path.basename(pcap_path).replace(".pcap", "")
    written = []

    for flow_id, pkts in flows:
        out_stem = f"{pcap_stem}__{flow_id}.txt"
        packets_path = os.path.join(outdir, "packets", out_stem)

        if os.path.isfile(packets_path):
            written.append(out_stem)  # already processed, just register it
            continue

        try:
            process_fields_from_packets(pkts, os.path.join(outdir, out_stem))
            process_pcap_from_packets(pkts, os.path.join(outdir, out_stem))
            written.append(out_stem)
        except MemoryError as e:
            print(f"    SKIPPED flow {flow_id} (OOM): {e}")
        except Exception as e:
            print(f"    SKIPPED flow {flow_id} (error): {e}")

    return written


def main():
    manifest = []

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
            pcap_size = os.path.getsize(pcap_path)

            if pcap_size > 500 * 1024 * 1024:
                print(f"  SKIPPED (too large: {pcap_size/1e9:.2f} GB): {fname}")
                continue

            if dirname == "Malware":
                label_stem = fname.replace(".pcap", "")
                label = LABEL_MAP.get(label_stem, len(LABEL_MAP) + 1)
            else:
                label_stem = "Benign"
                label = 0

            print(f"Processing {pcap_path} ...")
            flow_stems = process_one_pcap(pcap_path, outdir, label_stem)
            print(f"  -> {len(flow_stems)} flows")

            for stem in flow_stems:
                manifest.append({
                    "flow": f"{dirname}/{stem}",
                    "packet": os.path.join(outdir, "packets", stem),
                    "field": os.path.join(outdir, "fields", stem),
                    "header": os.path.join(outdir, "header", stem),
                    "direction": os.path.join(outdir, "direction", stem),
                    "label": label,
                    "label_name": label_stem,
                    "source_pcap": fname,  # keep provenance for grouped train/test splitting
                })

    print("\nEncoding direction files...")
    for entry in manifest:
        try:
            encode_file(entry["direction"])
        except ValueError as e:
            print(f"  Skipping {entry['direction']}: {e}")

    manifest_path = os.path.join(OUTPUT_BASE, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {len(manifest)} flow-level entries saved to {manifest_path}")

    # Quick per-class count, since this is the number that actually matters now
    from collections import Counter
    counts = Counter(e["label_name"] for e in manifest)
    print("\nFlows per class:")
    for name, n in sorted(counts.items()):
        print(f"  {name:>10}: {n}")


if __name__ == "__main__":
    main()
