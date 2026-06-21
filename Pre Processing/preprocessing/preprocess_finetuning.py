import argparse
import json
import os
import time
import uuid
from collections import defaultdict

from scapy.all import rdpcap, linehexdump


def get_flow_key(pkt):
    if pkt.haslayer("IP"):
        ip = pkt["IP"]
        proto = ip.proto
        if pkt.haslayer("TCP"):
            return (ip.src, ip.dst, pkt["TCP"].sport, pkt["TCP"].dport, proto)
        elif pkt.haslayer("UDP"):
            return (ip.src, ip.dst, pkt["UDP"].sport, pkt["UDP"].dport, proto)
        else:
            return (ip.src, ip.dst, 0, 0, proto)
    return None


def generate_hex_dump(pkt):
    layer = pkt.firstlayer().payload
    pdata = linehexdump(layer, onlyhex=1, dump=True)
    for layer_name in layer.layers():
        for field in layer_name.fields_desc:
            if field.name == "load":
                loaddata = getattr(pkt, field.name)
                loaddata = linehexdump(loaddata, onlyhex=1, dump=True)
                pdata = pdata.rsplit(loaddata, 1)
                if len(pdata) > 1:
                    pdata = pdata[0] + pdata[1]
                else:
                    pdata = pdata[0]
    if pdata == "":
        pdata = "emp"
    return pdata


def generate_field_header_positions(pkt):
    startlayer = pkt.firstlayer().payload
    field_line_parts = []
    header_line_parts = []
    h = 0
    n = 0
    layer = startlayer
    for layer_name in startlayer.layers():
        q = 0
        mem = 0
        for field in layer.fields_desc:
            fld = getattr(layer, field.name)
            if field.name == "load":
                n += 1
                continue
            if fld is None:
                n += 1
                continue
            if hasattr(field, "i2len"):
                if field.name not in ("options", "qd"):
                    try:
                        field_val = field.i2len(layer, fld)
                    except TypeError:
                        field_val = 2
                else:
                    lyr = pkt[layer_name]
                    nextlyr = pkt[layer_name].payload
                    layer_data = linehexdump(lyr, onlyhex=1, dump=True)
                    next_data = linehexdump(nextlyr, onlyhex=1, dump=True)
                    if len(next_data) > 0:
                        layer_data = layer_data.rsplit(next_data, 1)[0]
                    layer_val = layer_data.split(" ")
                    layer_val = [i for i in layer_val if i != ""]
                    layer_val = len(layer_val)
                    field_val = layer_val - q
                    if field_val > 0:
                        field_line_parts.extend([str(n)] * int(field_val))
                        header_line_parts.extend([str(h)] * int(field_val))
                        q += field_val
                        n += 1
                        break
            else:
                field_data = linehexdump(fld, onlyhex=1, dump=True)
                field_data = field_data.split(" ")
                field_data = [i for i in field_data if i != ""]
                field_val = len(field_data)

            if float(field_val) % 1 != 0.0:
                if field.name == "frag":
                    field_val = 2
                else:
                    mem = mem + float(field_val)
                    field_val = 0
            if mem % 1 == 0.0 and mem > 0:
                field_val = mem
                mem = 0

            field_line_parts.extend([str(n)] * int(field_val))
            header_line_parts.extend([str(h)] * int(field_val))
            q += field_val
            n += 1

        h += 1
        layer = layer.payload
        if layer is None:
            break

    return " ".join(field_line_parts), " ".join(header_line_parts)


def process_pcap_to_flows(pcap_path):
    packets = rdpcap(pcap_path)
    flows = defaultdict(list)
    for pkt in packets:
        key = get_flow_key(pkt)
        if key is not None:
            flows[key].append(pkt)
    return flows


def encode_direction(flow_packets):
    mac_counts = defaultdict(int)
    for pkt in flow_packets:
        if hasattr(pkt, "src"):
            mac_counts[pkt.src] += 1
    if not mac_counts:
        return ["2"] * len(flow_packets)
    ranked = sorted(mac_counts.items(), key=lambda x: -x[1])
    most_common_mac = ranked[0][0]
    dirs = []
    for pkt in flow_packets:
        if hasattr(pkt, "src") and pkt.src == most_common_mac:
            dirs.append("1")
        else:
            dirs.append("2")
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Preprocess USTC-TFC2016 dataset for SPARKLE fine-tuning")
    parser.add_argument("--input_dir", required=True, help="Path to USTC-TFC2016-master directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for preprocessed data")
    parser.add_argument("--min_flow_packets", type=int, default=5, help="Minimum packets per flow")
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(f"{out}/packets", exist_ok=True)
    os.makedirs(f"{out}/fields", exist_ok=True)
    os.makedirs(f"{out}/header", exist_ok=True)
    os.makedirs(f"{out}/direction", exist_ok=True)

    manifest = []
    label_names = {}

    for dirname in ("Benign", "Malware"):
        dirpath = os.path.join(args.input_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".pcap"):
                continue
            pcap_path = os.path.join(dirpath, fname)
            label_stem = fname.replace(".pcap", "")
            is_benign = dirname == "Benign"
            label = 0 if is_benign else None

            t0 = time.time()
            print(f"Reading {pcap_path}...", end=" ", flush=True)
            flows = process_pcap_to_flows(pcap_path)
            print(f"{sum(len(v) for v in flows.values())} packets, {len(flows)} raw flows")

            flow_count = 0
            for flow_key, flow_packets in flows.items():
                if len(flow_packets) < args.min_flow_packets:
                    continue

                if not is_benign:
                    label = label_names.get(label_stem)
                    if label is None:
                        label = len(label_names) + 1
                        label_names[label_stem] = label

                flow_id = str(uuid.uuid4())
                hex_lines = []
                field_lines = []
                header_lines = []

                for pkt in flow_packets:
                    hex_lines.append(generate_hex_dump(pkt))
                    fl, hl = generate_field_header_positions(pkt)
                    field_lines.append(fl)
                    header_lines.append(hl)

                dirs = encode_direction(flow_packets)
                with open(f"{out}/packets/{flow_id}.txt", "w") as f:
                    f.write("\n".join(hex_lines) + "\n")
                with open(f"{out}/fields/{flow_id}.txt", "w") as f:
                    f.write("\n".join(field_lines) + "\n")
                with open(f"{out}/header/{flow_id}.txt", "w") as f:
                    f.write("\n".join(header_lines) + "\n")
                with open(f"{out}/direction/{flow_id}.txt", "w") as f:
                    f.write("\n".join(dirs) + "\n")

                manifest.append({
                    "flow": f"{dirname}/{label_stem}/{flow_id}",
                    "packet": os.path.abspath(f"{out}/packets/{flow_id}.txt"),
                    "field": os.path.abspath(f"{out}/fields/{flow_id}.txt"),
                    "header": os.path.abspath(f"{out}/header/{flow_id}.txt"),
                    "direction": os.path.abspath(f"{out}/direction/{flow_id}.txt"),
                    "label": label,
                    "label_name": label_stem,
                    "num_packets": len(flow_packets),
                })
                flow_count += 1
            print(f"  -> {flow_count} flows in {time.time()-t0:.1f}s")

    with open(f"{out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {len(manifest)} total flows saved to {out}/manifest.json")
    if label_names:
        print(f"Malware labels: {label_names}")


if __name__ == "__main__":
    main()
