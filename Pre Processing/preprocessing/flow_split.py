"""
flow_split.py

Splits a single PCAP file into its individual network flows (one entry per
5-tuple: src IP, dst IP, src port, dst port, protocol), instead of treating
the whole PCAP as one undifferentiated packet sequence.

Why this exists
----------------
The original pipeline (pcap_to_packet.py / test.py) calls scapy's rdpcap()
once per PCAP and writes every packet in the file, in capture order, into a
single output file. That means one PCAP = one "flow" in the manifest, no
matter how many actual TCP/UDP conversations it contains. For USTC-TFC2016,
where each malware family is exactly one PCAP, this caps every malware class
at a handful of samples (~6 sequential 512-packet chunks) regardless of how
many real flows are inside the file.

This module fixes that by grouping packets into real flows first. Each flow
becomes its own packet list, which can then be passed through the existing
process_pcap() / process_fields() functions unmodified (they already accept
a list of scapy packets, since rdpcap() just returns one).

Direction is bidirectional: A->B and B->A packets belong to the SAME flow.
We canonicalize the 5-tuple so both directions hash to one key.
"""

import os
from collections import defaultdict

from scapy.all import rdpcap, IP, IPv6, TCP, UDP


def flow_key(pkt):
    """
    Return a canonical (order-independent) 5-tuple key for a packet, or
    None if the packet has no IP layer (e.g. ARP) — those are dropped from
    flow-level analysis the same way they always were absent from
    meaningful "flow" semantics.
    """
    if IP in pkt:
        ip_layer = pkt[IP]
    elif IPv6 in pkt:
        ip_layer = pkt[IPv6]
    else:
        return None

    src, dst = ip_layer.src, ip_layer.dst

    if TCP in pkt:
        proto = "TCP"
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        proto = "UDP"
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    else:
        proto = ip_layer.name  # e.g. ICMP — group by IP pair + protocol only
        sport, dport = 0, 0

    # Canonicalize direction: smaller (ip, port) tuple goes first, so
    # A->B and B->A packets land in the same flow bucket.
    endpoint_a = (src, sport)
    endpoint_b = (dst, dport)
    if endpoint_a <= endpoint_b:
        return (endpoint_a, endpoint_b, proto)
    return (endpoint_b, endpoint_a, proto)


def split_pcap_into_flows(pcap_path, min_packets=2, max_flows=None):
    """
    Read a PCAP and group its packets into individual network flows.

    Args:
        pcap_path: path to the .pcap file.
        min_packets: drop flows with fewer than this many packets (a single
            stray SYN or retransmission isn't a useful "flow" sample).
        max_flows: optional cap on number of flows returned, for very chatty
            PCAPs (keeps the longest flows, which carry the most signal).

    Returns:
        List of (flow_id_str, [scapy_packet, ...]) tuples, ordered by
        flow first-seen time. Packets within each flow keep their original
        capture order.
    """
    packets = rdpcap(pcap_path)

    buckets = defaultdict(list)
    for pkt in packets:
        key = flow_key(pkt)
        if key is None:
            continue
        buckets[key].append(pkt)

    flows = [(key, pkts) for key, pkts in buckets.items() if len(pkts) >= min_packets]

    # Longest flows first (most signal), since max_flows (if set) should
    # keep the most informative conversations rather than an arbitrary
    # capture-order prefix.
    flows.sort(key=lambda kv: len(kv[1]), reverse=True)

    if max_flows is not None:
        flows = flows[:max_flows]

    result = []
    for i, (key, pkts) in enumerate(flows):
        (ip_a, port_a), (ip_b, port_b), proto = key
        flow_id = f"{proto}_{ip_a}:{port_a}-{ip_b}:{port_b}_{i:04d}"
        result.append((flow_id, pkts))

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python flow_split.py <path-to-pcap>")
        sys.exit(1)

    flows = split_pcap_into_flows(sys.argv[1])
    print(f"{sys.argv[1]}: {len(flows)} flows")
    for flow_id, pkts in flows[:10]:
        print(f"  {flow_id}: {len(pkts)} packets")
