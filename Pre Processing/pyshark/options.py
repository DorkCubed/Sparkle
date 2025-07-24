import pyshark

def analyze_options(filename):

    with open("Output1.txt", "w") as f:
        n = 1
        cap = pyshark.FileCapture(filename)

        for packet in cap:
            f.write(f"\nPacket {n} Summary - Length: {len(packet)}\n")
            n = n + 1
            if hasattr(packet, 'tcp'):
                try:
                    val = packet.tcp.options
                    #val = val.replace(":", "")
                    f.write(f"      Field: options {val}\n")
                except(AttributeError):
                    continue

# Example usage
filename = r"p1.pcap"
analyze_options(filename)
