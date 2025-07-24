from scapy.all import *
import pyshark

def analyze_packet_lengths(filename):
    """Analyzes a PCAP file and prints the length of each field in each packet.

    Args:
        filename: The path to the PCAP file.
    """

    with open("Output.txt", "w") as f:
        n = 1
        packets = pyshark.FileCapture(filename)

        for packet in packets:
            f.write(f"\nPacket {n} Summary - Length: {len(packet)}\n")
            layer = packet
            while layer:
                f.write(f"  Layer: {layer.name}\n")
                for field in layer.fields_desc:
                    field_name = field.name
                    field_val = getattr(packet, field.name)  # Access field
                    if field_name == 'options':
                        options_d = packet[layer.name].options
                        field_length = linehexdump(options_d, onlyhex=1, dump=True)  

                    elif hasattr(field, 'i2len'):
                        field_length = field.i2len(packet, field_val)
                    else:
                        field_length = 0
                        #print(layer)
                    f.write(f"      Field: {field_name} Length: {field_length}\n")
                layer = layer.payload
            n = n + 1


# Example usage
filename = r"p1.pcap"
analyze_packet_lengths(filename)
