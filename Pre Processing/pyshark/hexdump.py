import pyshark

def print_header_data(pkt):
    with open('hexdump.txt', 'w') as f:
        for layer in pkt.layers:
            for field in layer.field_names:
                try:
                    field_value = layer.get_field_value(field)
                    if field_value is not None:
                        f.write(f'{layer.layer_name} {field} length: {(field_value)}\n')
                except AttributeError:
                    print('AttributeError')
                    continue
        f.write('\n')

cap = pyshark.FileCapture(r'C:\Users\Cubicle\Documents\Code\Sparkle\pyshark\pcap.pcap')
cap.apply_on_packets(print_header_data)