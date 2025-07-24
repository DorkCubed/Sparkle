from scapy.all import *

# this function converts the pcap file into a text file with the hexdump


def process_pcap(filename, outfile):

    # reading the packets and initializing the output file
    print(f'Generating packet hex for {filename}')
    packets = rdpcap(filename)

    outfile = outfile.rsplit("/", 1)
    if len(outfile) < 2:
        outfile.append(outfile[0])
        outfile[0] = '.'

    os.makedirs(outfile[0] + '/direction', exist_ok=True)
    os.makedirs(outfile[0] + '/packets', exist_ok=True)

    with open((outfile[0] + "/packets/" + outfile[1]), 'w') as f:
        with open((outfile[0] + "/direction/" + outfile[1]), 'w') as g:

            # for loop to iterate through each packet and layer
            n = 0
            for pkt in packets:
                layer = pkt.firstlayer()
                if hasattr(pkt, 'src'):
                    dirdata = str(pkt.src) + '\n'
                else:
                    dirdata = '0\n'

                # removing the frame layer
                layer = layer.payload
                # getting the hexdump of the packet
                pdata = (linehexdump(layer, onlyhex=1, dump=True))

                # removing load data if present
                for layer_name in layer.layers():
                    for field in layer_name.fields_desc:
                        if field.name == 'load':
                            loaddata = getattr(pkt, field.name)
                            loaddata = linehexdump(
                                loaddata, onlyhex=1, dump=True)
                            pdata = pdata.rsplit(loaddata, 1)
                            if len(pdata) > 1:
                                pdata = pdata[0] + pdata[1]
                            else:
                                pdata = pdata[0]

                # writing the hexdump to the output file
                if pdata == '':
                    pdata = 'emp'

                g.write(dirdata)
                f.write(pdata + '\n')
                n = n + 1


# filename = input("Enter the path of the pcap file: ")
# outfile = input("Enter the path of the output file: ")
# process_pcap(filename, outfile)
