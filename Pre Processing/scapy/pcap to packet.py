from scapy.all import *

#this function converts the pcap file into a text file with the hexdump
def process_pcap(filename, outfile):

    #reading the packets and initializing the output file
    print(f'Opening {filename}')
    packets = rdpcap(filename)
    with open(outfile, 'w') as f:
        
        #for loop to iterate through each packet and layer
        for pkt in packets:
            layer = pkt.firstlayer()

            #removing the frame layer
            layer = layer.payload;
            #getting the hexdump of the packet
            pdata = (linehexdump(layer, onlyhex=1, dump=True))

            #removing load data if present           
            if hasattr(pkt, 'load'):
                    loaddata = pkt.load
                    loaddata = linehexdump(loaddata, onlyhex=1, dump=True)
                    pdata = pdata.rsplit(loaddata, 1)
                    pdata = pdata.pop(1)
        
            #writing the hexdump to the output file
            f.write(pdata + '\n') if pdata != '' else None


filename = input("Enter the path of the pcap file: ")
outfile = input("Enter the path of the output file: ")
process_pcap(filename, outfile)

