from scapy.all import *

#this function converts the pcap file into a text file with fields
def process_fields(filename, outfile):

    #reading the packets and initializing the output file
    print(f'Generating fields for {filename}')
    packets = rdpcap(filename)

    outfile = outfile.rsplit("/", 1)
    if len(outfile) < 2:
        outfile.append(outfile[0])
        outfile[0] = '.'

    os.makedirs(outfile[0] + '/header', exist_ok=True)
    os.makedirs(outfile[0] + '/fields', exist_ok=True)

    with open((outfile[0] + "/fields/" + outfile[1]), 'w') as f:
        with open((outfile[0] + "/header/" + outfile[1]), 'w') as g:    


            #for loop to iterate through each packet and layer
            for pkt in packets:
                layer = pkt.firstlayer()
                layer = layer.payload

                k = 0
                n = 0
                for layer in layer.layers():
                    mem = 0
                    for field in layer.fields_desc:
                        fld = getattr(pkt, field.name)
                        if field.name == 'load':
                            continue
                        if fld == None:
                            continue
                        if hasattr(field, 'i2len'):
                            try: 
                                field_val = field.i2len(pkt, fld)
                            except TypeError: 
                                field_val = 2
                        else:
                            field_data = linehexdump(fld, onlyhex=1, dump=True)
                            field_data = field_data.split(' ')
                            field_data = [i for i in field_data if i != '']
                            field_val = len(field_data)
                            
                        if float(field_val) % 1 != 0.0:
                            mem = mem + float(field_val)
                            field_val = 0
                        if mem % 1 == 0.0 and mem > 0:
                            field_val = mem
                            mem = 0
                        
                        f.write((str(n) + ' ') * int(field_val))
                        n = n + 1
                        g.write((str(k) + ' ') * int(field_val))

                    k = k + 1
                f.write('\n')
                g.write('\n')

#filename = input("Enter the path of the pcap file: ")
#outfile = input("Enter the path of the output file: ")
#process_fields(filename, outfile)