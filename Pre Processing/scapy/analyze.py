from scapy.all import *

'''
def field_length(fld):
    leng = ' '
    FieldLenField(leng, None, length_of=fld)
    __build_class__(leng)
    return len(leng)
'''

def process_pcap(file_name):

    #reading the packets and initializing the output file
    print(f'Opening {file_name}')
    packets = rdpcap(file_name)
    with open('output1.txt', 'w') as f:
        
        #for loop to iterate through each packet and layer
        for pkt in packets:

            #printing the hexdump of the packet
            pdata = (linehexdump(pkt, onlyhex=1, dump=True) + '\n')
            f.write(pdata)
            
            '''
            binary = bin(int.from_bytes(pdata.encode('utf 8'), 'small'))
            f.write(str(binary) + '\n')
            '''

            #iterating through each layer and field
            for layer in pkt.layers():
                for field in layer.fields_desc:

                    #writing the values
                    '''
                    can't do 
                    if field.name != 'load':
                    because the other function will ignore the load in the packet hex dump otherwise
                    '''
                    f.write(str(field.name) + ' ')
                    field_val = field.i2len(pkt, getattr(pkt, field.name))
                    f.write(str(field_val) + '\n')

            f.write('\n')
                        
#main function

file = input('Enter the path of the pcap file: ')
process_pcap(file)