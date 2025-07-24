from scapy.all import *

#this function converts the pcap file into a text file with the hexdump and field lengths
def process_pcap(filename):

    #reading the packets and initializing the output file
    print(f'Opening {filename}')
    packets = rdpcap(filename)
    with open('temp_file.txt', 'w') as f:
        
        #for loop to iterate through each packet and layer
        for pkt in packets:

            #printing the hexdump of the packet
            pdata = (linehexdump(pkt, onlyhex=1, dump=True) + '\n')
            f.write(pdata)
            
            #iterating through each layer and field
            for layer in pkt.layers():
                for field in layer.fields_desc:

                    #writing the values
                    '''
                    you can't do 
                    if field.name != 'load':
                    because the split function will ignore the load in the packet hex dump otherwise
                    '''
                    if field.name == 'options':
                        field_val = len(pkt.options)
                    elif hasattr(field, 'i2len'):
                        field_val = field.i2len(pkt, getattr(pkt, field.name))
                    else:
                        field_val = 0
                
                    f.write(str(field.name) + ' ')
                    f.write(str(field_val) + '\n')

            f.write('\n')
    
    print('A temporary file temp_file.txt has been generated')

#this function convers the output of the process_pcap function into field representations according to the field lengths
def sentences_to_array(filename, outfile):

    #opening the input and output files
    with open(filename, 'r') as file:
        with open(outfile, 'w') as f:

            #reading the input file and iterating through each packet
            text = file.read()
            for paragraph in text.split('\n\n'):
               
                #initializing the packet array and progress counter
                packet = []
                progress = 0
                skipframe = 0
                mem = 0

                #iterating through each line in split up packet
                for sentence in paragraph.split('\n'):

                    words = []

                    for word in sentence.split(' '):
                        words.append(word)

                    if len(words) > 2:
                        packet = words
                        f.write('\n')

                    if len(words) == 2:
                        l = words[1]

                        #skipping the load field
                        if words[0] == 'load':
                            progress = progress + int(l)
                            
                        else:

                            #the mem variable is used to store the length of the field if it is less than a byte
                            if float(l) % 1 != 0.0:
                                mem = mem + float(l)
                            if mem % 1 == 0.0 and mem > 0:
                                l = int(mem)
                                mem = 0

                            if float(l) % 1 == 0.0 and float(l) > 0:
                                l = int(float(l))
                                start = progress
                                end = progress + l
                                if skipframe > 2:
                                    f.write(str(packet[start:end]) + '\n')
                                progress = progress + l
                                skipframe = skipframe + 1


#calling the functions
file = input('Enter the path of the pcap file: ')
outfile = input('Enter the path of the output file: ')
process_pcap(file)
sentences_to_array('temp_file.txt', outfile)