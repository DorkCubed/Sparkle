def sentences_to_array(filename, outfile):
    with open(filename, 'r') as file:
        with open(outfile, 'w') as f:
            text = file.read()
            for paragraph in text.split('\n\n'):
                '''
                f.write(paragraph + '\n' + '............' + '\n')
                '''
                packet = []
                progress = 0
                mem = 0

                for sentence in paragraph.split('\n'):
                    words = []
                    for word in sentence.split(' '):
                        words.append(word)
                    if len(words) > 2:
                        packet = words
                        f.write('\n')
                        #f.write(str(packet) + '\n')
                    if len(words) == 2:
                        l = words[1]
                        #f.write(l + '\n')

                        if words[0] == 'load':
                            progress = progress + int(l)
                            
                        else:
                            if float(l) % 1 != 0.0:
                                mem = mem + float(l)

                            if mem % 1 == 0.0 and mem > 0:
                                l = int(mem)
                                mem = 0

                            if float(l) % 1 == 0.0:
                                l = int(float(l))
                                start = progress
                                end = progress + l
                                f.write(str(packet[start:end]) + '\n')
                                progress = progress + l

filename = 'output1.txt'
outfile = 'output2.txt'
sentences_to_array(filename, outfile)