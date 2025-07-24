from scapy.all import *


def process_fields(filename, outfile):

    # reading the packets and initializing the output file
    print(f"Generating fields for {filename}")
    packets = rdpcap(filename)

    outfile = outfile.rsplit("/", 1)
    if len(outfile) < 2:
        outfile.append(outfile[0])
        outfile[0] = "."

    os.makedirs(outfile[0] + "/header", exist_ok=True)
    os.makedirs(outfile[0] + "/fields", exist_ok=True)

    with open((outfile[0] + "/fields/" + outfile[1]), "w") as f:
        with open((outfile[0] + "/header/" + outfile[1]), "w") as g:

            # for loop to iterate through each packet and layer
            for pkt in packets:
                framelayer = pkt.firstlayer()
                startlayer = framelayer.payload

                h = 0
                n = 0
                layer = startlayer
                for layer_name in startlayer.layers():
                    q = 0
                    mem = 0
                    for field in layer.fields_desc:
                        fld = getattr(layer, field.name)
                        if field.name == "load":
                            n = n + 1
                            continue
                        if fld == None:
                            n = n + 1
                            continue
                        if hasattr(field, "i2len"):
                            if field.name != "options" and field.name != "qd":
                                try:
                                    field_val = field.i2len(layer, fld)
                                except TypeError:
                                    field_val = 2

                            else:
                                lyr = pkt[layer_name]
                                nextlyr = pkt[layer_name].payload
                                layer_data = linehexdump(
                                    lyr, onlyhex=1, dump=True)
                                next_data = linehexdump(
                                    nextlyr, onlyhex=1, dump=True)
                                if len(next_data) > 0:
                                    layer_data = layer_data.rsplit(
                                        next_data, 1)[0]

                                layer_val = layer_data.split(" ")
                                layer_val = [i for i in layer_val if i != ""]
                                layer_val = len(layer_val)
                                field_val = layer_val - q
                                if field_val > 0:
                                    f.write((str(n) + " ") * int(field_val))
                                    g.write((str(h) + " ") * int(field_val))
                                    q = q + field_val
                                    n = n + 1
                                    break
                        else:
                            field_data = linehexdump(fld, onlyhex=1, dump=True)
                            field_data = field_data.split(" ")
                            field_data = [i for i in field_data if i != ""]
                            field_val = len(field_data)
                            print(field.name + str(field_val))

                        if float(field_val) % 1 != 0.0:
                            if field.name == "frag":
                                field_val = 2
                            else:
                                mem = mem + float(field_val)
                                field_val = 0
                        if mem % 1 == 0.0 and mem > 0:
                            field_val = mem
                            mem = 0

                        f.write((str(n) + " ") * int(field_val))
                        g.write((str(h) + " ") * int(field_val))
                        q = q + field_val
                        n = n + 1

                    h = h + 1
                    layer = layer.payload
                    if layer == None:
                        break

                f.write("\n")
                g.write("\n")


# filename = input("Enter the path of the pcap file: ")
# outfile = input("Enter the path of the output file: ")
# process_fields(filename, outfile)
