import sys
import os


def processfinal(dire):
    files = 0
    outnumber = 0
    oldnumber = 1
    outdir = ""
    os.makedirs(dire + '/final', exist_ok=True)

    with open("output.csv", "w+") as out:
        for direc in os.scandir(dire):
            working = dire + "/" + direc.name
            if len(working.split('.')) > 1:
                continue
            if direc.name == "final":
                continue
            out.write("Directory: " + direc.name + "\n")
            for filename in os.listdir(working + "/packets"):

                outnumber = files//2000
                if outnumber != oldnumber:
                    outdir = dire + '/final/' + "folder_" + str(outnumber)
                    os.makedirs(outdir + '/packets', exist_ok=True)
                    os.makedirs(outdir + '/fields', exist_ok=True)
                    os.makedirs(outdir + '/direction', exist_ok=True)
                    os.makedirs(outdir + '/header', exist_ok=True)
                    oldnumber = outnumber

                packets = 0
                i = 0

                if filename.endswith(".txt"):
                    with open(working + "/packets/" + filename) as f:
                        a = f.read().split('\n')
                    with open(working + "/fields/" + filename) as f:
                        b = f.read().split('\n')
                    with open(working + "/direction/" + filename) as f:
                        c = f.read().split('\n')
                    with open(working + "/header/" + filename) as f:
                        d = f.read().split('\n')

                    print(filename)

                    while i < len(a):
                        if len(a[i].split(' ')) > 510:
                            a.pop(i)
                            b.pop(i)
                            c.pop(i)
                            d.pop(i)
                            i = i - 1
                        packets = packets + 1
                        i = i + 1

                    files = files + 1
                    out.write("File " + str(files) + ": " + filename +
                              "\nPackets: " + str(packets) + "\n")

                    if len(a) == 0:
                        continue

                    with open(outdir + "/packets/" + "flow_" + str(files), "w") as f:
                        j = 0
                        while j < len(a):
                            f.write(a[j] + '\n')
                            j = j + 1
                        f.close()
                    with open(outdir + "/fields/" + "fields_pos_" + str(files), "w") as f:
                        j = 0
                        while j < len(b):
                            f.write(b[j] + '\n')
                            j = j + 1
                        f.close()
                    with open(outdir + "/direction/" + "direction_" + str(files), "w") as f:
                        j = 0
                        while j < len(c):
                            f.write(c[j] + '\n')
                            j = j + 1
                        f.close()
                    with open(outdir + "/header/" + "header_pos_" + str(files), "w") as f:
                        j = 0
                        while j < len(d):
                            f.write(d[j] + '\n')
                            j = j + 1
                        f.close()
                    continue
                else:
                    continue
            out.write("\n")
        out.close()


processfinal(sys.argv[1])
print("done")
