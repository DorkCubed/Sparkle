import sys
import os
from test import process_fields
from pcap_to_packet import process_pcap

for filename in os.listdir(sys.argv[1]):
    if filename.endswith(".pcap"):
        fname = sys.argv[1] + "/" + filename
        oname = sys.argv[1] + "/" + filename.replace(".pcap", ".txt")
        if os.path.getsize(fname) > 500:
            process_fields(fname, oname)
            process_pcap(fname, oname)
        continue
    else:
        continue
