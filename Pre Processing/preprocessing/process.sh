mkdir $1/split

for i in $(ls $1/*.pcap); do
    echo "Processing $i"
    fol="$1/split/$(date +%A%H%M)"
    mkdir $fol

    ./PcapSplitter -f $i -o $fol -m ip-src-dst
    echo $fol
    python process.py fol
done

