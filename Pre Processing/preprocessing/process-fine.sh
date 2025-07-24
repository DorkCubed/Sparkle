mkdir $1/split

for i in $(ls $1/*.pcap); do
    echo "Processing $i"
    j=$(basename $i)
    fol="$1/split/${j%.*}"
    mkdir $fol

    ./PcapSplitter -f $i -o $fol -m ip-src-dst
    python3 process.py $fol
done