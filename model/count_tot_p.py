def count_packets_in_hex_dump(file_path):
    packet_count = 0

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  
                    packet_count += 1
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    print(f"number of packets: {packet_count}")

file_path = '/home/satvik/spark/spark/2013-10-21_capture-1-only-dns-0001(packet).txt' 
count_packets_in_hex_dump(file_path)
