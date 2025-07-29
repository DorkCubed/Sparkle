# Function to measure packet length in terms of hex values in the hex dump file
def measure_packet_hex_lengths(file_path):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            # Split line by spaces to get individual hex values
            hex_values = line.strip().split()
            # Count the number of hex values
            packet_length = len(hex_values)
            # print(f"Packet {line_number}: Length = {packet_length} hex values")
            if(packet_length >= 512):
                print(f"Packet {line_number}: Length = {packet_length} hex values")

# Replace 'flow.txt' with the path to your hex dump file
measure_packet_hex_lengths('/home/satvik/spark/spark2/packets/packet_0.txt')
