import re

def count_mac_address_lines(file_path):
    # Regular expression to match MAC address format
    mac_address_pattern = re.compile(r'^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$')
    mac_count = 0

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if mac_address_pattern.match(line.strip()):
                    mac_count += 1
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    print(f"The number of MAC address lines in the file is: {mac_count}")

# Example usage
file_path = '/home/satvik/spark/spark/direction/2013-10-21_capture-1-only-dns-0001(direction).txt'  # Replace with your file path
count_mac_address_lines(file_path)
