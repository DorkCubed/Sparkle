def count_hex_values_in_file(file_path):
    hex_count = 0

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into hex values based on spaces
                hex_values = line.strip().split()
                hex_count += len(hex_values)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    print(f"The number of hex values in the file is: {hex_count}")

# Example usage
file_path = '/home/satvik/spark/spark/packets/packet.txt'  # Replace with your file path
count_hex_values_in_file(file_path)
