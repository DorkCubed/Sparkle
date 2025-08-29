import ipaddress
from collections import Counter


def encode_mac_address(mac_str, mac1, mac2):
    mac = mac_str.lower()  # Convert to lowercase for consistency
    if mac == mac1:
        return "1"
    elif mac == mac2:
        return "2"
    else:
        return mac  # Return the original MAC if it's not one of the two most common


def encode_ip_address(ip_str, ip1, ip2):
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return ip_str  # Return the original string if it's not a valid IP address

    if ip == ip1:
        encoded = "1"
    elif ip == ip2:
        encoded = "2"
    else:
        return ip_str  # Return the original string if it's not one of the two IP addresses

    return encoded


if __name__ == "__main__":
    # filename = input("Enter the name of the text file: ")
    filename = r"/home/satvik/spark/spark2/direction/direction_0.txt"
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        exit()

    # # Get the two most common IP addresses from the file
    # ip_counts = Counter(line.strip() for line in lines)
    # ip1_str, ip2_str = [ip for ip, count in ip_counts.most_common(2)]

    # try:
    #     ip1 = ipaddress.ip_address(ip1_str)
    #     ip2 = ipaddress.ip_address(ip2_str)
    # except ValueError:
    #     print("Error: Invalid IP addresses found in the file.")
    #     exit()

    # encoded_lines = []
    # for line in lines:
    #     ip_str = line.strip()
    #     encoded_ip = encode_ip_address(ip_str, ip1, ip2)
    #     encoded_lines.append(encoded_ip)

    # print("Encoded IP addresses:")
    # for line in encoded_lines:
    #     print(line)

    # out_file = open(filename, 'w')
    # for encoded_line in encoded_lines:
    #     file.write(encoded_line+"\n")

    # Get the two most common MAC addresses from the file
    mac_counts = Counter(lines)
    if len(mac_counts) < 2:
        print("Error: Not enough unique MAC addresses in the file.")
        exit()

    mac1, mac2 = [mac for mac, count in mac_counts.most_common(2)]

    encoded_lines = [encode_mac_address(line, mac1, mac2) for line in lines]

    print("Encoded MAC addresses:")
    for line in encoded_lines:
        print(line)

    # Write the encoded lines back to the file
    try:
        with open(filename, 'w') as out_file:
            for encoded_line in encoded_lines:
                out_file.write(encoded_line + "\n")
        print(f"Encoded data has been written to {filename}")
    except IOError:
        print(f"Error: Unable to write to {filename}")

out_file.close()
