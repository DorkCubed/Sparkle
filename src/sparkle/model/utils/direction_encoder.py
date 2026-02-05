import ipaddress
from collections import Counter

def encode_mac_address(mac_str, mac1, mac2):
    mac = mac_str.strip().lower()  # Normalize
    if mac == mac1:
        return "1"
    elif mac == mac2:
        return "2"
    return mac  # Return original if not top 2

def encode_ip_address(ip_str, ip1, ip2):
    ip_str = ip_str.strip()
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return ip_str  # Return original if invalid
    if ip == ip1:
        return "1"
    elif ip == ip2:
        return "2"
    return ip_str

def encode_file(filename, mode="mac"):
    """
    Encode the two most common MAC or IP addresses in a file.

    Args:
        filename (str): Path to the input file.
        mode (str): 'mac' or 'ip', depending on what to encode.
    """
    try:
        with open(filename, "r") as file:
            lines = [line.strip() for line in file]
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    if mode == "mac":
        counts = Counter(lines)
        if len(counts) < 2:
            raise ValueError(f"Not enough unique MAC addresses in {filename}")
        mac1, mac2 = [mac.lower() for mac, _ in counts.most_common(2)]
        encoded_lines = [encode_mac_address(line, mac1, mac2) for line in lines]

    elif mode == "ip":
        counts = Counter(lines)
        if len(counts) < 2:
            print("Error: Not enough unique IP addresses.")
            return

        ip_str1, ip_str2 = [ip for ip, _ in counts.most_common(2)]
        try:
            ip1 = ipaddress.ip_address(ip_str1)
            ip2 = ipaddress.ip_address(ip_str2)
        except ValueError:
            print("Error: Invalid IP addresses found.")
            return
        encoded_lines = [encode_ip_address(line, ip1, ip2) for line in lines]

    else:
        print("Error: mode must be 'mac' or 'ip'.")
        return

    try:
        with open(filename, 'w') as out_file:
            for line in encoded_lines:
                out_file.write(line + "\n")
        # print(f"Encoded data has been written to {filename}")
    except IOError:
        print(f"Error: Unable to write to {filename}")
