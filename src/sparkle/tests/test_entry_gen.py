entry = {'packet': ['/home/ubuntu/projects/Sparkle/dataset/packets/packet_0.txt'],
         'header': ['/home/ubuntu/projects/Sparkle/dataset/headers/header_pos_0.txt'],
         'field': ['/home/ubuntu/projects/Sparkle/dataset/fields/field_pos_0.txt'],
         'direction': ['/home/ubuntu/projects/Sparkle/dataset/direction/direction_1.txt']}
entry = {k: (v[0] if isinstance(v, list) else v) for k, v in entry.items()}

print(entry)