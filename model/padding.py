with open('/home/satvik/spark/spark/fields/field_pos.txt', 'r') as file:
    data = file.read().strip()

fields = data.splitlines()
fields = [list(map(int, field.split())) for field in fields]
max_length = max(len(field) for field in fields)
padded_fields = [field + [0] * (max_length - len(field)) for field in fields]

with open('padded_field_pos.txt', 'w') as file:
    for field in padded_fields:
        file.write(' '.join(map(str, field)) + '\n')

    