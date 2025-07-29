import torch
import numpy as np

with open('/home/satvik/spark/spark/packet_seq.txt', 'r') as file:
    lines = file.readlines()

data = []
for line in lines:
    row = list(map(int, line.split()))
    data.append(row)

data_np = np.array(data)
data_np_reshaped = data_np.reshape(4, -1) 
data_tensor = torch.tensor(data_np_reshaped, dtype=torch.int)

print("Tensor shape:", data_tensor.shape)
print("Tensor data:\n", data_tensor)

with open('/home/satvik/spark/spark/padded_field_pos.txt', 'r') as file:
    lines = file.readlines()

data_f = []
for line in lines:
    row = list(map(int, line.split()))
    data_f.append(row)

data_np_f = np.array(data_f)
data_np_reshaped_f = data_np_f.reshape(4, -1) 
data_tensor_f = torch.tensor(data_np_reshaped_f, dtype=torch.int)

print("Tensor shape:", data_tensor_f.shape)
print("Tensor data:\n", data_tensor_f)

def apply_sfbo_masking(packet_seq, max_span_lenth=6, padding_value=0):
    valid_indices = 