
from numba import cuda
import pyarrow as pa
import torch
import numpy as np
from collections import OrderedDict
print(torch.cuda.is_available())
# Step 1: Load the .pth file into PyTorch tensor
def load_pth_to_tensor(file_path):
    # Load the .pth file
    tensor = torch.load(file_path)
    return tensor

# Step 2: Transfer the tensor or state dict to GPU
def transfer_data_to_gpu(data):
    if isinstance(data, torch.Tensor):
        # If it's a tensor, transfer it to GPU
        if not data.is_cuda:
            data = data.cuda()
        gpu_array = cuda.to_device(data.cpu().numpy())
    elif isinstance(data, OrderedDict):
        # If it's an OrderedDict (like a state_dict), iterate and transfer each tensor
        gpu_data = OrderedDict()
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if not value.is_cuda:
                    value = value.cuda()
                gpu_data[key] = cuda.to_device(value.cpu().numpy())
        return gpu_data
    else:
        raise TypeError("Unsupported data type. Expected Tensor or OrderedDict.")
    
    return gpu_array


# Step 5: Example usage
file_path_1 = 'sub_model_0.pth'
file_path_2 = 'sub_model_1.pth'

# Load .pth files into PyTorch tensors
tensor_1 = load_pth_to_tensor(file_path_1)
tensor_2 = load_pth_to_tensor(file_path_2)

# Transfer the data to GPU memory
gpu_data_1 = transfer_data_to_gpu(tensor_1)
gpu_data_2 = transfer_data_to_gpu(tensor_2)
