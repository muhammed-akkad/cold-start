import cupy as cp
import torch
import json

def load_and_retrieve_tensors(json_file_path):
    with open(json_file_path, 'r') as f:
        tensor_index = json.load(f)

    retrieved_tensors = {}

    for name, (offset, size, shape, stride, dtype_str) in tensor_index.items():
        dtype = getattr(torch, dtype_str.split('.')[-1])

        # Allocate raw memory with CuPy and create a tensor from it
        d_storage = cp.cuda.MemoryPointer(cp.cuda.memory.Memory(offset), size)
        d_array = cp.ndarray(shape, dtype=dtype, memptr=d_storage)

        # Convert CuPy array to PyTorch tensor
        d_tensor = torch.as_tensor(d_array.get(), device='cuda').view(*shape).type(dtype)

        retrieved_tensors[name] = d_tensor

    return retrieved_tensors
tensors = load_and_retrieve_tensors('./tensor_index.json')