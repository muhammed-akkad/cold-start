import torch
import os
import shutil
import json
import cuda_saver
import torchvision.models as models
import ctypes
import time
from torchvision.models import MobileNet_V3_Large_Weights

# Function to load IPC handle from a file
def load_ipc_handle(filename):
    with open(filename, 'rb') as f:
        ipc_handle_bytes = f.read()
    # Reconstruct cudaIpcMemHandle_t from bytes
    ipc_handle = (ctypes.c_char * len(ipc_handle_bytes)).from_buffer_copy(ipc_handle_bytes)
    return ipc_handle

def load_model_from_ipc(model_class, tensor_index_file, ipc_handles_dir):
    # Load tensor metadata
    with open(tensor_index_file, 'r') as f:
        tensor_index = json.load(f)

    state_dict = {}
    for name, meta in tensor_index.items():
        shape = meta[2]
        dtype_str = meta[4] 
        dtype_name = dtype_str.replace('torch.', '') 
        dtype = getattr(torch, dtype_name) 
        dtype_code = dtype_to_code(dtype)
        ipc_handle_file = os.path.join(ipc_handles_dir, f"{name}_ipc_handle.bin")

        # Load the IPC handle
        ipc_handle = load_ipc_handle(ipc_handle_file)
        ipc_handle_str = ipc_handle.raw  # Get the raw bytes

        # Create the tensor from the IPC handle
        tensor = cuda_saver.tensor_from_ipc_handle(ipc_handle_str, shape, dtype_code)
        
        state_dict[name] = tensor

    # Instantiate the model and load the state dict
    model = model_class().cuda()
    model.load_state_dict(state_dict)
    return model

def dtype_to_code(dtype):
    dtype_map = {
        torch.float32: 6, 
        torch.float64: 7, 
        torch.int32: 3,    
        torch.int64: 4,    
    }
    return dtype_map.get(dtype, -1)
def delete_pytorch_cache():
    # Delete PyTorch's cached model weights
    cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

def clear_os_cache():
    # Clear the OS file system cache (Linux only)
    os.system('sudo sync')
    os.system('sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"')
    
def main():
    delete_pytorch_cache()
    clear_os_cache()

    model = models.mobilenet_v3_large().cuda()
    # Load the model normally for comparison
    start_time_normal = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal_model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).to(device)
    end_time_normal = time.time()

    normal_model.eval()


    start_time_ipc = time.time()
    # Load state dict from IPC using the C++ extension
    state_dict = cuda_saver.load_model_from_ipc('tensor_index.json', 'handlers_gpu')

    # Instantiate the model and load the state dict

    model.load_state_dict(state_dict)
    end_time_ipc = time.time()

    model.eval()
    # Calculate loading times
    normal_load_time = end_time_normal - start_time_normal
    ipc_load_time = end_time_ipc - start_time_ipc

    print(f"Model loaded normally in {normal_load_time:.6f} seconds")
    print(f"Model loaded using IPC in {ipc_load_time:.6f} seconds")
if __name__ == "__main__":
    main() 