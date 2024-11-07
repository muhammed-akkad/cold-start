import errno
import mmap
import os
import json
import time
import torch
import ctypes
import cuda_saver
import posix_ipc

def load_model(model_class, model_path):
    """
    Loads a model's state dictionary from GPU, CPU, and Disk, and returns the model.

    Args:
        model_class: The class of the model to instantiate.
        model_path (str): Path where the model parts are saved.

    Returns:
        model: The model with the state dictionary loaded.
    """
    # Load tensor index
    with open(os.path.join(model_path, "tensor_index.json"), 'r') as f:
        tensor_index = json.load(f)

    state_dict = {}

    # Load tensors based on their location
    for name, meta in tensor_index.items():
        location = meta['location']
        size = meta['size']
        shape = meta['shape']
        dtype_str = meta['dtype']
        dtype = getattr(torch, dtype_str.split('.')[1])  # Convert 'torch.float32' to 'float32'

        if location == 'gpu':
            # Load tensor from GPU using IPC handle
            ipc_handle_file = os.path.join(model_path, 'handlers_gpu', f"{name}_ipc_handle.bin")
            tensor = cuda_saver.load_model_tensor(ipc_handle_file, shape, dtype)
            state_dict[name] = tensor

        elif location == 'cpu':
            # Load tensor from shared CPU memory using posix_ipc
            shm_name = f"/{name}"

            try:
                # Open the shared memory object
                shm = posix_ipc.SharedMemory(shm_name, flags=posix_ipc.O_RDONLY)
            except posix_ipc.ExistentialError:
                raise FileNotFoundError(f"Shared memory {shm_name} not found")

            # Map the shared memory into the process's address space
            fd = shm.fd
            h_memory = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
            # Close the file descriptor as it's no longer needed
            shm.close_fd()

            # Create tensor from shared memory (keep on CPU for now)
            buffer = memoryview(h_memory)
            tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
            state_dict[name] = tensor

        elif location == 'disk':
            # Load tensor directly from disk
            filename = os.path.join(model_path, 'tensors_data', f"{name}_data.bin")
            with open(filename, 'rb') as f:
                buffer = f.read()
            tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
            # Keep on CPU for now
            state_dict[name] = tensor

        else:
            raise ValueError(f"Unknown location '{location}' for tensor '{name}'")

    # Instantiate the model
    model = model_class()

    # Load state_dict
    model.load_state_dict(state_dict)

    # Move the entire model to GPU
    model.to('cuda')

    return model

def main():
    from torchvision import models

    model_path = './'
    model_class = models.mobilenet_v3_large
    # Prepare input data
    input_tensor = torch.randn(1, 3, 224, 224).to('cuda')
    torch.cuda.init()
    torch.cuda.synchronize()
    # ---------------------------
    # Custom Loading Method
    # ---------------------------
    start_time = time.time()
    # Load the model with parameters split between GPU, CPU, and Disk
    custom_model = load_model(model_class, model_path)
    custom_model.eval()
    custom_loading_time = time.time() - start_time

    # Inference with custom model
    start_time = time.time()
    custom_output = custom_model(input_tensor)
    custom_inference_time = time.time() - start_time

    # ---------------------------
    # Standard Loading Method
    # ---------------------------
    # start_time = time.time()
    # # Load the model using the standard method
    # standard_model = model_class(weights=models.MobileNet_V3_Large_Weights.DEFAULT).to('cuda')
    # standard_model.eval()
    # standard_loading_time = time.time() - start_time

    # # Inference with standard model
    # start_time = time.time()
    # standard_output = standard_model(input_tensor)
    # standard_inference_time = time.time() - start_time

    # ---------------------------
    # Compare Outputs
    # ---------------------------
    # outputs_match = torch.allclose(custom_output, standard_output, atol=1e-6)

    # ---------------------------
    # Print Results
    # ---------------------------
    print("Custom Loading Method:")
    print(f"  Loading Time: {custom_loading_time:.4f} seconds")
    print(f"  Inference Time: {custom_inference_time:.4f} seconds")

    # print("\nStandard Loading Method:")
    # print(f"  Loading Time: {standard_loading_time:.4f} seconds")
    # print(f"  Inference Time: {standard_inference_time:.4f} seconds")

    # print(f"\nDo the outputs match? {'Yes' if outputs_match else 'No'}")


if __name__ == "__main__":
    main()
