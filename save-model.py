import json
import os
from typing import List, Optional, Tuple, Union, Dict
import torch
import torchvision.models as models
import cuda_saver  # Import the C++ module
import sys
from io import StringIO
import socket
import struct
import ctypes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=True).to(device)  # Your PyTorch model


def connect_to_daemon():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 8080))
        return sock
    except socket.error as e:
        print(f"Failed to connect to the daemon: {e}")
        return None

def send_command_to_daemon(sock, command):
    try:
        sock.sendall(command.encode('utf-8'))
        response = sock.recv(1024).decode('utf-8')
        return response
    except socket.error as e:
        print(f"Failed to send command to daemon: {e}")
        return None

def save_dict(model_state_dict: Dict[str, torch.Tensor], model_path: str):
    tensor_names = list(model_state_dict.keys())
    tensor_data_index = {}
    for name, param in model_state_dict.items():
        if param.is_cuda:
            data_ptr = param.data_ptr()  # Get the pointer to the data on the GPU
            size = param.numel() * param.element_size()  # Calculate the size of the tensor in bytes
            tensor_data_index[name] = (data_ptr, size)
        else:
            raise ValueError(f"Tensor {name} is not on the GPU.")
        
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    # save tensors using the C++ function
    print("Starting save_tensors_cpp")

    tensor_offsets = cuda_saver.save_tensors_cpp(tensor_names, tensor_data_index)
    output = sys.stdout.getvalue()

    # Reset stdout to original
    sys.stdout = original_stdout

    print("Captured C++ output:")
    print(output)
    # create tensor index
    tensor_index = {}
    for name, param in model_state_dict.items():
        # name: offset, size
        tensor_index[name] = (tensor_offsets[name], tensor_data_index[name][1], tuple(param.shape), tuple(param.stride()), str(param.dtype))

    # save tensor index
    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)


def save_tensors_py(tensor_names, tensor_data_index):
    tensor_offsets = {}

    print("Starting save_tensors_py")

    sock = connect_to_daemon()
    if not sock:
        print("Failed to connect to the daemon")
        return tensor_offsets

    for name in tensor_names:
        print(f"Processing tensor: {name}")

        data_ptr = tensor_data_index[name][0]
        size = tensor_data_index[name][1]

        print(f"Data pointer: {data_ptr}, Size: {size}")

        # Send allocate command to the daemon
        command = f"ALLOCATE {name} {size}"
        response = send_command_to_daemon(sock, command)

        if response and response.startswith("ALLOCATED"):
            print(f"Received response: {response}")

            # Prepare the COPY command
            copy_command = f"COPY {name} {data_ptr} {size}"
            copy_response = send_command_to_daemon(sock, copy_command)

            if copy_response and copy_response.startswith("COPIED"):
                print(f"Copied data to GPU memory for tensor: {name}")

                # Assuming daemon returns the GPU pointer as part of the response
                gpu_pointer = int(copy_response.split()[-1])
                tensor_offsets[name] = gpu_pointer
            else:
                print(f"Copy failed for tensor: {name}")
                send_command_to_daemon(sock, f"FREE {name}")
        else:
            print(f"Memory allocation failed for tensor: {name}")

    sock.close()
    print("Finished save_tensors_py")
    return tensor_offsets

def main():

        save_dict(model.state_dict(), "./")



if __name__ == "__main__":
    main()

