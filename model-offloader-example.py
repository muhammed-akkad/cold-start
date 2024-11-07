import json
import os
from typing import List, Optional, Tuple, Union, Dict
import torch
import torchvision.models as models
import cuda_saver  # Import the C++ module
import sys
from io import StringIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.models import MobileNet_V3_Large_Weights

model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).to(device)




def save_dict(model_state_dict: Dict[str, torch.Tensor], model_path: str, gpu_percent: float, cpu_percent: float):
    """
    Saves the state dictionary of a model to a specified path, dividing parameters into GPU, CPU, and Disk groups based on cumulative size.

    Args:
        model_state_dict (Dict[str, torch.Tensor]): The state dictionary of the model.
        model_path (str): The path to save the tensors and index.
        gpu_percent (float): Percentage of total size to assign to GPU memory.
        cpu_percent (float): Percentage of total size to assign to CPU memory.
    """
    import os
    import json
    import cuda_saver

    # Calculate the total size of all tensors
    total_size = sum(param.numel() * param.element_size() for param in model_state_dict.values())

    # Calculate size thresholds
    gpu_threshold = gpu_percent / 100 * total_size
    cpu_threshold = (gpu_percent + cpu_percent) / 100 * total_size

    accumulated_size = 0
    current_location = 'gpu'

    os.makedirs(model_path, exist_ok=True)

    tensor_index = {}

    for name, param in model_state_dict.items():
        size = param.numel() * param.element_size()
        data_ptr = param.data_ptr()

        # Prepare tensor data
        tensor_data = (data_ptr, size)
        tensor_names = [name]  # Single tensor name
        tensor_data_index = {name: tensor_data}

        # Process the tensor based on the current location
        if current_location == 'gpu':
            # Save tensor to GPU
            cuda_saver.save_tensors_to_gpu(tensor_names, tensor_data_index)
        elif current_location == 'cpu':
            # Save tensor to CPU
            cuda_saver.save_tensors_to_cpu(tensor_names, tensor_data_index)
        else:
            # Save tensor to Disk
            cuda_saver.save_tensors_to_disk(tensor_names, tensor_data_index)

        # Collect tensor metadata for the index
        tensor_index[name] = {
            'location': current_location,
            'size': size,
            'shape': list(param.shape),
            'stride': list(param.stride()),
            'dtype': str(param.dtype)
        }

        accumulated_size += size

        # Check if we need to switch to the next memory location
        if accumulated_size >= gpu_threshold and current_location == 'gpu':
            current_location = 'cpu'
        elif accumulated_size >= cpu_threshold and current_location == 'cpu':
            current_location = 'disk'

    # Save tensor index
    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f, indent=4)

def main():
        save_dict(model.state_dict(), "./", 40,40)
        
if __name__ == "__main__":
    main()

