import json
import torch
import os
import cuda_saver 

def reassemble_model_from_gpu(json_file_path):
    with open(json_file_path, "r") as f:
        tensor_index = json.load(f)

    model = torch.nn.Module()  # Create an empty model object to hold the layers
    
    for name, (ptr, size, shape, stride, dtype) in tensor_index.items():
        # Reconstruct the tensor using the C++ extension
        tensor = cuda_saver.load_tensor_from_gpu(ptr, shape, stride, dtype)

        # Assign the tensor to the model's state dict directly
        components = name.split('.')
        current_module = model
        for comp in components[:-1]:
            if not hasattr(current_module, comp):
                setattr(current_module, comp, torch.nn.Module())
            current_module = getattr(current_module, comp)
        setattr(current_module, components[-1], torch.nn.Parameter(tensor))

    return model

# Example usage
model_path = "./"  # Path where the JSON file is stored
model = reassemble_model_from_gpu(os.path.join(model_path, "tensor_index.json"))

print("Model successfully reassembled from GPU memory.")
