import json
import os
from typing import List, Optional, Tuple, Union, Dict
import torch
import torchvision.models as models
import cuda_saver  # Import the C++ module
import sys
from io import StringIO
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.models import MobileNet_V3_Large_Weights

#model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).to(device)
model_name = "facebook/opt-6.7b"
local_cache = "./local_cache_opt_6.7b"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=local_cache  # This will store the downloaded files locally on disk
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",      # or {"": "cpu"}
    cache_dir=local_cache  # Ensures weights are cached on disk
)



import os
import json
import torch
from typing import Dict
import cuda_saver  # Assuming this is your custom module

def save_model(
    model: torch.nn.Module,
    model_path: str,
    gpu_percent: float,
    cpu_percent: float,
    save_config: bool = True
):
    """
    Saves a PyTorch model in a distributed fashion (GPU, CPU, or Disk), along with an optional config.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        model_path (str): Where to save the model and related metadata.
        gpu_percent (float): Percentage of total size to store on GPU.
        cpu_percent (float): Percentage of total size to store on CPU.
        save_config (bool): Whether to save the model configuration. Default is True.
    """

    # 1. Create the output directory
    os.makedirs(model_path, exist_ok=True)


    # 3. Get the state dict
    model_state_dict = model.state_dict()

    # 4. Calculate thresholds for GPU/CPU splitting
    total_size = sum(param.numel() * param.element_size() for param in model_state_dict.values())
    gpu_threshold = gpu_percent / 100 * total_size
    cpu_threshold = (gpu_percent + cpu_percent) / 100 * total_size

    accumulated_size = 0
    current_location = 'gpu'

    # 5. Prepare the index that maps each tensor’s name -> metadata
    tensor_index = {}

    # 6. Iterate through parameters and save them to the chosen location
    for name, param in model_state_dict.items():
        size = param.numel() * param.element_size()
        data_ptr = param.data_ptr()

        # Create a data structure describing this tensor
        tensor_data = (data_ptr, size)
        tensor_data_index = {name: tensor_data}

        # Decide where to save this tensor
        if current_location == 'gpu':
            cuda_saver.save_tensors_to_gpu([name], tensor_data_index)
        elif current_location == 'cpu':
            cuda_saver.save_tensors_to_cpu([name], tensor_data_index)
        else:
            cuda_saver.save_tensors_to_disk([name], tensor_data_index)

        # Update our indexing info
        tensor_index[name] = {
            'location': current_location,
            'size': size,
            'shape': list(param.shape),
            'stride': list(param.stride()),
            'dtype': str(param.dtype)
        }

        accumulated_size += size

        # If we have reached the GPU threshold, switch to CPU
        if accumulated_size >= gpu_threshold and current_location == 'gpu':
            current_location = 'cpu'
        # If we have also reached the CPU threshold, switch to disk
        elif accumulated_size >= cpu_threshold and current_location == 'cpu':
            current_location = 'disk'

    # 7. Save the index
    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f, indent=4)

    # 8. (Optional) Save the model config — Hugging Face style, or a custom config
    #    Only if `save_config=True` and the model has `config` attribute
    if save_config and hasattr(model, "config"):
        try:
            # If this is a standard HF model, we can do:
            model.config.save_pretrained(model_path)
            
            # If you also want to handle generation config:
            if hasattr(model, "can_generate") and model.can_generate():
                # Example from your original snippet:
                model.generation_config.save_pretrained(model_path)

        except AttributeError:
            # If the model doesn't have a standard config, you can handle that here:
            custom_config_path = os.path.join(model_path, "custom_config.json")
            with open(custom_config_path, "w") as f:
                # Dump custom attributes from model or model.config as needed
                json.dump({"custom": "config"}, f, indent=4)


    """
    no_split_modules = get_no_split_modules(model, model._no_split_modules)
    with open(os.path.join(model_path, "no_split_modules.json"), "w") as f:
        json.dump(no_split_modules, f)

    tied_no_split_modules = get_tied_no_split_modules(model, no_split_modules)
    with open(os.path.join(model_path, "tied_no_split_modules.json"), "w") as f:
        json.dump(tied_no_split_modules, f)
    """

    print(f"Model and metadata saved to {model_path}")

def main():
        #save_dict(model.state_dict(), "./", 40,40)
        save_model(model, "./", gpu_percent=40, cpu_percent=40)
        
if __name__ == "__main__":
    main()

