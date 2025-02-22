import json
import os
import psutil
import torch
import torchvision.models as models
import cuda_saver  # Import the C++ module
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.models import MobileNet_V3_Large_Weights

#model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).to(device)
model_name = "facebook/opt-1.3b"
local_cache = "./local_cache_opt_1.3b"

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
    Saves a PyTorch model in a distributed fashion (GPU, CPU, or Disk),
    along with an optional config if it's a Hugging Face model 
    (or if you'd like to save a custom config for a standard model).

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        model_path (str): Where to save the model and related metadata.
        gpu_percent (float): Percentage of total size to store on GPU.
        cpu_percent (float): Percentage of total size to store on CPU.
        save_config (bool): Whether to save the model configuration. Default is True.
                            - For Hugging Face models, this will save model.config and 
                              (optionally) generation_config.
                            - For standard PyTorch models, you can save a custom JSON if desired.
    """

    # 1. Create the output directory
    os.makedirs(model_path, exist_ok=True)

    # 2. Get the state dict
    model_state_dict = model.state_dict()

    # 3. Calculate thresholds for GPU/CPU splitting
    total_size = sum(param.numel() * param.element_size() for param in model_state_dict.values())
    gpu_threshold = gpu_percent / 100 * total_size
    cpu_threshold = (gpu_percent + cpu_percent) / 100 * total_size

    accumulated_size = 0
    current_location = 'gpu'

    # 4. Prepare the index that maps each tensor’s name -> metadata
    tensor_index = {}

    # 5. Iterate through parameters and save them to the chosen location
    for name, param in model_state_dict.items():
        size_bytes = param.numel() * param.element_size()
        data_ptr = param.data_ptr()

        # Create a data structure describing this tensor
        tensor_data = (data_ptr, size_bytes)
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
            'size_bytes': size_bytes,
            'shape': list(param.shape),
            'stride': list(param.stride()),
            'dtype': str(param.dtype),
        }

        accumulated_size += size_bytes

        # If we have reached the GPU threshold, switch to CPU
        if accumulated_size >= gpu_threshold and current_location == 'gpu':
            current_location = 'cpu'
        # If we have reached the CPU threshold, switch to disk
        elif accumulated_size >= cpu_threshold and current_location == 'cpu':
            current_location = 'disk'

    # 6. Save the index
    index_path = os.path.join(model_path, "tensor_index.json")
    with open(index_path, "w") as f:
        json.dump(tensor_index, f, indent=4)

    # 7. (Optional) Save the model config if `save_config == True`
    #    If it's a Hugging Face model, we can do model.config.save_pretrained.
    #    Otherwise, we can do a simple custom config file if needed.
    if save_config:
        # Check if this model looks like a Hugging Face model
        # Typically, HF models have a `.config` attribute.
        if hasattr(model, "config"):
            # Attempt standard Hugging Face saving:
            try:
                model.config.save_pretrained(model_path)

                # If you want to handle generation config (for LLMs, e.g. GPT, OPT, etc.)
                if hasattr(model, "can_generate") and model.can_generate():
                    model.generation_config.save_pretrained(model_path)

            except AttributeError:
                # This might happen if `.config` is not a standard HF config object
                # In that case, save a custom config
                custom_config_path = os.path.join(model_path, "custom_config.json")
                with open(custom_config_path, "w") as f:
                    # Write out some relevant data from `model.config` or a placeholder
                    json.dump({"custom_config": "data"}, f, indent=4)
        else:
            # It's not a HF model, so we can optionally do a custom JSON
            custom_config_path = os.path.join(model_path, "custom_config.json")
            with open(custom_config_path, "w") as f:
                # Example: store the class name or something else you find relevant
                # For standard PyTorch models like torchvision, there's typically no config
                json.dump({"model_type": type(model).__name__}, f, indent=4)

    print(f"Model and metadata saved to {model_path}")

def get_model_size_bytes(model: torch.nn.Module):
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes


def inital_allocation(model: torch.nn.Module):
    """
    Attempts to allocate 40% of the model on GPU and 40% on CPU.
    If there's not enough GPU or CPU memory, it reduces until it fits 
    or sets them to 0 if needed.
    Returns (gpu_percent, cpu_percent).
    """
    total_bytes = get_model_size_bytes(model) 

    # Desired
    desired_gpu_pct = 40.0
    desired_cpu_pct = 40.0

    # Get free GPU memory
    if torch.cuda.is_available():
        free_gpu_bytes, _ = torch.cuda.mem_get_info(device=0)
    else:
        free_gpu_bytes = 0


    free_cpu_bytes = psutil.virtual_memory().available

    gpu_percent = desired_gpu_pct
    cpu_percent = desired_cpu_pct
    
    while True:
        gpu_needed_bytes = total_bytes * (gpu_percent / 100.0)
        cpu_needed_bytes = total_bytes * (cpu_percent / 100.0)

        if gpu_needed_bytes > free_gpu_bytes:
            gpu_percent -= 5
            if gpu_percent < 0:
                gpu_percent = 0
                break
            continue
        
        if cpu_needed_bytes > free_cpu_bytes:
            cpu_percent -= 5
            if cpu_percent < 0:
                cpu_percent = 0
                break
            continue

        break

    if (gpu_percent + cpu_percent) > 100:
        leftover = 100 - gpu_percent
        if leftover < 0:
            gpu_percent = 100
            cpu_percent = 0
        else:
            cpu_percent = min(cpu_percent, leftover)
    with open('model_usage.json', 'r') as f:
        usage_data = json.load(f)
        
    usage_data[model.__class__.__name__] = {
        "count": 0,
        "gpu_percent": gpu_percent,
        "cpu_percent": cpu_percent
    }
    
    with open('model_usage.json', 'w') as f:
        json.dump(usage_data, f, indent=4)
    return gpu_percent, cpu_percent


   
def main():
        #save_dict(model.state_dict(), "./", 40,40)
        gpu_percent, cpu_percent = inital_allocation(model)
        print(f"Chosen: GPU={gpu_percent}%, CPU={cpu_percent}%")

        #save_model(model, "./", gpu_percent=40, cpu_percent=40)
        
if __name__ == "__main__":
    main()

