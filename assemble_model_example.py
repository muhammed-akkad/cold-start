import os
import time
import json
import mmap
import posix_ipc
import torch
import concurrent.futures
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict
from accelerate import init_empty_weights, dispatch_model
from collections import defaultdict
import torchvision.models as tv_models
import cuda_saver  # Your custom module for handling tensor loading
from PIL import Image
import torchvision.transforms as T
USAGE_COUNTS_FILE = "model_usage.json"

model_usage_counts = defaultdict(int)
def _load_tensor_from_index_entry(model_path: str, name: str, meta: Dict) -> (str, torch.Tensor):
    """
    Helper function to load a single parameter based on its metadata.
    Returns a tuple of (parameter_name, tensor).
    """
    location = meta['location']
    size = meta['size_bytes']
    shape = meta['shape']
    dtype_str = meta['dtype']
    dtype = getattr(torch, dtype_str.split('.')[-1])  # e.g., "torch.float32" -> torch.float32

    t_load_start = time.time()

    if location == 'gpu':
        # GPU: Load directly onto GPU
        ipc_file = os.path.join(model_path, 'handlers_gpu', f"{name}_ipc_handle.bin")
        tensor = cuda_saver.load_model_tensor(ipc_file, shape, dtype)

    
    elif location == 'cpu':
        # CPU: Load from shared memory
        shm_name = f"/{name}"
        try:
            shm = posix_ipc.SharedMemory(shm_name, flags=posix_ipc.O_RDONLY)
        except posix_ipc.ExistentialError:
            raise FileNotFoundError(f"Shared memory {shm_name} not found.")

        fd = shm.fd
        h_memory = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        shm.close_fd()

        buffer = memoryview(h_memory)
        tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
        tensor = tensor
    
    elif location == 'disk':
        # Disk: Load tensor from .bin file (assigned to CPU for now)
        filename = os.path.join(model_path, 'tensors_data', f"{name}_data.bin")
        with open(filename, 'rb') as f:
            buff = f.read()
        tensor = torch.frombuffer(buff, dtype=dtype).reshape(shape)
        tensor = tensor
    
    else:
        raise ValueError(f"Unknown location '{location}' for tensor '{name}'")

    duration = time.time() - t_load_start
    #print(f"Loaded {name} from {location} in {duration:.3f} seconds.")
    return name, tensor


def _assign_param(name, state_dict, module_dict):
    """
    Helper function to assign one parameter from state_dict to the correct module.
    Returns a message so we can track completions (optional).
    """
    if name in state_dict:
        # Create the nn.Parameter
        tensor = state_dict[name]
        new_param = torch.nn.Parameter(tensor)

        # Split out module path vs. param name
        if '.' in name:
            module_path, param_name = name.rsplit('.', 1)
        else:
            module_path, param_name = '', name
        
        # Look up the module
        module = module_dict.get(module_path)
        if module is not None:
            setattr(module, param_name, new_param)
            return f"[ASSIGNED] {name} -> {new_param.device}"
        else:
            return f"[WARNING] Module {module_path} not found for param {name}"
    else:
        return f"[MISSING] {name} not in state_dict"

def _instantiate_model(model_class, config):
    # A wrapper so we can run it in a future
    with init_empty_weights():
        model = model_class.from_config(config)
    return model

def build_device_map_from_json(tensor_index):
    """
    tensor_index is a dict like:
      {
        "decoder.layer.0.weight": { "location": "gpu", ... },
        "decoder.layer.0.bias":   { "location": "gpu", ... },
        "decoder.layer.1.weight": { "location": "cpu", ... },
        ...
      }
    Returns a dictionary suitable for dispatch_model(...), e.g.:
      {
        "decoder.layer.0": "cuda",
        "decoder.layer.1": "cpu",
        ...
      }
    """
    device_map = {}

    for full_param_name, meta in tensor_index.items():
        # Extract the module name by dropping the last part (weight, bias, etc.)
        if '.' in full_param_name:
            module_path, _ = full_param_name.rsplit('.', 1)
        else:
            # If there's no dot, param name == module_path
            module_path = full_param_name

        # Convert "gpu" -> "cuda", "cpu" -> "cpu"
        loc = meta["location"]
        if loc == "gpu":
            device_name = "cuda"
        elif loc == "cpu":
            device_name = "cpu"
        elif loc == "disk":
            # Usually means we'll load to CPU then maybe move later
            device_name = "cpu"
        else:
            raise ValueError(f"Unknown location: {loc}")

        # If we already saw this module, check consistency 
        # (All parameters of the same module should be on the same device)
        if module_path in device_map:
            existing_dev = device_map[module_path]
            if existing_dev != device_name:
                print(f"[WARNING] module {module_path} has conflicting devices: "
                      f"{existing_dev} vs {device_name}")
            # Possibly override or keep the first one. For simplicity, keep the first.
        else:
            device_map[module_path] = device_name

    return device_map

def load_model_parallel(
    model_class,
    model_path: str,
    hf_model_name: str = "facebook/opt-1.3b"
):
    """
    Loads a model with parallel reading of tensor_index.json, parallel loading of 
    parameter data from disk/memory, and optional parallel assignment of parameters.
    Also captures profiling/timing information for each step.
    """
    # ------------------
    # Initialize Profiling
    # ------------------
    profile_times = {
        "read_index": 0.0,
        "load_config": 0.0,
        "assign_tensors": 0.0,
    }

    # ------------------
    # Step 1: Read tensor_index.json
    # ------------------
    t0 = time.time()
    tensor_index_path = os.path.join(model_path, "tensor_index.json")
    with open(tensor_index_path, 'r') as f:
        tensor_index = json.load(f)
    profile_times["read_index"] += time.time() - t0

    # ------------------
    # Step 2: Load HF config
    # ------------------
    t0 = time.time()
    config = AutoConfig.from_pretrained(hf_model_name)
    profile_times["load_config"] += time.time() - t0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Task A: instantiate model
        model_future = executor.submit(_instantiate_model, model_class, config)
        
        # Task B: load all parameters (in parallel)
        #  We'll create a separate ThreadPool for the parameter-level parallelism
        def _load_all_tensors():
            st_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = []
                for name, meta in tensor_index.items():
                    futures.append(pool.submit(_load_tensor_from_index_entry, model_path, name, meta))
                for fut in concurrent.futures.as_completed(futures):
                    param_name, tensor = fut.result()
                    st_dict[param_name] = tensor
            return st_dict

        state_dict_future = executor.submit(_load_all_tensors)

        # Wait for both tasks
        model = model_future.result()
        state_dict = state_dict_future.result()
    profile_times["model_instantiation"] = time.time() - t0  # includes both tasks done concurrently
    


    # ------------------
    # Step 5: Parallel assignment of loaded tensors
    # ------------------
    t0 = time.time()

    # Build a lookup once
    module_dict = dict(model.named_modules())

    named_params = list(model.named_parameters())  # so we can parallelize
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        assign_futures = []
        for (name, param) in named_params:
            fut = executor.submit(_assign_param, name, state_dict, module_dict)
            assign_futures.append(fut)

        for fut in concurrent.futures.as_completed(assign_futures):
            msg = fut.result()
            results.append(msg)

    # (Optional) Print assignment messages if you want debug info
    # for msg in results:
    #    print(msg)

    profile_times["assign_tensors"] += time.time() - t0

    device_map = build_device_map_from_json(tensor_index)

    model = dispatch_model(model, device_map=device_map)
    # ------------------
    # Print a profiling summary
    # ------------------
    """     print("\n--- Load Model Parallel Profiling Summary ---")
    for step, secs in profile_times.items():
        print(f"{step}: {secs:.4f} seconds") """

    return model

def pre_load_model(model_class,
    model_path: str,
    hf_model_name: str = "facebook/opt-1.3b"):
    load_usage_counts()
    model_name = hf_model_name
    increment_usage(model_name)
    model = load_model_parallel(model_class, model_path, hf_model_name)
    save_usage_counts()
    return model


def load_usage_counts():
    if os.path.exists(USAGE_COUNTS_FILE):
        with open(USAGE_COUNTS_FILE, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            model_usage_counts[k] = v

def save_usage_counts():
    with open(USAGE_COUNTS_FILE, 'w') as f:
        json.dump(dict(model_usage_counts), f)

def increment_usage(model_name: str):
    model_usage_counts[model_name] += 1
    
def _assign_param_or_buffer(full_name, loaded_tensors, module_dict):
    """
    full_name: e.g. "features.0.1.running_mean"
    loaded_tensors: a dict of {param_name: torch.Tensor}
    module_dict: e.g. dict(model.named_modules()) so we can find the correct submodule

    We check whether it's a parameter or a buffer in the submodule and assign accordingly.
    """
    if full_name not in loaded_tensors:
        return f"[MISSING] {full_name}"

    tensor = loaded_tensors[full_name]
    #print(full_name)
    # Separate the submodule path vs. the attribute name (e.g., "running_mean")
    if '.' in full_name:
        module_path, attrib_name = full_name.rsplit('.', 1)
    else:
        module_path, attrib_name = '', full_name

    module = module_dict.get(module_path)
    if module is None:
        return f"[WARNING] Submodule not found for {full_name}"

    # Check if this attribute is a parameter or a buffer in the submodule
    is_param = attrib_name in module._parameters
    is_buffer = attrib_name in module._buffers
    
    if is_param:
        # It's a learnable parameter
        param = torch.nn.Parameter(tensor)
        setattr(module, attrib_name, param)
        return f"[ASSIGNED PARAM] {full_name}"
    
    elif is_buffer:
        # It's a buffer (e.g. BN running stats)
        # You can either assign directly or use register_buffer:
        module.register_buffer(attrib_name, tensor)
        #print(f"[ASSIGNED BUFFER] {full_name} to {module_path}.{attrib_name} = {tensor[:4]}")
        return f"[ASSIGNED BUFFER] {full_name}"

    else:
        # It's neither in _parameters nor _buffers.
        # Possibly a leftover, or a mismatch in naming. Log or skip it.
        return f"[WARNING] {full_name} not found as param or buffer"
def instantiate_hf_model(model_name):
    """
    For a Hugging Face model, we do:
       config = AutoConfig.from_pretrained(model_name)
       with init_empty_weights():
           model = AutoModelForCausalLM.from_config(config)
    or adapt if you use a different HF model class.
    """
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    return model

def instantiate_torchvision_model(model_name):
    """
    For a TorchVision model, we skip all config objects and directly instantiate inside init_empty_weights.
    For example, if model_name == 'mobilenet_v3_large'.
    """
    with init_empty_weights():
        if model_name == "mobilenet_v3_large":
            model = tv_models.mobilenet_v3_large()
        elif model_name == "resnet50":
            model = tv_models.resnet50()
        elif model_name == "mobilenet_v2":
            model = tv_models.mobilenet_v2()
        elif model_name == "vgg19":
            model = tv_models.vgg19()
        else:
            raise ValueError(f"Unsupported TorchVision model: {model_name}")
    return model


# -------------------------------
# Single universal assemble function
# -------------------------------
def assemble_model(
    model_type: str,    # 'torchvision' or 'hf'
    model_name: str,    # e.g. 'mobilenet_v3_large' or 'facebook/opt-1.3b'
    model_path: str     # path with tensor_index.json and shards
):
    """
    Reads `tensor_index.json` from `model_path`,
    Instantiates an empty model according to `model_type`,
    Loads and assigns parameters in parallel,
    Dispatches submodules to devices according to the index.

    Returns:
      (model, timings): A 2-tuple where
        - model is the fully loaded PyTorch model
        - timings is a dict of time measurements, including total time
    """

    overall_start = time.time()

    # ----------------------------------------------------------------------
    # 1) Load the tensor_index.json
    # ----------------------------------------------------------------------
    t0 = time.time()
    tensor_index_file = os.path.join(model_path, "tensor_index.json")
    with open(tensor_index_file, "r") as f:
        tensor_index = json.load(f)
    t_index = time.time() - t0

    # ----------------------------------------------------------------------
    # 2) Instantiate the empty model
    # ----------------------------------------------------------------------
    t0 = time.time()
    if model_type == "hf":
        model = instantiate_hf_model(model_name)
    elif model_type == "torchvision":
        model = instantiate_torchvision_model(model_name)
    else:
        raise ValueError("model_type must be 'hf' or 'torchvision'")
    t_instantiate = time.time() - t0

    # ----------------------------------------------------------------------
    # 3) Parallel-load all parameters from shards
    # ----------------------------------------------------------------------
    t0 = time.time()
    st_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for param_name, meta in tensor_index.items():
            futures.append(
                executor.submit(_load_tensor_from_index_entry, model_path, param_name, meta)
            )
        for fut in concurrent.futures.as_completed(futures):
            name, tensor = fut.result()
            st_dict[name] = tensor
    t_load = time.time() - t0

    # ----------------------------------------------------------------------
    # 4) Parallel-assign the loaded tensors to the model's parameters
    # ----------------------------------------------------------------------
    t0 = time.time()
    module_dict = dict(model.named_modules())

    all_keys = list(st_dict.keys())
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        assign_futures = []
        for name in all_keys:
            assign_futures.append(
                executor.submit(_assign_param_or_buffer, name, st_dict, module_dict)
            )
        for fut in concurrent.futures.as_completed(assign_futures):
            _ = fut.result()
    t_assign = time.time() - t0

    # ----------------------------------------------------------------------
    # 5) Dispatch model to devices (example: everything to "cuda")
    # ----------------------------------------------------------------------
    t0 = time.time()
    device_map = {"": "cuda"}
    model = dispatch_model(model, device_map=device_map)
    t_dispatch = time.time() - t0

    # ----------------------------------------------------------------------
    # Measure total time
    # ----------------------------------------------------------------------
    total_time = time.time() - overall_start

    # ----------------------------------------------------------------------
    # Build timings dict
    # ----------------------------------------------------------------------
    timings = {
        "read_index_sec":          t_index,
        "instantiate_model_sec":   t_instantiate,
        "load_params_sec":         t_load,
        "assign_params_sec":       t_assign,
        "dispatch_model_sec":      t_dispatch,
        "total_sec":               total_time
    }

    # Optional summary printout
    print(f"\n--- {model_name} Assembly Summary ---")
    print(f"Read index:        {t_index:.2f} s")
    print(f"Instantiate model: {t_instantiate:.2f} s")
    print(f"Load parameters:   {t_load:.2f} s")
    print(f"Assign parameters: {t_assign:.2f} s")
    print(f"Dispatch model:    {t_dispatch:.2f} s")
    print(f"Total:             {total_time:.2f} s")

    return model, timings

def test_mobilenet(loaded_model, image_path):
    """
    Test a loaded MobileNet V3 by running inference on a sample image.
    Prints out the top-5 classes with scores.
    """
    # 1) Put the model in eval mode, just to be safe
    loaded_model.eval()

    # 2) Load the image with PIL
    img = Image.open(image_path).convert("RGB")

    # 3) Apply the MobileNet V3 transforms
    #    Typically, for ImageNet-trained MobileNet, you'd do:
    transform = T.Compose([
        T.Resize(256),             # resize shorter side to 256
        T.CenterCrop(224),         # center crop to 224x224
        T.ToTensor(),              # turn into a [C,H,W] float tensor in [0..1]
        T.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet norms
                    std=[0.229, 0.224, 0.225]),
    ])
    tensor_input = transform(img).unsqueeze(0)  # shape (1, 3, 224, 224)

    # 4) Move input to the same device as the model
    #    (If you used accelerate dispatch, check where the first param is)
    device = next(loaded_model.parameters()).device
    tensor_input = tensor_input.to(device)

    # 5) Forward pass
    with torch.no_grad():
        output = loaded_model(tensor_input)  # shape: (1, 1000) for ImageNet
    
    # 6) Get class probabilities
    probs = torch.nn.functional.softmax(output[0], dim=0)

    # 7) Top-5 labels
    top5 = torch.topk(probs, k=5)
    top5_probs = top5.values.cpu().numpy()
    top5_indices = top5.indices.cpu().numpy()

    # 8) If you want to map indices -> class names for ImageNet:
    #    Download the ImageNet labels file from somewhere or define your own
    #    For a quick hack, you can do something like:
    IMAGENET_LABELS = {0: "tench, Tinca tinca", 1: "goldfish, Carassius auratus", 771: "safe"}
    # (Truncated for brevity—there are 1000 labels in total.)

    print("Top-5 Predictions:")
    for i in range(5):
        idx = top5_indices[i]
        label_str = IMAGENET_LABELS.get(idx, f"Class {idx}")
        score = top5_probs[i]
        print(f"  {label_str} ({idx}): {score:.4f}")

def main():
    """     model_path = "./"  # Directory where your model shards (GPU, CPU, Disk) are saved
        hf_model_name = "facebook/opt-1.3b"  # or "facebook/opt-350m", etc.

        # 1) Load the tokenizer from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        # 2) Prepare text input
        input_text = "The meaning of life is"
        input_tokens = tokenizer(input_text, return_tensors="pt").to("cuda")

        # 3) Load the model using our custom loader
        start_time = time.time()
        custom_model = pre_load_model(AutoModelForCausalLM, model_path, hf_model_name=hf_model_name)
        custom_model.eval()
        custom_loading_time = time.time() - start_time

        # 4) Run inference (generate text)
        start_time = time.time()
        generated_ids = custom_model.generate(
            input_tokens["input_ids"],  # tokenized input
            max_length=50, 
            do_sample=True, 
            temperature=0.7
        )
        custom_inference_time = time.time() - start_time

        # 5) Decode the output tokens into text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 6) Print the results
        print("Custom Loading Method:")
        print(f"  Loading Time: {custom_loading_time:.4f} seconds")
        print(f"  Inference Time: {custom_inference_time:.4f} seconds")
        print(f"  Generated Text: {generated_text}")
    """
    start_time = time.time()
    model = assemble_model('hf', 'facebook/opt-1.3b', './')
    end_time = time.time()
    print( end_time - start_time)
    
    model.eval()
    # test_mobilenet(model, "images.webp")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fresh = tv_models.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT).to(device)
    # test_mobilenet(fresh, "images.webp")

if __name__ == "__main__":
    main()