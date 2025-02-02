import errno
import mmap
import os
import json
import time
import torch
import ctypes
import cuda_saver
import posix_ipc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import mmap
import posix_ipc
import torch
import cuda_saver
import json
import concurrent.futures

def load_single_tensor(name, meta, model_path):
    """
    Loads one tensor (GPU/CPU/Disk) based on metadata.
    
    Args:
        name (str): Tensor name.
        meta (dict): Metadata with keys 'location', 'size', 'shape', 'dtype'.
        model_path (str): Base directory containing handlers_gpu/, tensors_data/, etc.
    
    Returns:
        (str, torch.Tensor): The tensor name and the loaded tensor.
    """
    location = meta["location"]
    size = meta["size"]
    shape = meta["shape"]
    dtype_str = meta["dtype"]
    
    # Convert "torch.float32" -> "float32"
    dtype = getattr(torch, dtype_str.split('.')[1])  
    
    if location == 'gpu':
        # GPU: Load via IPC handle
        ipc_file = os.path.join(model_path, 'handlers_gpu', f"{name}_ipc_handle.bin")
        tensor = cuda_saver.load_model_tensor(ipc_file, shape, dtype)
    
    elif location == 'cpu':
        # CPU: Load via POSIX shared memory
        import errno
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
    
    elif location == 'disk':
        # Disk: Load tensor from .bin file
        filename = os.path.join(model_path, 'tensors_data', f"{name}_data.bin")
        with open(filename, 'rb') as f:
            mm = mmap.mmap(f.fileno(), length=size, access=mmap.ACCESS_READ)
        tensor = torch.frombuffer(mm, dtype=dtype).reshape(shape)
    
    else:
        raise ValueError(f"Unknown location '{location}' for tensor '{name}'")

    return name, tensor
def load_model_parallel_profiled(
    model_class, 
    model_path, 
    hf_model_name="facebook/opt-1.3b", 
    num_workers=4
):
    """
    Loads a model's state dictionary (split across GPU, CPU, and disk) in parallel and
    returns the assembled Hugging Face model, while profiling the time spent in each step.

    Args:
        model_class: The Hugging Face model class to instantiate (e.g., AutoModelForCausalLM).
        model_path (str): Directory where the model parts (tensor_index.json, handlers_gpu/, tensors_data/) are saved.
        hf_model_name (str): Name of the Hugging Face model to load config from (default: facebook/opt-1.3b).
        num_workers (int): Number of threads to use for parallel loading.
        
    Returns:
        model (nn.Module): The model with weights loaded from GPU, CPU, and Disk.
    """

    # A dictionary to store timing for various steps
    profile_times = {
        "read_index": 0.0,
        "prepare_items": 0.0,
        "thread_pool_submit": 0.0,
        "thread_pool_wait": 0.0,
        "load_config": 0.0,
        "instantiate_model": 0.0,
        "load_state_dict": 0.0,
        "move_to_gpu": 0.0,
    }

    # 1) Load the tensor index (dict with tensor metadata)
    t0 = time.time()
    with open(os.path.join(model_path, "tensor_index.json"), 'r') as f:
        tensor_index = json.load(f)
    profile_times["read_index"] += time.time() - t0

    # 2) Prepare items (list of (name, meta)) to load in parallel
    t0 = time.time()
    items_to_load = list(tensor_index.items())
    profile_times["prepare_items"] += time.time() - t0

    # 3) Use a thread pool to load tensors concurrently
    #    We'll split the timing into "submit tasks" vs. "wait for tasks"
    t_submit_start = time.time()
    state_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(load_single_tensor, name, meta, model_path)
            for name, meta in items_to_load
        ]
    profile_times["thread_pool_submit"] += time.time() - t_submit_start

    # 4) Collect results as they complete
    t_wait_start = time.time()
    for future in concurrent.futures.as_completed(futures):
        name, tensor = future.result()  # result() will raise exception if any
        state_dict[name] = tensor
    profile_times["thread_pool_wait"] += time.time() - t_wait_start

    # 5) Load the model config from Hugging Face
    t0 = time.time()
    config = AutoConfig.from_pretrained(hf_model_name)
    profile_times["load_config"] += time.time() - t0

    # 6) Instantiate the model architecture with that config
    t0 = time.time()
    model = model_class.from_config(config)
    profile_times["instantiate_model"] += time.time() - t0

    # 7) Load the assembled state_dict into the model
    t0 = time.time()
    model.load_state_dict(state_dict, strict=True)
    profile_times["load_state_dict"] += time.time() - t0

    # 8) Move model to GPU
    t0 = time.time()
    model.to("cuda")
    profile_times["move_to_gpu"] += time.time() - t0

    # Print a profiling summary
    print("\n--- Profiling Summary (Parallel Load) ---")
    for k, v in profile_times.items():
        print(f"{k}: {v:.4f} seconds")

    return model


def load_model_no_dict(model_class, model_path, hf_model_name="facebook/opt-1.3b"):
    """
    Loads each parameter directly into the model without building an intermediate state_dict.
    """

    # 1) Load the model config
    config = AutoConfig.from_pretrained(hf_model_name)

    # 2) Instantiate an empty model with that config
    model = model_class.from_config(config)
    model.eval()  # or stay in train mode if needed

    # 3) Read the tensor index (which has name -> metadata)
    with open(os.path.join(model_path, "tensor_index.json"), "r") as f:
        tensor_index = json.load(f)

    # 4) Loop over each parameter name in the model
    #    (We assume param_name in model matches param_name in your tensor_index)
    #    If you have exact name matches, this is straightforward. If not, you must map names carefully.
    for param_name, param_tensor in model.named_parameters():
        if param_name not in tensor_index:
            # If something's missing from your index, handle or skip
            # E.g., param might be newly added or might need special init
            print(f"Warning: {param_name} not found in tensor_index. Skipping.")
            continue

        meta = tensor_index[param_name]
        location = meta["location"]
        size = meta["size"]
        shape = meta["shape"]
        dtype_str = meta["dtype"]
        dtype = getattr(torch, dtype_str.split('.')[1])  # "torch.float32" -> "float32"

        # 5) Load the tensor from GPU/CPU/disk
        if location == "gpu":
            ipc_file = os.path.join(model_path, 'handlers_gpu', f"{param_name}_ipc_handle.bin")
            loaded_tensor = cuda_saver.load_model_tensor(ipc_file, shape, dtype)

        elif location == "cpu":
            shm_name = f"/{param_name}"
            try:
                shm = posix_ipc.SharedMemory(shm_name, flags=posix_ipc.O_RDONLY)
            except posix_ipc.ExistentialError:
                raise FileNotFoundError(f"Shared memory {shm_name} not found.")
            
            fd = shm.fd
            h_memory = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
            shm.close_fd()

            buffer = memoryview(h_memory)
            loaded_tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)

        elif location == "disk":
            filename = os.path.join(model_path, 'tensors_data', f"{param_name}_data.bin")
            with open(filename, 'rb') as f:
                buffer = f.read()
            loaded_tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)

        else:
            raise ValueError(f"Unknown location '{location}' for tensor '{param_name}'")

        # 6) Copy the loaded data directly into the model's parameter
        #    .data.copy_(...) ensures the memory from loaded_tensor goes into param_tensor in-place.
        if loaded_tensor.shape != param_tensor.data.shape:
            print(f"Shape mismatch for {param_name}: got {loaded_tensor.shape}, "
                  f"expected {param_tensor.shape}")
            continue

        param_tensor.data.copy_(loaded_tensor)

    # 7) Move entire model to GPU (if desired)
    model.to("cuda")

    return model

def main():
    model_path = "./"  # Directory where your model shards (GPU, CPU, Disk) are saved
    hf_model_name = "facebook/opt-1.3b"  # or "facebook/opt-350m", etc.

    # Prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    input_text = "The meaning of life is"
    input_tokens = tokenizer(input_text, return_tensors="pt").to("cuda")


    # ---------------------------
    # Standard Loading Method
    # ---------------------------
    start_time = time.time()
    standard_model = AutoModelForCausalLM.from_pretrained(hf_model_name).to("cuda")
    standard_model.eval()
    standard_loading_time = time.time() - start_time

    # Inference with standard model
    start_time = time.time()
    standard_output_ids = standard_model.generate(
        input_tokens["input_ids"],
        max_length=50,
        do_sample=True,
        temperature=0.7
    )
    standard_inference_time = time.time() - start_time

    standard_generated_text = tokenizer.decode(standard_output_ids[0], skip_special_tokens=True)

    # Print results for standard loading
    print("\nStandard Loading Method:")
    print(f"  Loading Time: {standard_loading_time:.4f} seconds")
    print(f"  Inference Time: {standard_inference_time:.4f} seconds")
    print(f"  Generated Text: {standard_generated_text}")


    
    # ---------------------------
    # Custom Loading Method
    # ---------------------------
    start_time = time.time()
    custom_model = load_model_parallel_profiled(AutoModelForCausalLM, model_path, hf_model_name=hf_model_name)
    custom_model.eval()
    custom_loading_time = time.time() - start_time

    # Inference with custom model
    start_time = time.time()
    custom_output_ids = custom_model.generate(
        input_tokens["input_ids"],  
        max_length=50, 
        do_sample=True, 
        temperature=0.7
    )
    custom_inference_time = time.time() - start_time

    custom_generated_text = tokenizer.decode(custom_output_ids[0], skip_special_tokens=True)

    # Print results for custom loading
    print("Custom Loading Method:")
    print(f"  Loading Time: {custom_loading_time:.4f} seconds")
    print(f"  Inference Time: {custom_inference_time:.4f} seconds")
    print(f"  Generated Text: {custom_generated_text}")

    # (Optional) Compare outputs
    outputs_match = (custom_generated_text == standard_generated_text)
    print(f"\nDo the outputs match? {'Yes' if outputs_match else 'No'}")
if __name__ == "__main__":
    main()
