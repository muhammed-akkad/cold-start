import gc
import os
import time
import sys
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from assemble_model_example import assemble_model

def _warmup_cuda():
    num_gpus = torch.cuda.device_count()
    print(f"Warming up {num_gpus} GPUs")
    for i in tqdm(range(num_gpus)):
        torch.ones(1).to(f"cuda:{i}")
        torch.cuda.synchronize()

def _warmup_inference():
    print("Warming up inference")
    model_name = "facebook/opt-6.7b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    model = model.to("cuda")
    prompts = ["The quick brown fox jumps over the lazy dog."]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)
    del _, tokenizer, inputs, model
    gc.collect()
    torch.cuda.empty_cache()

def benchmark_inference(model: nn.Module, model_path: str):
    prompts = ["The quick brown fox jumps over the lazy dog."]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=50)
        end_time = time.time()
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference_time = end_time - start_time
    throughput = outputs.shape[1] / inference_time
    del outputs, tokenizer, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return inference_time, throughput, output_text

def measure_single(model_name: str, model_format: str, model_dir: str, replica: int):
    print(f"Loading {model_name}_{replica}")
    model_record = {"model_name": f"{model_name}_{replica}"}
    # Use a unique model_path per replica
    if model_format == "iceCrusher":
        model_path = model_dir
        start_time = time.time()
        model, timings = assemble_model('hf', 'facebook/opt-6.7b', './')
        model_record['profiling'] = timings
        end_time = time.time()
    elif model_format == "safetensors":
        model_path = os.path.join(model_dir, f"{model_name}_safetensors")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        end_time = time.time()
    else:
        raise ValueError("Unknown model format. Use 'iceCrusher' or 'safetensors'.")

    model_record["loading_time_sec"] = end_time - start_time

    # Inference benchmark
    inf_time, throughput, output_text = benchmark_inference(model, model_name)
    model_record["inference_time_sec"] = inf_time
    model_record["throughput"] = throughput
    model_record["output_text"] = output_text

    print("Benchmark Result:", model_record)
    # Clean up and exit this process
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return model_record

