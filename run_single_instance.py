#!/usr/bin/env python3
import argparse
import json
import gc
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from assemble_model_example import assemble_model

def benchmark_inference(model, model_path):
    prompts = ["Once upon a time,"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
    model.eval()
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_format", type=str, required=True, choices=["sllm", "safetensors"])
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--replica", type=int, default=0)
    args = parser.parse_args()

    result = {"model_name": args.model_name, "replica": args.replica, "model_format": args.model_format}

    if args.model_format == "sllm":
        model_path = os.path.join(args.model_dir, f"{args.model_name}_{args.replica}")
        start_time = time.time()
        # Use your assembly loader (for example, using 'hf' type for OPT)
        model = assemble_model('hf', args.model_name, args.model_dir)
        end_time = time.time()
    elif args.model_format == "safetensors":
        model_path = os.path.join(args.model_dir, f"{args.model_name}_safetensors_{args.replica}")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        end_time = time.time()
    result["loading_time_sec"] = end_time - start_time

    # Ensure the model is on GPU
    model = model.to("cuda")

    inf_time, throughput, output_text = benchmark_inference(model, model_path)
    result["inference_time_sec"] = inf_time
    result["throughput_tokens_per_sec"] = throughput
    result["output_text"] = output_text

    print(json.dumps(result))

if __name__ == "__main__":
    main()
