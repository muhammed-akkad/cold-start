import gc
import os
import time
import sys
import torch
from torch import nn
from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer  # no longer needed if purely torchvision
import torchvision.models as models
from assemble_model_example import assemble_model
from safetensors.torch import load_file

def _warmup_cuda_tv():
    num_gpus = torch.cuda.device_count()
    print(f"Warming up {num_gpus} GPUs")
    for i in tqdm(range(num_gpus)):
        torch.ones(1).to(f"cuda:{i}")
        torch.cuda.synchronize()

def _warmup_inference_tv():
    print("Warming up inference for MobileNetV2")
    import torchvision.models as models
    model = models.mobilenet_v2(pretrained=True).eval().cuda()
    dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.no_grad():
        _ = model(dummy_input)
    del model
    gc.collect()
    torch.cuda.empty_cache()

def benchmark_inference(model: nn.Module):
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    start_time = time.time()
    with torch.no_grad():
        output = model(dummy_input)
    end_time = time.time()
    inference_time = end_time - start_time
    throughput = 1.0 / inference_time  # 1 image / time
    output_info = str(output.shape)
    del output, dummy_input
    gc.collect()
    torch.cuda.empty_cache()
    return inference_time, throughput, output_info

def measure_single_tv(model_name: str, model_format: str, model_dir: str, replica: int):
    print(f"Loading {model_name}_{replica}")
    model_record = {"model_name": f"{model_name}_{replica}"}

    if model_format == "iceCrusher":
        start_time = time.time()
        model, timings = assemble_model(
            model_type='torchvision',
            model_name='resnet50',
            model_path='./'
        )
        end_time = time.time()
        model_record['profiling'] = timings

    elif model_format == "safetensors":

        start_time = time.time()
        # 1) Create the model architecture (pretrained=False, since we will load our own weights)
        model = models.resnet50(pretrained=False)

        # 2) Build the path to your safetensors file
        #    Example assumption: "mobilenet_v2.safetensors" is located under model_dir
        safetensors_path = os.path.join(model_dir, "vgg19_safetensors/model.safetensors")
        if not os.path.isfile(safetensors_path):
            raise FileNotFoundError(f"Could not find safetensors file at {safetensors_path}")

        # 3) Load state_dict from the safetensors file
        state_dict = load_file(safetensors_path)

        # 4) Assign to model
        model.load_state_dict(state_dict)

        # 5) Move to GPU and eval mode
        model = model.eval().cuda()

        end_time = time.time()


    else:
        raise ValueError("Unknown model format. Use 'iceCrusher' or 'safetensors'.")

    model_record["loading_time_sec"] = end_time - start_time

    # Inference benchmark
    inf_time, throughput, output_info = benchmark_inference(model)
    model_record["inference_time_sec"] = inf_time
    model_record["throughput"] = throughput
    model_record["output_info"] = output_info

    print("Benchmark Result:", model_record)

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return model_record
