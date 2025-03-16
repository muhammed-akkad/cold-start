# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import argparse
import os

import torch
from tqdm import tqdm

# HF stuff (only used if model_type=="hf")
from transformers import AutoModelForCausalLM, AutoTokenizer

# TV stuff (only used if model_type=="tv")
import torchvision.models as models

# For saving custom "SLLM" format (if you have such a function)
from model_offloader_example import save_model

# For safetensors if user wants it
from safetensors.torch import save_file


def get_args():
    parser = argparse.ArgumentParser(description="Save a model with ServerlessLLM")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["hf", "tv"],
        help="Type of the model: 'hf' for Hugging Face, 'tv' for torchvision",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to save (e.g., 'facebook/opt-1.3b' or 'mobilenet_v2')",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        required=True,
        choices=["sllm", "safetensors"],
        help="Format to save the model in",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save models",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help="Number of replicas to save",
    )
    return parser.parse_args()


def load_hf_model(model_name: str):
    """
    Loads a Hugging Face model (AutoModelForCausalLM) in FP16,
    along with its tokenizer.
    """
    print(f"Loading Hugging Face model {model_name} into memory...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_tv_model(model_name: str):
    """
    Loads a torchvision model by name, in FP32 (typical for pretrained).
    If you want to force half precision, you can manually cast to half.
    """
    print(f"Loading torchvision model {model_name} into memory...")

    # Simple example: handle some popular models
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError(
            f"Unsupported torchvision model_name='{model_name}'. "
            "Add it to load_tv_model(...) if needed."
        )

    model.eval()  # Inference mode
    # Optionally cast to half if you want:
    # model.half()
    # but be sure to do model.cuda() if you want it on GPU prior to saving
    return model


def main():
    args = get_args()

    save_dir = args.save_dir
    save_format = args.save_format
    model_name = args.model_name
    model_type = args.model_type
    replicas = args.num_replicas

    # Ensure directory exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory {save_dir} does not exist")

    # 1) Load model
    tokenizer = None
    if model_type == "hf":
        model, tokenizer = load_hf_model(model_name)
    elif model_type == "tv":
        model = load_tv_model(model_name)
    else:
        raise ValueError("model_type must be 'hf' or 'tv'")

    # 2) Save replicas
    if save_format == "sllm":
        print(f"Saving {replicas} SLLM models to {save_dir}")
        for i in tqdm(range(replicas)):
            model_dir = os.path.join(save_dir, f"{model_name}_{i}")
            os.makedirs(model_dir, exist_ok=True)

            # If you have a custom function to save "SLLM" format
            # it might need GPU/CPU ratios, etc. This is a placeholder call:
            save_model(model, model_dir, gpu_percent=40, cpu_percent=40)

            # For HF models, you might also save tokenizer:
            if tokenizer is not None:
                tokenizer.save_pretrained(model_dir)

    elif save_format == "safetensors":
        print(f"Saving {replicas} safetensors models to {save_dir}")
        for i in tqdm(range(replicas)):
            # We'll store each replica in a subdirectory, e.g. "mymodel_safetensors_0"
            model_dir = os.path.join(save_dir, f"{model_name}_safetensors")
            os.makedirs(model_dir, exist_ok=True)

            if model_type == "hf":
                # Hugging Face's built-in "save_pretrained" can produce .bin or .safetensors
                # if you have 'safetensors' installed and set SAFE_TENSORS_AVAILABLE=1
                # Otherwise, you can just do model.save_pretrained(...) which might produce .bin
                # THEN you'd convert to safetensors. Or do the param-based approach:

                model.save_pretrained(model_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(model_dir)

            else:
                # model_type == "tv"
                # We'll save the torchvision model's state_dict to safetensors
                from safetensors.torch import save_file

                # E.g. "model.safetensors" within the subdirectory
                safetensors_path = os.path.join(model_dir, "model.safetensors")
                state_dict = model.state_dict()
                save_file(state_dict, safetensors_path)

                # Optionally store some metadata about architecture
                # so you can reload it. E.g.:
                with open(os.path.join(model_dir, "arch.txt"), "w") as f:
                    f.write(model_name)

    else:
        raise ValueError(f"Invalid save format {save_format}")

    print("Done saving!")


if __name__ == "__main__":
    main()
