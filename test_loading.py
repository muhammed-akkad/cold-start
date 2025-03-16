import argparse
import json
import os
from benchmark_utils import _warmup_cuda, _warmup_inference, measure_single
from benchmark_util_torch import _warmup_inference_tv, measure_single_tv
def save_result_to_json(result, output_filename):
    """
    Appends 'result' to a list of results in 'output_filename'.
    If 'output_filename' doesn't exist, it creates a new file with a list.
    """
    # 1) Check if the file exists
    if os.path.exists(output_filename):
        # 2) If so, load existing data
        with open(output_filename, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # If the file is empty or corrupted, start a new list
                data = []
    else:
        data = []

    # 3) Append the new result
    data.append(result)

    # 4) Write back to JSON
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Appended results to {output_filename}")

def get_args():
    parser = argparse.ArgumentParser(description="Single Model Load Benchmark")
    parser.add_argument("--model-type", type=str, required=True, choices=["tv", "hf"])
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-format", type=str, required=True, choices=["iceCrusher", "safetensors"])
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--replica", type=int, default=0)
    
    return parser.parse_args()

def main():
    args = get_args()
    _warmup_cuda()
    if args.model_type == "tv":
        _warmup_inference_tv()
        result = measure_single_tv(args.model_name, args.model_format, args.model_dir, args.replica)
    else:
        _warmup_inference()
        result = measure_single(args.model_name, args.model_format, args.model_dir, args.replica)


    output_filename = "results_facebook_opt-6.7b_B.json"
    save_result_to_json(result, output_filename)

    print(f"Saved benchmark results to {output_filename}")

if __name__ == "__main__":
    main()
