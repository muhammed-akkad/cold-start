#!/bin/bash
set -e

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 model_type model_name model_format models_directory num_replicas"
    echo "Example: $0 tv mobilenet_v2 safetensors ./models 5"
    exit 1
fi

MODEL_TYPE=$1
MODEL_NAME=$2
MODEL_FORMAT=$3
MODELS_DIR=$4
NUM_REPLICAS=$5

if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory $MODELS_DIR does not exist."
    exit 1
fi

# 1) If the user wants a torchvision model in safetensors format, download it.
if  [ "$MODEL_FORMAT" = "safetensors" ]; then
    echo "Downloading $MODEL_NAME and saving to safetensors via download_models.py..."
    python3 download_models.py \
        --model-name "$MODEL_NAME" \
        --save-dir "$MODELS_DIR"   \
        --save-format "$MODEL_FORMAT" \
        --model-type "$MODEL_TYPE"
fi

# 2) Run the test_loading.py for the requested number of replicas
for ((i=0; i<$NUM_REPLICAS; i++)); do
    echo "Running replica $i for model $MODEL_NAME with format $MODEL_FORMAT..."
    python3 test_loading.py \
      --model-type "$MODEL_TYPE" \
      --model-name "$MODEL_NAME" \
      --model-format "$MODEL_FORMAT" \
      --model-dir "$MODELS_DIR" \
      --replica "$i"
done
