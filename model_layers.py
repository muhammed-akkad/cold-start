import torch
from transformers import OPTForCausalLM

# Load the model with pretrained weights
model_name = "facebook/opt-6.7b"
model = OPTForCausalLM.from_pretrained(model_name)

# Get the state_dict (which includes all trained weights and biases)
state_dict = model.state_dict()

# Calculate total size in bytes
total_size_bytes = sum(param.numel() * param.element_size() for param in state_dict.values())

# Convert bytes to megabytes (1 MB = 1024 * 1024 bytes)
total_size_mb = total_size_bytes / (1024 * 1024)

print(f"Total number of parameter tensors in state_dict: {len(state_dict)}")
print(f"Total size of state_dict (including weights): {total_size_mb:.2f} MB")
