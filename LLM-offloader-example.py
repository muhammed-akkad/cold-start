from transformers import GPT2LMHeadModel
from model-offloader-example import save_dict
# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")

# Save model parameters using your save_dict function
save_dict(model.state_dict(), "./llm_test", gpu_percent=40, cpu_percent=40)
