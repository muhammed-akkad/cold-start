import json
import os
from typing import List, Optional, Tuple, Union, Dict
import torch
import torchvision.models as models
import cuda_saver  # Import the C++ module
import sys
from io import StringIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.models import MobileNet_V3_Large_Weights

model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).to(device)




def save_dict(model_state_dict: Dict[str, torch.Tensor], model_path: str):
    tensor_names = list(model_state_dict.keys())
    tensor_data_index = {}
    for name, param in model_state_dict.items():
        if param.is_cuda:
            data_ptr = param.data_ptr()  
            size = param.numel() * param.element_size()  
            tensor_data_index[name] = (data_ptr, size)
        else:
            raise ValueError(f"Tensor {name} is not on the GPU.")

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    # save tensors using the C++ function
    print("Starting save_tensors_cpp")

    tensor_offsets = cuda_saver.save_tensors_to_gpu(tensor_names, tensor_data_index)
    
    output = sys.stdout.getvalue()

    # Reset stdout to original
    sys.stdout = original_stdout

    print("Captured C++ output:")
    print(output)
    # create tensor index
    tensor_index = {}
    for name, param in model_state_dict.items():
        # name: offset, size
        tensor_index[name] = (tensor_offsets[name], tensor_data_index[name][1], tuple(param.shape), tuple(param.stride()), str(param.dtype))

    # save tensor index
    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)



def main():
        save_dict(model.state_dict(), "./")
        
if __name__ == "__main__":
    main()

