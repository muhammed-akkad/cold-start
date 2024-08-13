import json
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from collections import OrderedDict
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Example usage:
model = models.resnet18(pretrained=True).to(device)  # Your PyTorch model

# Initialize a list to hold information about each layer
layers_info = []

layers = list(model.children()) 

# Save each sub-model
for i, sub_model in enumerate(layers):
    file_name = f'sub_model_{i}.pth'
    torch.save(sub_model.state_dict(), file_name)
    # Get the size of the .pth file
    file_size_in_bytes = os.path.getsize(file_name)
    # Gather information about the layer
    layer_info = {
        "index": i,
        "type": str(type(sub_model)),
        "name": f"sub_model_{i}.pth",
        "file_size": file_size_in_bytes  / (1024 ** 2) 
   }
    
    # Append this layer's information to the list
    layers_info.append(layer_info)

# Save the JSON object to a file
with open("model_layers_info.json", "w") as json_file:
    json.dump(layers_info, json_file, indent=4)
    
# Load sub-models
loaded_sub_models = []
for i in range(len(layers)):
    sub_model = layers[i]
    sub_model.load_state_dict(torch.load(f'sub_model_{i}.pth'))
    loaded_sub_models.append(sub_model)
    

# Define the data transform
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load a sample image from CIFAR-10 dataset
def load_sample_image():
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    img, label = dataset[10]  # Get the first image
    return img.unsqueeze(0)  # Add batch dimension

sample_image = load_sample_image().to(device)




# apply the model the loaded image
index=0
output = sample_image
for sub_model in loaded_sub_models:
    print(index,sub_model)
    if isinstance(sub_model, (nn.Linear)):
        output = output.view(output.size(0), -1)  # Flatten to (batch_size, channels)
    output = sub_model(output)
    index+=1    


predicted_class = torch.argmax(output).item()


# Display the result
print(f'Predicted class: {predicted_class}')

# Convert tensor to PIL image for visualization
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)  # Denormalize
    return Image.fromarray(tensor)

# Show the image
img = tensor_to_image(sample_image)
plt.imshow(img)
plt.title(f'Predicted class: {predicted_class}')
plt.show() 



