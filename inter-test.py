

import torch
import torchvision.models as models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=True).to(device) 