
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b7
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

class efficientnet(nn.Module):
    def __init__(self):
        super(efficientnet, self).__init__()

        self.model = efficientnet_b7(pretrained=True) 
        list(self.model.features.children())[0][0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = nn.Sequential(self.model.classifier[0], 
                                            nn.Linear(in_features=2560, out_features=100, bias=True),
                                            nn.Linear(in_features=100, out_features=2, bias=True)) 
    def forward(self, x):
        return self.model(x)

# vit input size must be 224 x 224
# class vit(nn.Module):
#     def __init__(self):
#         super(vit, self).__init__()

#         self.model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT) 
#         self.model.conv_proj = nn.Conv2d(1, 1024, kernel_size=(16, 16), stride=(16, 16))
#         self.model.heads.head = nn.Sequential( 
#                                             nn.Linear(in_features=1024, out_features=100, bias=True),
#                                             nn.Linear(in_features=100, out_features=2, bias=True))
#     def forward(self, x):
#         return self.model(x)


class resnext(nn.Module):
    def __init__(self):
        super(resnext, self).__init__()

        self.model = resnext101_64x4d(weights = ResNeXt101_64X4D_Weights.DEFAULT) 
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = resnext()

    B = 3
    H = 192
    W = 256

    x = torch.rand(B, 1, H, W)

    x= model.forward(x)

    print(x.shape)