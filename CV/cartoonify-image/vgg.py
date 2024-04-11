from collections import namedtuple
import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        # vgg_pretrained_features = models.vgg16(pretrained=True).features
        vgg_pretrained_features = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
    
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, X):
        y = self.slice1(X)
        y_relu1_2 = y
        y = self.slice2(y)
        y_relu2_2 = y
        y = self.slice3(y)
        y_relu3_3 = y
        y = self.slice4(y)
        y_relu4_3 = y
        outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = outputs(y_relu1_2, y_relu2_2, y_relu3_3, y_relu4_3)
        return out