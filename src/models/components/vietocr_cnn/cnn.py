import torch
from torch import nn

import src.models.components.vietocr_cnn.vgg  as vgg
from src.models.components.vietocr_cnn.resnet import Resnet50, Resnet150

class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()
        self.backbone = backbone

        if backbone == 'vgg11_bn':
            self.model = vgg.vgg11_bn(**kwargs)
        elif backbone == 'vgg19_bn':
            self.model = vgg.vgg19_bn(**kwargs)
        elif backbone == 'resnet50':
            self.model = Resnet50(**kwargs)
        elif backbone == 'resnet150':
            self.model = Resnet150(**kwargs)

    def forward(self, x):
        return self.model(x)
    
    def freeze_layers(self, number_of_freezed_layer):
        if self.backbone == 'vgg19_bn':
            for i, param in enumerate(self.model.features.parameters()):
                if i <= number_of_freezed_layer:
                    param.requires_grad = False
                else:
                    break
        elif self.backbone == 'resnet50':
            for i, param in enumerate(self.model.parameters()):
                if i <= number_of_freezed_layer:
                    param.requires_grad = False
                else:
                    break

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
