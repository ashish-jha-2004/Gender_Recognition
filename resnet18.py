import torch.nn.functional as F
import torchvision.models as models
from torch import nn


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.freeze_layer4_convs = ['1.conv2.weight', '1.conv1.weight', '0.conv2.weight', '0.conv1.weight']
        self.freeze_layer3_convs = ['1.conv2.weight', '1.conv1.weight', '0.conv2.weight', '0.conv1.weight']
        self.freeze_layer2_convs = ['1.conv2.weight', '1.conv1.weight']
        self.resnet18 = models.resnet18(pretrained=True)
        self.set_parameter_requires_grad()
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 8)

    def set_parameter_requires_grad(self):
        for param in self.resnet18.parameters():
            param.requires_grad = False

        for name, children in self.resnet18.named_children():
            if name == 'layer2':
                for child, params in children.named_parameters():
                    if child in self.freeze_layer2_convs:
                        params.requires_grad = True

            if name == 'layer3':
                for child, params in children.named_parameters():
                    if child in self.freeze_layer3_convs:
                        params.requires_grad = True

            if name == 'layer4':
                for child, params in children.named_parameters():
                    if child in self.freeze_layer4_convs:
                        params.requires_grad = True

        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet18(x)
        output = F.log_softmax(x, dim=1)
        return output
