import os
import torch
import torch.nn as nn
import torchvision.models as models

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and first_block == False:     # (H, W) -> (H/2, W/2)
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
    else:
        blk.append(Residual(num_channels, num_channels))
    return blk

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, padding=1, kernel_size=3, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        y = self.relu(y)
        return y

class MyResNet18(nn.Module):
    def __init__(self, n_out):
        super(MyResNet18, self).__init__()

        self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # 1

        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 4
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))   # 4
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))  # 4
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))  # 4

        # nn.Linear()   1
        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, n_out))

    def forward(self, x):
        return self.net(x)

class MyResNet50(nn.Module):
    ...

myResNet50 = models.resnet50(pretrained=True)
myResNet50.fc = nn.Linear(2048, 5)

myResNet18 = models.resnet18(pretrained=True)
myResNet18.fc = nn.Linear(512, 5)

