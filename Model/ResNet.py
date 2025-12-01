import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding=0)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        inputs = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            inputs = self.downsample(inputs)
            inputs = self.bn3(inputs)

        out = x + inputs
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self,num_class=200, dropout_rate=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = ResidualBlock(64,64, stride=1)
        self.block2 = ResidualBlock(64,64, stride=1)
        self.block3 = ResidualBlock(64,128,stride=2)
        self.block4 = ResidualBlock(128,128,stride=1)
        self.block5 = ResidualBlock(128,256,stride=2)
        self.block6 = ResidualBlock(256,256,stride=1)
        self.block7 = ResidualBlock(256,512,stride=2)
        self.block8 = ResidualBlock(512,512,stride=1)

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512,num_class)

    def forward(self, x):
        B = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.avg(x)

        x = x.view(B, -1)

        x = self.dropout(x)
        x = self.fc(x)

        return x