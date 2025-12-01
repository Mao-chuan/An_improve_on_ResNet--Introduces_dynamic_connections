import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class CnnBlock(nn.Module):

    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x

class CNN(nn.Module):

    def __init__(self,num_classes=200,dropout_rate=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.cnnBlock1 = CnnBlock(64, 64)
        self.cnnBlock2 = CnnBlock(64, 64, stride=1)
        self.cnnBlock3 = CnnBlock(64, 128, stride=2)
        self.cnnBlock4 = CnnBlock(128, 128, stride=1)
        self.cnnBlock5 = CnnBlock(128, 256, stride=2)
        self.cnnBlock6 = CnnBlock(256, 256, stride=1)
        self.cnnBlock7 = CnnBlock(256, 512, stride=2)
        self.cnnBlock8 = CnnBlock(512, 512, stride=1)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.cnnBlock1(x)
        x = self.cnnBlock2(x)
        x = self.cnnBlock3(x)
        x = self.cnnBlock4(x)
        x = self.cnnBlock5(x)
        x = self.cnnBlock6(x)
        x = self.cnnBlock7(x)
        x = self.cnnBlock8(x)

        x = self.avg(x)

        x = x.view(B, -1)

        x = self.dropout(x)
        x = self.fc(x)

        return x