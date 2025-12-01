import torch
import torch.nn.functional as F
from torch import nn

class DResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, sequence=0):  # sequence start with 1
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.bn3 = nn.BatchNorm2d(out_channels)

        self.__can_append = sequence > 2
        if self.__can_append:
            # self.__can_append = True
            self.cite_convolutions = nn.ModuleList()
            self.cite_adaptive_shape = nn.ModuleList()
            self.cite_bns = nn.ModuleList()

    def cite_append(self, x, cite):
        if self.__can_append:
            shape = x.shape
            out_channel = shape[1]
            for i in cite:
                if i is not None:
                    in_channel = i.shape[1]
                    self.cite_convolutions.append(nn.Conv2d(in_channel, out_channel, padding=0, stride=1, kernel_size=1))
                    self.cite_adaptive_shape.append(nn.AdaptiveAvgPool2d((shape[-2],shape[-1])))
                    self.cite_bns.append(nn.BatchNorm2d(out_channel))
                else:
                    continue
            self.__can_append = False

    def forward(self, x, cite: list):
        if self.__can_append:
           self.cite_append(x, cite)

        if not self.__can_append:
            for idx,c in enumerate(cite):
                if c is not None:
                    dealed_cite = self.cite_convolutions[idx](c)
                    dealed_cite = self.cite_adaptive_shape[idx](dealed_cite)
                    dealed_cite = self.cite_bns[idx](dealed_cite)

                    x = x + dealed_cite

                else:
                    continue

        # x = x + f(cite)

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

class DResNet(nn.Module):
    def __init__(self, num_class=200, dropout_rate=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = DResNetBlock(64, 64, sequence=1)
        self.block2 = DResNetBlock(64, 64, stride=1, sequence=2)
        self.block3 = DResNetBlock(64, 128, stride=2, sequence=3)
        self.block4 = DResNetBlock(128, 128, stride=1, sequence=4)
        self.block5 = DResNetBlock(128, 256, stride=2, sequence=5)
        self.block6 = DResNetBlock(256, 256, stride=1, sequence=6)
        self.block7 = DResNetBlock(256, 512, stride=2, sequence=7)
        self.block8 = DResNetBlock(512, 512, stride=1, sequence=8)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_class)

        self.outputs = [None for _ in range(6)]  # store the output of each layer in each batch

        self.weight = nn.Parameter(torch.eye(6) + 0.1 * torch.rand(6, 6) * torch.tril(torch.ones(6, 6), diagonal=-1))

    def forward(self, x):
        self.outputs = [None for _ in range(6)]  # reset the output of last batch

        B = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.block1(x, self.outputs)
        self.outputs[0] = x

        x = self.block2(x, self.outputs)
        self.outputs[1] = x

        x = self.block3(x, [o * w for o, w in zip(self.outputs,self.weight[0])])
        self.outputs[2] = x

        x = self.block4(x, [o * w for o, w in zip(self.outputs,self.weight[1])])
        self.outputs[3] = x

        x = self.block5(x, [o * w for o, w in zip(self.outputs,self.weight[2])])
        self.outputs[4] = x

        x = self.block6(x, [o * w for o, w in zip(self.outputs,self.weight[3])])
        self.outputs[5] = x

        x = self.block7(x, [o * w for o, w in zip(self.outputs,self.weight[4])])
        # self.outputs[6] = x

        x = self.block8(x, [o * w for o, w in zip(self.outputs,self.weight[5])])
        # self.outputs[7] = x

        x = self.avg(x)

        x = x.view(B, -1)

        x = self.dropout(x)
        x = self.fc(x)

        return x

