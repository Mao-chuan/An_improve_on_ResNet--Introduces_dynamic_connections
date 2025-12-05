import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class DynamicGate(nn.Module):
    def __init__(self, forward_layers: int):
        super().__init__()
        self.forward_layers = forward_layers
        self.weights = nn.Parameter(torch.ones(forward_layers))

    def forward(self, x: torch.Tensor) -> List[Optional[torch.Tensor]]:
        if self.forward_layers == 0:
            return []

        gate_weights = torch.sigmoid(self.weights)  # [forward_layers]

        forward_outputs = []
        for i in range(self.forward_layers):
            weighted_x = x * gate_weights[i]
            forward_outputs.append(weighted_x)

        return forward_outputs

class AdaptiveConnection(nn.Module):

    def __init__(self, in_channels, out_channels, target_spatial_size):
        super().__init__()
        self.target_spatial_size = target_spatial_size

        if in_channels != out_channels:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.channel_adapter = nn.Identity()

        # print(target_spatial_size)

    def forward(self, x):
        x = self.channel_adapter(x)

        # if(x.size(2) != self.target_spatial_size[0] or x.size(3) != self.target_spatial_size[1]):print(x.size,self.target_spatial_size,"True")
        # else:print(x.size,self.target_spatial_size,"False")
        if x.size(2) != self.target_spatial_size[0] or x.size(3) != self.target_spatial_size[1]:
            if self.target_spatial_size[0] > 0 and self.target_spatial_size[1] > 0:
                x = F.interpolate(
                    x,
                    size=self.target_spatial_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                valid_size = (max(1, self.target_spatial_size[0]), max(1, self.target_spatial_size[1]))
                x = F.adaptive_avg_pool2d(x, valid_size)

        return x
class DResNetBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layer_idx: int,
                 total_layers: int,
                 stride: int = 1):
        super().__init__()

        self.layer_idx = layer_idx
        self.total_layers = total_layers

        self.forward_layers = total_layers - layer_idx - 1

        self.dynamic_gate = DynamicGate(self.forward_layers) if self.forward_layers > 0 else None

        self.adapters = nn.ModuleDict()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def add_adapter(self, source_layer: int, in_channels: int, spatial_size: Tuple[int, int]):
        adapter_name = f'adapter_{source_layer}'
        # print(spatial_size)
        self.adapters[adapter_name] = AdaptiveConnection(
            in_channels, self.conv1.in_channels, spatial_size
        )

    def forward(self,
                x: torch.Tensor,
                previous_outputs: List[Tuple[int, torch.Tensor]]) -> Tuple[
        torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        combined_input = x.clone()

        for source_layer, prev_output in previous_outputs:
            adapter_name = f'adapter_{source_layer}'
            if adapter_name in self.adapters:
                # print(prev_output.shape,combined_input.shape)
                # raise ValueError("!!!")
                adapted_output = self.adapters[adapter_name](prev_output)  # 64
                # print(adapted_output.shape)
                combined_input = combined_input + adapted_output  # 64 32

        identity = combined_input
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.conv1(combined_input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        current_output = self.relu(out)

        forward_outputs = []
        if self.dynamic_gate is not None:
            gate_outputs = self.dynamic_gate(current_output)
            for i, gate_out in enumerate(gate_outputs):
                if gate_out is not None:
                    target_layer = self.layer_idx + i + 1
                    forward_outputs.append((target_layer, gate_out))

        return current_output, forward_outputs


class DResNet(nn.Module):

    def __init__(self,
                 num_classes: int = 200,
                 dropout_rate: float = 0.2):
        super().__init__()

        self.layer_config = [
            # (in_channels, out_channels, stride)
            (64, 64, 1),  # block1
            (64, 64, 1),  # block2
            (64, 128, 2),  # block3
            (128, 128, 1),  # block4
            (128, 256, 2),  # block5
            (256, 256, 1),  # block6
            (256, 512, 2),  # block7
            (512, 512, 1),  # block8
        ]

        self.total_layers = len(self.layer_config)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.blocks = nn.ModuleList()
        self.layer_info = []

        spatial_sizes = []

        current_size = 64
        for _, _, stride in self.layer_config:
            spatial_sizes.append((current_size, current_size))
            if stride > 1:
                current_size = current_size // 2

        for i, (in_ch, out_ch, stride) in enumerate(self.layer_config):
            block = DResNetBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                layer_idx=i,
                total_layers=self.total_layers,
                stride=stride
            )

            self.blocks.append(block)
            self.layer_info.append({
                'channels': out_ch,
                'spatial_size': spatial_sizes[i],
                'layer_idx': i
            })

        self._setup_adapters()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _setup_adapters(self):
        for i, block in enumerate(self.blocks):
            current_info = self.layer_info[i]

            for j in range(i):
                source_info = self.layer_info[j]
                block.add_adapter(
                    source_layer=source_info['layer_idx'],
                    in_channels=source_info['channels'],
                    spatial_size=current_info['spatial_size']
                )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        all_outputs = [None] * self.total_layers
        forward_buffer = [[] for _ in range(self.total_layers)]

        for i, block in enumerate(self.blocks):
            previous_outputs = []
            for source_layer, output_tensor in forward_buffer[i]:
                previous_outputs.append((source_layer, output_tensor))

            forward_buffer[i] = []

            current_output, new_forward_outputs = block(x, previous_outputs)

            all_outputs[i] = current_output
            x = current_output

            for target_layer, output_tensor in new_forward_outputs:
                if target_layer < self.total_layers:
                    forward_buffer[target_layer].append([i, output_tensor])

        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

