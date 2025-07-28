import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    """
    Residual bottleneck block used in ResNet architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    downsample : bool
        Whether to apply stride for temporal downsampling.

    Returns
    -------
    Output tensor after applying bottleneck residual connection.
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, stride)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        return F.relu(out + identity)


class ResNetSleep(nn.Module):
    """
    Deep residual network model for sleep stage classification.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default: 5).
    base_filters : int
        Base number of convolutional filters (default: 16).
    block_layers : int
        Number of residual block layers (default: 4).
    num_classes : int
        Number of output sleep stages (default: 5).
    """
    def __init__(self, in_channels=5, base_filters=16, block_layers=4, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1, 16), stride=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.res_blocks = nn.Sequential()
        channels = 64
        for i in range(block_layers):
            for j in range(4):
                downsample = j == 0 and i > 0
                self.res_blocks.add_module(
                    f"block_{i}_{j}",
                    BottleneckBlock(channels, base_filters * (2 ** i), downsample)
                )
                channels = base_filters * (2 ** i)

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.res_blocks(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
