"""A small residual CNN trained from scratch; only torch is required at inference."""
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 8)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = ChannelAttention(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.attention(self.bn2(self.conv2(x)))
        return self.relu(x + residual)


class MultiTaskCNN(nn.Module):
    def __init__(self, num_crops=3, num_health=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32, 2),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 192, 2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.crop_features = self._head_features()
        self.health_features = self._head_features()
        self.crop_head = nn.Linear(128, num_crops)
        self.health_head = nn.Linear(128, num_health)
        self._initialize()

    @staticmethod
    def _head_features():
        return nn.Sequential(
            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, x):
        x = self.features(self.stem(x))
        features = torch.cat((self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)), dim=1)
        return (
            self.crop_head(self.crop_features(features)),
            self.health_head(self.health_features(features)),
        )
