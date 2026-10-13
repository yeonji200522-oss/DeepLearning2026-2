"""MBConv CNN trained from scratch; standalone torch-only submission code."""
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.layers(x)


def conv_norm_act(in_channels, out_channels, kernel=1, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class MBConv(nn.Module):
    """Expand channels, filter spatially, attend, then project to a narrow output."""
    def __init__(self, in_channels, out_channels, stride=1, kernel=3, expansion=4):
        super().__init__()
        expanded = in_channels * expansion
        self.expand = (conv_norm_act(in_channels, expanded)
                       if expansion != 1 else nn.Identity())
        self.depthwise = conv_norm_act(expanded, expanded, kernel, stride, expanded)
        # Squeeze relative to input width, not the expanded width, to keep SE small.
        self.attention = ChannelAttention(expanded, max(4, in_channels // 4))
        self.project = nn.Sequential(
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_path = 0.0

    def forward(self, x):
        output = self.project(self.attention(self.depthwise(self.expand(x))))
        if self.use_residual:
            if self.training and self.drop_path:
                keep = 1 - self.drop_path
                mask = (torch.rand((x.shape[0], 1, 1, 1), device=x.device) < keep).to(x.dtype)
                output = output * mask / keep
            output = x + output
        return output


class MultiTaskCNN(nn.Module):
    def __init__(self, num_crops=3, num_health=2):
        super().__init__()
        self.stem = conv_norm_act(3, 24, kernel=3)
        # (output channels, repeats, first stride, spatial kernel, expansion).
        # Retain 16x16 leaf/lesion features for nine blocks before the last reduction.
        stages = (
            (16, 1, 1, 3, 1),
            (24, 2, 2, 3, 4),
            (40, 3, 2, 5, 4),
            (64, 5, 2, 5, 6),
            (96, 4, 1, 5, 6),
            (160, 2, 2, 5, 6),
        )
        blocks = []
        channels = 24
        for out_channels, repeats, stride, kernel, expansion in stages:
            for repeat in range(repeats):
                blocks.append(MBConv(channels, out_channels,
                                     stride if repeat == 0 else 1, kernel, expansion))
                channels = out_channels
            if out_channels == 96:
                self.detail_index = len(blocks) - 1
        blocks.append(conv_norm_act(channels, 192))
        self.features = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Fuse 16x16 detail and 8x8 semantic features, with a crop-only spatial route.
        self.detail_projection = nn.Linear(192, 384, bias=False)
        self.crop_layout = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(384, 64, bias=False),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 320, bias=False),
        )
        self.crop_features = self._head_features(320)
        self.health_features = self._head_features(256)
        self.crop_head = nn.Linear(320, num_crops)
        self.health_head = nn.Linear(256, num_health)
        self._initialize()

    @staticmethod
    def _head_features(width):
        return nn.Sequential(
            nn.Linear(384, width),
            nn.LayerNorm(width),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
        )

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                mode = "fan_in" if module.groups == module.in_channels else "fan_out"
                nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.detail_projection.weight)
        nn.init.zeros_(self.crop_layout[-1].weight)

    def forward(self, x):
        x = self.stem(x)
        for index, block in enumerate(self.features):
            x = block(x)
            if index == self.detail_index:
                detail_map = x
                detail = torch.cat((self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)), dim=1)
        features = torch.cat((self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)), dim=1)
        features = features + self.detail_projection(detail)
        return (
            self.crop_head(self.crop_features(features) + self.crop_layout(detail_map)),
            self.health_head(self.health_features(features)),
        )
