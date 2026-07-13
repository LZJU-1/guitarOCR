from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(ConvBlock(channels, channels), ConvBlock(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class TabSymbolDetector(nn.Module):
    """Compact CenterNet-style detector with output stride 4."""

    output_stride = 4

    def __init__(self, class_count: int = 11) -> None:
        super().__init__()
        self.class_count = class_count
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ResidualBlock(64),
            ConvBlock(64, 96),
            ResidualBlock(96),
            ResidualBlock(96),
        )
        self.heatmap_head = nn.Sequential(
            ConvBlock(96, 64),
            nn.Conv2d(64, class_count, 1),
        )
        self.size_head = nn.Sequential(ConvBlock(96, 32), nn.Conv2d(32, 2, 1))
        self.offset_head = nn.Sequential(ConvBlock(96, 32), nn.Conv2d(32, 2, 1))
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.heatmap_head(features), self.size_head(features), self.offset_head(features)
