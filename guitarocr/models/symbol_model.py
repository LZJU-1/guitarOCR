from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class AtomicSymbolCNN(nn.Module):
    """Small grayscale classifier for pre-cropped 64x64 notation symbols."""

    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 24),
            ConvBlock(24, 32, stride=2),
            ConvBlock(32, 48),
            ConvBlock(48, 64, stride=2),
            ConvBlock(64, 96),
            ConvBlock(96, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.15),
            nn.Linear(128, class_count),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
