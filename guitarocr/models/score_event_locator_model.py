from __future__ import annotations

import torch
from torch import nn


INPUT_WIDTH = 512
INPUT_HEIGHT = 192
OUTPUT_STRIDE = 4


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.body(inputs))


class Conv1dBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
        )


class ScoreEventLocator(nn.Module):
    """Locate onset columns after collapsing a score measure along y."""

    output_stride = OUTPUT_STRIDE

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 24, stride=2),
            DepthwiseResidual(24),
            ConvBlock(24, 48, stride=2),
            DepthwiseResidual(48),
            ConvBlock(48, 80),
            DepthwiseResidual(80),
            DepthwiseResidual(80),
        )
        self.context = nn.Sequential(
            Conv1dBlock(160, 96),
            Conv1dBlock(96, 96),
        )
        self.heatmap_head = nn.Sequential(Conv1dBlock(96, 48), nn.Conv1d(48, 1, 1))
        self.offset_head = nn.Sequential(Conv1dBlock(96, 32), nn.Conv1d(32, 1, 1))
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        vertical_context = torch.cat((features.amax(dim=2), features.mean(dim=2)), dim=1)
        context = self.context(vertical_context)
        return self.heatmap_head(context), self.offset_head(context)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
