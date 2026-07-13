from __future__ import annotations

import torch
from torch import nn


STATE_CLASSES = ["empty", "note", "rest"]
DURATION_CLASSES = [1, 2, 4, 8, 16, 32, 64]
DOT_CLASSES = ["none", "single", "double"]
DIVISION_CLASSES = ["1:1", "3:2", "6:4", "4:2", "5:4", "10:8", "12:8", "7:4", "9:8"]
INPUT_WIDTH = 256
INPUT_HEIGHT = 192


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
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


class RhythmContextCNN(nn.Module):
    """Predict both TuxGuitar voices for an event-centred score crop.

    The full crop is retained because beams, ties and simultaneous voices cross
    artificial top/bottom boundaries. The current corpus contains only two
    songs with visible voice 1, so voice-1 results must be reported separately.
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 24, 5, 2),
            ResidualBlock(24),
            ConvBlock(24, 48, 3, 2),
            ResidualBlock(48),
            ConvBlock(48, 80, 3, 2),
            ResidualBlock(80),
            ConvBlock(80, 128, 3, 2),
            ResidualBlock(128),
            ConvBlock(128, 160, 3, 2),
            ResidualBlock(160),
            nn.AdaptiveAvgPool2d((3, 4)),
        )
        self.context = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160 * 3 * 4, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
        )
        self.heads = nn.ModuleList()
        for _voice_index in range(2):
            self.heads.extend(
                [
                    nn.Linear(256, len(STATE_CLASSES)),
                    nn.Linear(256, len(DURATION_CLASSES)),
                    nn.Linear(256, len(DOT_CLASSES)),
                    nn.Linear(256, len(DIVISION_CLASSES)),
                ]
            )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.context(self.backbone(inputs))
        return tuple(head(features) for head in self.heads)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
