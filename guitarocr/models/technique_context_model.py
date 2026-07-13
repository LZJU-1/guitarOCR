from __future__ import annotations

import torch
from torch import nn

from guitarocr.data.build_score_rhythm_dataset import TECHNIQUE_CLASSES
from guitarocr.models.rhythm_context_model import ConvBlock, ResidualBlock, INPUT_HEIGHT, INPUT_WIDTH


class TechniqueContextCNN(nn.Module):
    """Small event-context multi-label classifier for guitar techniques."""

    def __init__(self, class_count: int = len(TECHNIQUE_CLASSES)):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 24, 5, 2), ResidualBlock(24),
            ConvBlock(24, 48, 3, 2), ResidualBlock(48),
            ConvBlock(48, 80, 3, 2), ResidualBlock(80),
            ConvBlock(80, 128, 3, 2), ResidualBlock(128),
            ConvBlock(128, 160, 3, 2), ResidualBlock(160),
            nn.AdaptiveAvgPool2d((3, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(160 * 3 * 4, 256), nn.SiLU(inplace=True),
            nn.Dropout(0.15), nn.Linear(256, class_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(inputs))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
