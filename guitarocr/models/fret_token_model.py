from __future__ import annotations

import torch
from torch import nn


INPUT_WIDTH = 64
INPUT_HEIGHT = 32
CLASSES = ["blank", "X", *(str(value) for value in range(37))]


class FretTokenCNN(nn.Module):
    """Small event-conditioned classifier for one TAB string position."""

    def __init__(self, class_count: int = len(CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 80, 3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.SiLU(inplace=True),
            nn.Conv2d(80, 112, 3, padding=1, bias=False),
            nn.BatchNorm2d(112),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.12),
            nn.Linear(112, class_count),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image))
