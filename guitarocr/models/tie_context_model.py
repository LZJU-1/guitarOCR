from __future__ import annotations

import torch
from torch import nn

from guitarocr.models.rhythm_context_model import INPUT_HEIGHT, INPUT_WIDTH, RhythmContextCNN


Y_BINS = 48
COUNT_CLASSES = list(range(7))


class TieContextCNN(nn.Module):
    """Event-centred tie-in classifier with count and target-y relationship heads."""

    def __init__(self):
        super().__init__()
        rhythm = RhythmContextCNN()
        self.backbone = rhythm.backbone
        self.context = rhythm.context
        self.presence_head = nn.Linear(256, 2)
        self.count_head = nn.Linear(256, len(COUNT_CLASSES))
        self.note_count_head = nn.Linear(256, len(COUNT_CLASSES))
        self.y_head = nn.Linear(256, Y_BINS)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.context(self.backbone(inputs))
        return (
            self.presence_head(features),
            self.count_head(features),
            self.note_count_head(features),
            self.y_head(features),
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "COUNT_CLASSES",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "TieContextCNN",
    "Y_BINS",
    "parameter_count",
]
