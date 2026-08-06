"""Auxiliary edge prediction head."""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvNormAct


class EdgeHead(nn.Module):
    """Predict crack boundaries from shallow decoder features.

    Input shape: ``[B, C, H, W]``. Output shape: ``[B, 1, H, W]``.
    """

    def __init__(self, in_channels: int, norm: str = "group", activation: str = "silu") -> None:
        super().__init__()
        hidden = max(in_channels, 16)
        self.net = nn.Sequential(
            ConvNormAct(in_channels, hidden, 3, 1, norm, activation),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
