"""Attention modules for thin-structure segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation, get_norm


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention for ``[B, C, H, W]`` tensors."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class CBAMBlock(nn.Module):
    """CBAM channel + spatial attention for decoder refinement."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_pool = self.mlp(F.adaptive_max_pool2d(x, 1))
        x = x * torch.sigmoid(avg + max_pool)
        spatial = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(spatial))


class AttentionGate(nn.Module):
    """Additive attention gate for skip features.

    Args:
        skip_channels: Channels in the skip tensor.
        gate_channels: Channels in the gating tensor.
        inter_channels: Internal attention width.

    Shapes:
        skip: ``[B, skip_channels, H, W]``
        gate: ``[B, gate_channels, h, w]``; it is bilinearly resized to skip size.
    """

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
        self.psi = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(inter_channels, 1, 1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * alpha


def make_attention(kind: str | None, channels: int) -> nn.Module:
    """Build an attention module by name."""
    if kind is None:
        return nn.Identity()
    kind = kind.lower()
    if kind in ("none", "identity"):
        return nn.Identity()
    if kind == "se":
        return SEBlock(channels)
    if kind == "cbam":
        return CBAMBlock(channels)
    if kind in ("gate", "attention_gate"):
        return nn.Identity()
    raise ValueError(f"Unsupported attention: {kind}")
