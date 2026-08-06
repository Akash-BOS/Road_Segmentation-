"""Context modules for LCMS UNet++."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct
from .utils import conv1x1, get_activation, get_norm


class ASPP(nn.Module):
    """Lightweight DeepLabV3-style ASPP with channel compression.

    Args:
        in_channels: Bottleneck channels from the encoder.
        out_channels: Compressed context width, commonly 256.
        rates: Atrous rates for 3x3 branches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: Sequence[int] = (1, 6, 12, 18),
        norm: str = "group",
        activation: str = "silu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        branches = []
        for rate in rates:
            if rate == 1:
                branches.append(ConvNormAct(in_channels, out_channels, 1, 1, norm, activation))
            else:
                branches.append(ConvNormAct(in_channels, out_channels, 3, rate, norm, activation))
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_channels, out_channels),
            get_norm(norm, out_channels),
            get_activation(activation),
        )
        self.project = nn.Sequential(
            conv1x1(out_channels * (len(rates) + 1), out_channels),
            get_norm(norm, out_channels),
            get_activation(activation),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        outputs = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        outputs.append(F.interpolate(pooled, size=size, mode="bilinear", align_corners=False))
        return self.project(torch.cat(outputs, dim=1))


class PyramidPooling(nn.Module):
    """Small PSP-style context module for optional global context."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        bins: Sequence[int] = (1, 2, 3, 6),
        norm: str = "group",
        activation: str = "silu",
    ) -> None:
        super().__init__()
        branch_channels = max(out_channels // len(bins), 1)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(bin_size), ConvNormAct(in_channels, branch_channels, 1, 1, norm, activation))
            for bin_size in bins
        ])
        self.project = ConvNormAct(in_channels + branch_channels * len(bins), out_channels, 1, 1, norm, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        pooled = [F.interpolate(stage(x), size=size, mode="bilinear", align_corners=False) for stage in self.stages]
        return self.project(torch.cat([x, *pooled], dim=1))


def make_context(
    kind: str | None,
    in_channels: int,
    out_channels: int,
    norm: str,
    activation: str,
    dropout: float,
    rates: Sequence[int] = (1, 6, 12, 18),
) -> nn.Module:
    """Create a context module and always return ``out_channels`` features."""
    if kind is None:
        return ConvNormAct(in_channels, out_channels, 1, 1, norm, activation)
    kind = kind.lower()
    if kind == "aspp":
        return ASPP(in_channels, out_channels, rates, norm, activation, dropout)
    if kind in ("pyramid", "ppm", "psp"):
        return PyramidPooling(in_channels, out_channels, norm=norm, activation=activation)
    if kind in ("identity", "none"):
        return ConvNormAct(in_channels, out_channels, 1, 1, norm, activation)
    raise ValueError(f"Unsupported context module: {kind}")
