"""Reusable convolution blocks for LCMS UNet++."""
from __future__ import annotations

import torch
import torch.nn as nn

from .utils import get_activation, get_norm


class ConvNormAct(nn.Sequential):
    """Conv2d -> normalization -> activation.

    Input and output tensors use shape ``[B, C, H, W]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        norm: str = "group",
        activation: str = "silu",
    ) -> None:
        padding = dilation if kernel_size == 3 else 0
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            get_norm(norm, out_channels),
            get_activation(activation),
        )


class ResidualDoubleConv(nn.Module):
    """Residual two-convolution block used by the nested decoder.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        norm: Normalization type.
        activation: Activation type.
        dropout: Spatial dropout probability.
        dilation: Atrous dilation for both 3x3 convolutions.

    Shapes:
        Input: ``[B, in_channels, H, W]``
        Output: ``[B, out_channels, H, W]``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "group",
        activation: str = "silu",
        dropout: float = 0.0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, dilation, norm, activation)
        self.conv2 = ConvNormAct(out_channels, out_channels, 3, dilation, norm, activation)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.out_act(out + identity)
