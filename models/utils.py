"""Initialization and layer factory utilities for LCMS UNet++."""
from __future__ import annotations

from typing import Callable

import torch.nn as nn


def get_norm(norm: str, channels: int, groups: int = 8) -> nn.Module:
    """Create a 2D normalization layer.

    Args:
        norm: One of ``batch``, ``group``, ``instance``, or ``none``.
        channels: Number of feature channels.
        groups: Preferred group count for GroupNorm.
    """
    norm = norm.lower()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        valid_groups = min(groups, channels)
        while channels % valid_groups != 0 and valid_groups > 1:
            valid_groups -= 1
        return nn.GroupNorm(valid_groups, channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


def get_activation(activation: str) -> nn.Module:
    """Create an activation layer compatible with mixed precision."""
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU(inplace=False)
    if activation == "silu":
        return nn.SiLU(inplace=False)
    if activation == "gelu":
        return nn.GELU()
    if activation in ("leaky_relu", "leaky"):
        return nn.LeakyReLU(0.1, inplace=False)
    if activation in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {activation}")


def initialize_module(module: nn.Module, mode: str = "kaiming") -> None:
    """Initialize newly created convolution and linear layers.

    Args:
        module: Module tree to initialize in-place.
        mode: ``kaiming`` or ``xavier``.
    """
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            if mode == "kaiming":
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif mode == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            else:
                raise ValueError(f"Unsupported init mode: {mode}")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            if getattr(layer, "weight", None) is not None:
                nn.init.ones_(layer.weight)
            if getattr(layer, "bias", None) is not None:
                nn.init.zeros_(layer.bias)


def conv1x1(in_channels: int, out_channels: int, bias: bool = False) -> nn.Conv2d:
    """Create a 1x1 convolution."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
