"""Backbone encoders for LCMS UNet++."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn


class ResNet50Encoder(nn.Module):
    """Torchvision ResNet-50 feature encoder with LCMS channel adaptation.

    The hierarchy is:
        x0: conv1/bn/relu, stride 2, 64 channels
        x1: maxpool + layer1, stride 4, 256 channels
        x2: layer2, stride 8, 512 channels
        x3: layer3, stride 16, 1024 channels
        x4: layer4, stride 32 unless dilation is enabled, 2048 channels

    Args:
        in_channels: LCMS input channel count.
        pretrained: Load ImageNet weights when available.
        dilated: Replace layer3/layer4 strides with dilation only when explicitly enabled.
    """

    out_channels: List[int] = [64, 256, 512, 1024, 2048]

    def __init__(self, in_channels: int = 3, pretrained: bool = True, dilated: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        replace_stride = [False, True, True] if dilated else [False, False, False]
        backbone = resnet50(weights=weights, replace_stride_with_dilation=replace_stride)
        if in_channels != 3:
            backbone.conv1 = self._adapt_first_conv(backbone.conv1, in_channels)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @staticmethod
    def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
        """Adapt RGB conv1 weights to arbitrary channel counts by averaging and repeating."""
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            rgb_weight = conv.weight.data
            mean_weight = rgb_weight.mean(dim=1, keepdim=True)
            adapted = mean_weight.repeat(1, in_channels, 1, 1)
            adapted = adapted * (3.0 / float(in_channels))
            new_conv.weight.copy_(adapted)
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias.data)
        return new_conv

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return feature maps ``x0`` ... ``x4``."""
        features: Dict[str, torch.Tensor] = OrderedDict()
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features["x0"] = x0
        features["x1"] = x1
        features["x2"] = x2
        features["x3"] = x3
        features["x4"] = x4
        return features


def build_encoder(backbone: str, in_channels: int, pretrained: bool, dilated: bool) -> ResNet50Encoder:
    """Build a supported encoder by name."""
    if backbone.lower() != "resnet50":
        raise ValueError(f"Unsupported backbone: {backbone}")
    return ResNet50Encoder(in_channels=in_channels, pretrained=pretrained, dilated=dilated)
