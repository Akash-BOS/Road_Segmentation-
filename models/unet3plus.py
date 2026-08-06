"""UNet 3+ style full-scale skip segmentation model.

This is a PyTorch implementation adapted to the LCMS training pipeline. It keeps
the key UNet 3+ idea: each decoder stage aggregates encoder/decoder features
from every scale after resizing them to the current target scale.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDoubleConv
from .edge_head import EdgeHead
from .utils import conv1x1, initialize_module


class UNet3PlusAggregation(nn.Module):
    """Full-scale feature aggregation block used by UNet 3+ decoders."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        cat_channels: int,
        norm: str,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList([conv1x1(channels, cat_channels) for channels in in_channels])
        self.fuse = ResidualDoubleConv(
            cat_channels * len(in_channels),
            out_channels,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, features: list[torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        resized = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            resized.append(feature)
        return self.fuse(torch.cat(resized, dim=1))


class UNet3Plus(nn.Module):
    """UNet 3+ with full-scale skip connections and optional deep supervision.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of output classes.
        base_channels: Width of the first encoder stage.
        cat_channels: Width used for each projected full-scale skip branch.
        norm: ``batch``, ``group``, ``instance``, or ``none``.
        activation: Activation name.
        deep_supervision: Return auxiliary logits from intermediate decoders.
        edge_head: Return an auxiliary foreground boundary head.
        dropout: Dropout probability inside convolution blocks.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        cat_channels: int = 32,
        norm: str = "group",
        activation: str = "silu",
        deep_supervision: bool = True,
        edge_head: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.has_edge_head = edge_head

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.enc1 = ResidualDoubleConv(in_channels, channels[0], norm=norm, activation=activation, dropout=dropout)
        self.enc2 = ResidualDoubleConv(channels[0], channels[1], norm=norm, activation=activation, dropout=dropout)
        self.enc3 = ResidualDoubleConv(channels[1], channels[2], norm=norm, activation=activation, dropout=dropout)
        self.enc4 = ResidualDoubleConv(channels[2], channels[3], norm=norm, activation=activation, dropout=dropout)
        self.enc5 = ResidualDoubleConv(channels[3], channels[4], norm=norm, activation=activation, dropout=dropout)
        self.pool = nn.MaxPool2d(2, 2)

        decoder_channels = cat_channels * 5
        self.dec4 = UNet3PlusAggregation(channels, decoder_channels, cat_channels, norm, activation, dropout)
        self.dec3 = UNet3PlusAggregation(
            [channels[0], channels[1], channels[2], decoder_channels, channels[4]],
            decoder_channels,
            cat_channels,
            norm,
            activation,
            dropout,
        )
        self.dec2 = UNet3PlusAggregation(
            [channels[0], channels[1], decoder_channels, decoder_channels, channels[4]],
            decoder_channels,
            cat_channels,
            norm,
            activation,
            dropout,
        )
        self.dec1 = UNet3PlusAggregation(
            [channels[0], decoder_channels, decoder_channels, decoder_channels, channels[4]],
            decoder_channels,
            cat_channels,
            norm,
            activation,
            dropout,
        )

        self.final = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        if deep_supervision:
            self.aux_heads = nn.ModuleDict(
                {
                    "aux0": nn.Conv2d(decoder_channels, num_classes, 1),
                    "aux1": nn.Conv2d(decoder_channels, num_classes, 1),
                    "aux2": nn.Conv2d(decoder_channels, num_classes, 1),
                }
            )
        else:
            self.aux_heads = nn.ModuleDict()
        self.edge = EdgeHead(decoder_channels, norm, activation) if edge_head else None

        initialize_module(self)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.dec4([e1, e2, e3, e4, e5], e4.shape[2:])
        d3 = self.dec3([e1, e2, e3, d4, e5], e3.shape[2:])
        d2 = self.dec2([e1, e2, d3, d4, e5], e2.shape[2:])
        d1 = self.dec1([e1, d2, d3, d4, e5], e1.shape[2:])

        outputs: Dict[str, torch.Tensor] = {
            "out": F.interpolate(self.final(d1), size=input_size, mode="bilinear", align_corners=False)
        }
        if self.deep_supervision:
            aux_features = {"aux0": d2, "aux1": d3, "aux2": d4}
            for name, feature in aux_features.items():
                outputs[name] = F.interpolate(self.aux_heads[name](feature), size=input_size, mode="bilinear", align_corners=False)
        if self.edge is not None:
            outputs["edge"] = F.interpolate(self.edge(d1), size=input_size, mode="bilinear", align_corners=False)
        return outputs
