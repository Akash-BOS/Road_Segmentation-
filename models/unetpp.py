"""Research-grade UNet++ for LCMS road crack semantic segmentation."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aspp import make_context
from .decoder import UNetPPDecoder
from .edge_head import EdgeHead
from .encoder import build_encoder
from .utils import initialize_module


class UNetPP(nn.Module):
    """UNet++ with ResNet-50 encoder, context module, attention, and edge head.

    Args:
        in_channels: Number of LCMS input channels. RGB is 3, grayscale is 1.
        num_classes: Number of segmentation output classes. Use 1 for binary BCE, 2 for CE.
        backbone: Encoder name. Currently ``resnet50``.
        pretrained: Use ImageNet weights for the encoder when available.
        decoder_channels: Decoder widths from deep to shallow, e.g. ``(256,128,64,32)``.
        norm: ``batch``, ``group``, or ``instance``.
        activation: ``relu``, ``silu``, ``gelu``, or ``leaky_relu``.
        attention: ``cbam``, ``se``, ``gate``, or ``None``.
        context: ``aspp``, ``pyramid``, ``identity``, or ``None``.
        deep_supervision: Return ``aux0`` ... ``aux2`` predictions.
        edge_head: Return an auxiliary ``edge`` prediction.
        dropout: Dropout probability in decoder/context blocks.
        aspp_rates: Atrous rates for ASPP.
        dilated_encoder: Enable ResNet stride-to-dilation replacement explicitly.

    Input shape:
        ``[B, in_channels, H, W]``

    Output:
        ``{"out": logits}``, plus optional ``aux0``, ``aux1``, ``aux2``, and ``edge`` tensors,
        all upsampled to ``[B, C, H, W]`` except edge which is ``[B, 1, H, W]``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        norm: str = "group",
        activation: str = "silu",
        attention: str | None = "cbam",
        context: str | None = "aspp",
        deep_supervision: bool = True,
        edge_head: bool = True,
        dropout: float = 0.1,
        aspp_rates: Sequence[int] = (1, 6, 12, 18),
        dilated_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.has_edge_head = edge_head

        self.encoder = build_encoder(backbone, in_channels, pretrained, dilated_encoder)
        context_channels = decoder_channels[0]
        self.context = make_context(
            context,
            in_channels=self.encoder.out_channels[-1],
            out_channels=context_channels,
            norm=norm,
            activation=activation,
            dropout=dropout,
            rates=aspp_rates,
        )
        decoder_encoder_channels = [*self.encoder.out_channels[:-1], context_channels]
        self.decoder = UNetPPDecoder(
            encoder_channels=decoder_encoder_channels,
            decoder_channels=decoder_channels,
            norm=norm,
            activation=activation,
            attention=attention,
            dropout=dropout,
        )

        shallow_channels = decoder_channels[-1]
        self.final = nn.Conv2d(shallow_channels, num_classes, kernel_size=1)
        if deep_supervision:
            self.aux_heads = nn.ModuleDict({
                "aux0": nn.Conv2d(shallow_channels, num_classes, 1),
                "aux1": nn.Conv2d(shallow_channels, num_classes, 1),
                "aux2": nn.Conv2d(shallow_channels, num_classes, 1),
            })
        else:
            self.aux_heads = nn.ModuleDict()
        self.edge = EdgeHead(shallow_channels, norm, activation) if edge_head else None

        initialize_module(self.context)
        initialize_module(self.decoder)
        initialize_module(self.final)
        initialize_module(self.aux_heads)
        if self.edge is not None:
            initialize_module(self.edge)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[2:]
        features = self.encoder(x)
        features["x4"] = self.context(features["x4"])
        decoded = self.decoder(features)

        out = self.final(decoded["x0_4"])
        outputs: Dict[str, torch.Tensor] = {
            "out": F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        }
        if self.deep_supervision:
            aux_map = {"aux0": "x0_1", "aux1": "x0_2", "aux2": "x0_3"}
            for name, feature_name in aux_map.items():
                aux = self.aux_heads[name](decoded[feature_name])
                outputs[name] = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
        if self.edge is not None:
            edge = self.edge(decoded["x0_4"])
            outputs["edge"] = F.interpolate(edge, size=input_size, mode="bilinear", align_corners=False)
        return outputs
