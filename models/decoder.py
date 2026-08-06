"""Nested UNet++ decoder."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionGate, make_attention
from .blocks import ResidualDoubleConv
from .utils import conv1x1


class UNetPPDecoder(nn.Module):
    """Memory-conscious UNet++ nested dense decoder.

    Args:
        encoder_channels: Channels for ``x0`` ... ``x4`` after encoder/context projection.
        decoder_channels: Feature widths from deep to shallow, e.g. ``(256, 128, 64, 32)``.
        norm: Normalization kind.
        activation: Activation kind.
        attention: ``None``, ``se``, ``cbam``, or ``gate``.
        dropout: Decoder dropout.

    Shapes:
        Input features: dict of ``x0`` ... ``x4``.
        Output features: dict containing ``x0_1`` ... ``x0_4``.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        norm: str = "group",
        activation: str = "silu",
        attention: str | None = "cbam",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four values: deep to shallow")

        c0 = decoder_channels[3]
        c1 = decoder_channels[2]
        c2 = decoder_channels[1]
        c3 = decoder_channels[0]
        c4 = decoder_channels[0]
        self.channels = [c0, c1, c2, c3, c4]

        self.proj0 = conv1x1(encoder_channels[0], c0)
        self.proj1 = conv1x1(encoder_channels[1], c1)
        self.proj2 = conv1x1(encoder_channels[2], c2)
        self.proj3 = conv1x1(encoder_channels[3], c3)

        use_gates = attention is not None and attention.lower() in ("gate", "attention_gate")
        self.gates = nn.ModuleDict()
        if use_gates:
            self.gates["x0_0"] = AttentionGate(c0, c1, max(c0 // 2, 8))
            self.gates["x1_0"] = AttentionGate(c1, c2, max(c1 // 2, 8))
            self.gates["x2_0"] = AttentionGate(c2, c3, max(c2 // 2, 8))
            self.gates["x3_0"] = AttentionGate(c3, c4, max(c3 // 2, 8))

        block_attention = None if use_gates else attention
        self.conv0_1 = self._block(c0 + c1, c0, norm, activation, block_attention, dropout)
        self.conv1_1 = self._block(c1 + c2, c1, norm, activation, block_attention, dropout)
        self.conv2_1 = self._block(c2 + c3, c2, norm, activation, block_attention, dropout)
        self.conv3_1 = self._block(c3 + c4, c3, norm, activation, block_attention, dropout)

        self.conv0_2 = self._block(c0 * 2 + c1, c0, norm, activation, block_attention, dropout)
        self.conv1_2 = self._block(c1 * 2 + c2, c1, norm, activation, block_attention, dropout)
        self.conv2_2 = self._block(c2 * 2 + c3, c2, norm, activation, block_attention, dropout)

        self.conv0_3 = self._block(c0 * 3 + c1, c0, norm, activation, block_attention, dropout)
        self.conv1_3 = self._block(c1 * 3 + c2, c1, norm, activation, block_attention, dropout)

        self.conv0_4 = self._block(c0 * 4 + c1, c0, norm, activation, block_attention, dropout)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        norm: str,
        activation: str,
        attention: str | None,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            ResidualDoubleConv(in_channels, out_channels, norm=norm, activation=activation, dropout=dropout),
            make_attention(attention, out_channels),
        )

    @staticmethod
    def _up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def _gate(self, name: str, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if name in self.gates:
            return self.gates[name](skip, gate)
        return skip

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x0_0 = self.proj0(features["x0"])
        x1_0 = self.proj1(features["x1"])
        x2_0 = self.proj2(features["x2"])
        x3_0 = self.proj3(features["x3"])
        x4_0 = features["x4"]

        x0_1 = self.conv0_1(torch.cat([self._gate("x0_0", x0_0, x1_0), self._up(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([self._gate("x1_0", x1_0, x2_0), self._up(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([self._gate("x2_0", x2_0, x3_0), self._up(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([self._gate("x3_0", x3_0, x4_0), self._up(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))

        return {"x0_1": x0_1, "x0_2": x0_2, "x0_3": x0_3, "x0_4": x0_4}
