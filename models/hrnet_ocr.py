"""HRNet + OCR semantic segmentation model for LCMS road anomalies."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation, get_norm, initialize_module


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str = "batch", activation: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = get_norm(norm, out_channels)
        self.act = get_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = get_norm(norm, out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                get_norm(norm, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


def _make_layer(in_channels: int, out_channels: int, blocks: int, norm: str, activation: str) -> nn.Sequential:
    layers = [BasicBlock(in_channels, out_channels, norm=norm, activation=activation)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels, norm=norm, activation=activation))
    return nn.Sequential(*layers)


class HighResolutionModule(nn.Module):
    """Multi-branch HRNet block with repeated cross-resolution fusion."""

    def __init__(self, channels: Sequence[int], blocks: int, norm: str, activation: str) -> None:
        super().__init__()
        self.channels = list(channels)
        self.branches = nn.ModuleList([
            _make_layer(ch, ch, blocks, norm, activation)
            for ch in self.channels
        ])
        self.fuse_layers = nn.ModuleList()
        for target_idx, target_ch in enumerate(self.channels):
            fuse_row = nn.ModuleList()
            for source_idx, source_ch in enumerate(self.channels):
                if source_idx == target_idx:
                    fuse_row.append(nn.Identity())
                elif source_idx > target_idx:
                    fuse_row.append(nn.Sequential(
                        nn.Conv2d(source_ch, target_ch, 1, bias=False),
                        get_norm(norm, target_ch),
                    ))
                else:
                    ops: list[nn.Module] = []
                    current_ch = source_ch
                    for step in range(target_idx - source_idx):
                        out_ch = target_ch if step == target_idx - source_idx - 1 else current_ch
                        ops.extend([
                            nn.Conv2d(current_ch, out_ch, 3, stride=2, padding=1, bias=False),
                            get_norm(norm, out_ch),
                        ])
                        if step != target_idx - source_idx - 1:
                            ops.append(get_activation(activation))
                        current_ch = out_ch
                    fuse_row.append(nn.Sequential(*ops))
                # type: ignore[arg-type]
            self.fuse_layers.append(fuse_row)
        self.act = get_activation(activation)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        features = [branch(x) for branch, x in zip(self.branches, inputs)]
        fused: list[torch.Tensor] = []
        for target_idx, target in enumerate(features):
            target_size = target.shape[2:]
            y = self.fuse_layers[target_idx][target_idx](features[target_idx])
            for source_idx, source in enumerate(features):
                if source_idx == target_idx:
                    continue
                transformed = self.fuse_layers[target_idx][source_idx](source)
                if source_idx > target_idx:
                    transformed = F.interpolate(transformed, size=target_size, mode="bilinear", align_corners=False)
                y = y + transformed
            fused.append(self.act(y))
        return fused


class SpatialGatherModule(nn.Module):
    """Aggregate per-class object context vectors from pixel features."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        classes = probs.shape[1]
        probs = probs.view(batch, classes, -1)
        probs = F.softmax(self.scale * probs, dim=2)
        features = features.view(batch, channels, -1).permute(0, 2, 1)
        context = torch.matmul(probs, features).permute(0, 2, 1).unsqueeze(3)
        return context


class ObjectAttentionBlock(nn.Module):
    """Pixel-object attention used by OCRNet."""

    def __init__(self, in_channels: int, key_channels: int, norm: str, activation: str) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            get_norm(norm, key_channels),
            get_activation(activation),
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            get_norm(norm, key_channels),
            get_activation(activation),
        )
        self.value = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            get_norm(norm, in_channels),
            get_activation(activation),
        )

    def forward(self, pixels: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = pixels.shape
        query = self.query(pixels).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(objects).squeeze(3)
        value = self.value(objects).squeeze(3).permute(0, 2, 1)
        sim = torch.matmul(query, key) * (key.shape[1] ** -0.5)
        sim = F.softmax(sim, dim=-1)
        context = torch.matmul(sim, value).permute(0, 2, 1).contiguous()
        return context.view(batch, -1, height, width)


class OCRHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, mid_channels: int, key_channels: int, norm: str, activation: str) -> None:
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            get_norm(norm, mid_channels),
            get_activation(activation),
        )
        self.aux_head = nn.Conv2d(in_channels, num_classes, 1)
        self.gather = SpatialGatherModule()
        self.object_context = ObjectAttentionBlock(mid_channels, key_channels, norm, activation)
        self.cls_head = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 1, bias=False),
            get_norm(norm, mid_channels),
            get_activation(activation),
            nn.Dropout2d(0.05),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aux = self.aux_head(features)
        feats = self.conv3x3(features)
        context = self.gather(feats, aux)
        context = self.object_context(feats, context)
        out = self.cls_head(torch.cat([feats, context], dim=1))
        return out, aux


class HRNetOCR(nn.Module):
    """Compact HRNetV2-style backbone with OCR head.

    The network keeps a 1/4-resolution branch throughout the backbone, making it
    better suited than heavy encoder-decoder downsampling for thin cracks.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 32,
        ocr_mid_channels: int = 512,
        ocr_key_channels: int = 256,
        norm: str = "batch",
        activation: str = "relu",
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        channels = [width, width * 2, width * 4, width * 8]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            get_norm(norm, 64),
            get_activation(activation),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            get_norm(norm, 64),
            get_activation(activation),
        )
        self.layer1 = _make_layer(64, channels[0], 4, norm, activation)
        self.transition1 = nn.ModuleList([
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False),
                get_norm(norm, channels[1]),
                get_activation(activation),
            ),
        ])
        self.stage2 = nn.Sequential(
            HighResolutionModule(channels[:2], 2, norm, activation),
        )
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False),
                get_norm(norm, channels[2]),
                get_activation(activation),
            ),
        ])
        self.stage3 = nn.Sequential(
            HighResolutionModule(channels[:3], 2, norm, activation),
            HighResolutionModule(channels[:3], 2, norm, activation),
        )
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False),
                get_norm(norm, channels[3]),
                get_activation(activation),
            ),
        ])
        self.stage4 = nn.Sequential(
            HighResolutionModule(channels, 2, norm, activation),
            HighResolutionModule(channels, 2, norm, activation),
        )
        self.ocr = OCRHead(sum(channels), num_classes, ocr_mid_channels, ocr_key_channels, norm, activation)
        initialize_module(self)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[2:]
        x = self.layer1(self.stem(x))
        branches = [self.transition1[0](x), self.transition1[1](x)]
        branches = self.stage2(branches)
        branches = [self.transition2[0](branches[0]), self.transition2[1](branches[1]), self.transition2[2](branches[1])]
        branches = self.stage3(branches)
        branches = [
            self.transition3[0](branches[0]),
            self.transition3[1](branches[1]),
            self.transition3[2](branches[2]),
            self.transition3[3](branches[2]),
        ]
        branches = self.stage4(branches)

        high_size = branches[0].shape[2:]
        features = [branches[0]]
        for branch in branches[1:]:
            features.append(F.interpolate(branch, size=high_size, mode="bilinear", align_corners=False))
        features_cat = torch.cat(features, dim=1)
        logits, aux = self.ocr(features_cat)
        outputs: Dict[str, torch.Tensor] = {
            "out": F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        }
        if self.deep_supervision:
            outputs["aux0"] = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
        return outputs
