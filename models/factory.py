"""Model factory for LCMS segmentation architectures."""
from __future__ import annotations

from typing import Any

from .hrnet_ocr import HRNetOCR
from .unet3plus import UNet3Plus
from .unetpp import UNetPP


def build_model_from_args(args: dict[str, Any], pretrained: bool = True):
    model_name = str(args.get("model", "unetpp")).lower()
    num_classes = int(args.get("num_classes", 6))
    in_channels = int(args.get("in_channels", 3))
    norm = args.get("norm", "group")
    activation = args.get("activation", "silu")
    deep_supervision = bool(args.get("deep_supervision", True))

    if model_name in {"unetpp", "unet++"}:
        attention_name = args.get("attention", "gate")
        context_name = args.get("context", "aspp")
        return UNetPP(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            norm=norm,
            activation=activation,
            attention=None if attention_name == "none" else attention_name,
            context=None if context_name == "none" else context_name,
            deep_supervision=deep_supervision,
            edge_head=not bool(args.get("no_edge_head", False)),
            dilated_encoder=bool(args.get("dilated_encoder", False)),
        )

    if model_name in {"unet3plus", "unet3+", "unet_3plus", "unet_3_plus"}:
        return UNet3Plus(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=int(args.get("unet3plus_base_channels", 32)),
            cat_channels=int(args.get("unet3plus_cat_channels", 32)),
            norm=norm,
            activation=activation,
            deep_supervision=deep_supervision,
            edge_head=not bool(args.get("no_edge_head", False)),
            dropout=float(args.get("unet3plus_dropout", 0.1)),
        )

    if model_name in {"hrnet_ocr", "hrnet+ocr", "hrnetocr"}:
        return HRNetOCR(
            in_channels=in_channels,
            num_classes=num_classes,
            width=int(args.get("hrnet_width", 32)),
            ocr_mid_channels=int(args.get("ocr_mid_channels", 512)),
            ocr_key_channels=int(args.get("ocr_key_channels", 256)),
            norm=args.get("hrnet_norm", "batch"),
            activation=args.get("hrnet_activation", "relu"),
            deep_supervision=deep_supervision,
        )

    raise ValueError(f"Unsupported model architecture: {model_name}")
