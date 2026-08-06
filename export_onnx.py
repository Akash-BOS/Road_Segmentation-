"""Export LCMS segmentation checkpoint to ONNX."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lcms_unetpp.logging_utils import log_environment, setup_logging
from lcms_unetpp.models import build_model_from_args


class OutOnlyWrapper(nn.Module):
    """Return only the segmentation logits for ONNX export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LCMS segmentation model to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=419, type=int)
    parser.add_argument("--in-channels", default=3, type=int)
    parser.add_argument("--num-classes", default=6, type=int)
    parser.add_argument("--opset", default=17, type=int)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--dynamo-export",
        action="store_true",
        help="Use PyTorch's newer dynamo ONNX exporter. Requires onnxscript to be installed.",
    )
    parser.add_argument("--log-file", default="", help="Append console output and tracebacks to this file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file) if args.log_file else Path(args.onnx_path).with_suffix(".log")
    logger = setup_logging(log_path, "lcms_unetpp.export_onnx")
    log_environment(logger)
    logger.info("command args: %s", json.dumps(vars(args), indent=2))
    logger.info("loading checkpoint=%s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint.get("args", {})
    model_args = {**vars(args), **model_args}
    model = build_model_from_args(model_args, pretrained=False)
    state = checkpoint.get("ema_model") or checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    logger.info("model loaded checkpoint_epoch=%s", checkpoint.get("epoch"))
    wrapper = OutOnlyWrapper(model)
    dummy = torch.randn(1, args.in_channels, args.height, args.width)
    logger.info(
        "export start dummy_shape=%s dynamic=%s opset=%d dynamo_export=%s",
        tuple(dummy.shape),
        args.dynamic,
        args.opset,
        args.dynamo_export,
    )
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {0: "batch", 2: "height", 3: "width"}, "out": {0: "batch", 2: "height", 3: "width"}}
    Path(args.onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.onnx_path,
        input_names=["input"],
        output_names=["out"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
        dynamo=args.dynamo_export,
    )
    logger.info("export done onnx_path=%s", args.onnx_path)
    print(f"ONNX exported to {args.onnx_path}")


if __name__ == "__main__":
    main()

