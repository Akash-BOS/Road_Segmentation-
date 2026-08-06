"""Evaluate LCMS UNet++ checkpoints."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lcms_unetpp.config import apply_config_defaults, load_yaml_config
from lcms_unetpp.data import COLOR_MAP, EvalTransform, LCMSCrackDataset
from lcms_unetpp.engine import evaluate
from lcms_unetpp.logging_utils import log_environment, setup_logging
from lcms_unetpp.losses import CrackSegmentationLoss
from lcms_unetpp.models import build_model_from_args


def _main_logits(outputs):
    return outputs["out"] if isinstance(outputs, dict) else outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test LCMS semantic segmentation")
    parser.add_argument("--config", default="", help="Optional YAML config. CLI arguments override config defaults.")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="TEST", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=419, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--in-channels", default=3, type=int)
    parser.add_argument("--num-classes", default=6, type=int)
    parser.add_argument("--mask-mode", default="color", choices=["color", "index", "binary"])
    parser.add_argument("--model", default="unetpp", choices=["unetpp", "unet3plus", "hrnet_ocr"])
    parser.add_argument("--norm", default="group", choices=["batch", "group", "instance"])
    parser.add_argument("--activation", default="silu", choices=["relu", "silu", "gelu", "leaky_relu"])
    parser.add_argument("--attention", default="gate", choices=["cbam", "se", "gate", "none"])
    parser.add_argument("--context", default="aspp", choices=["aspp", "pyramid", "identity", "none"])
    parser.add_argument("--deep-supervision", action="store_true", default=True)
    parser.add_argument("--no-edge-head", action="store_true")
    parser.add_argument("--dilated-encoder", action="store_true")
    parser.add_argument("--hrnet-width", default=32, type=int)
    parser.add_argument("--hrnet-norm", default="batch", choices=["batch", "group", "instance"])
    parser.add_argument("--hrnet-activation", default="relu", choices=["relu", "silu", "gelu", "leaky_relu"])
    parser.add_argument("--ocr-mid-channels", default=512, type=int)
    parser.add_argument("--ocr-key-channels", default=256, type=int)
    parser.add_argument("--unet3plus-base-channels", default=32, type=int)
    parser.add_argument("--unet3plus-cat-channels", default=32, type=int)
    parser.add_argument("--unet3plus-dropout", default=0.1, type=float)
    parser.add_argument("--output-json", default="test_metrics.json")
    parser.add_argument("--save-predictions", action="store_true", help="Save predicted masks and overlays as PNG files")
    parser.add_argument("--prediction-dir", default="predictions", help="Folder for saved prediction images")
    parser.add_argument("--overlay-alpha", default=0.45, type=float, help="Prediction overlay opacity")
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip test-time augmentation")
    parser.add_argument("--log-file", default="", help="Append console output and tracebacks to this file")
    args = parser.parse_args()
    if args.config:
        args = apply_config_defaults(parser, args, load_yaml_config(args.config))
    return args


def parse_loss_class_weights(value: str, num_classes: int) -> list[float] | None:
    if not value or not value.strip():
        return None
    weights = [1.0] * num_classes
    for item in value.split(","):
        class_id, weight = item.split(":", 1)
        class_idx = int(class_id.strip())
        if 0 <= class_idx < num_classes:
            weights[class_idx] = float(weight.strip())
    return weights


def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a class-id mask to an RGB color mask."""
    palette = np.zeros((max(num_classes, 1), 3), dtype=np.uint8)
    for color, class_id in COLOR_MAP.items():
        if class_id < len(palette):
            palette[class_id] = color
    return palette[np.clip(mask, 0, len(palette) - 1)]


def prepare_overlay_image(image_path: Path, height: int, width: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(image_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if image.dtype != np.uint8:
        max_value = 65535.0 if image.max() > 255 else 255.0
        image = np.clip(image.astype(np.float32) / max_value * 255.0, 0, 255).astype(np.uint8)
    return image


@torch.no_grad()
def save_predictions(
    model: torch.nn.Module,
    dataset: LCMSCrackDataset,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_classes: int,
    height: int,
    width: int,
    overlay_alpha: float,
    use_tta: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    model.eval()
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    for index, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        logits = _main_logits(model(images))
        if use_tta:
            flip_logits = _main_logits(model(torch.flip(images, dims=[3])))
            logits = (logits + torch.flip(flip_logits, dims=[3])) / 2.0
        if logits.shape[1] == 1:
            pred = (torch.sigmoid(logits[:, 0]) > 0.5).long()
            color_classes = 2
        else:
            pred = logits.argmax(1)
            color_classes = num_classes
        pred_mask = pred[0].cpu().numpy().astype(np.uint8)

        stem = dataset.image_paths[index].stem
        pred_rgb = colorize_mask(pred_mask, color_classes)
        image_rgb = prepare_overlay_image(dataset.image_paths[index], height, width)
        overlay_rgb = cv2.addWeighted(image_rgb, 1.0 - overlay_alpha, pred_rgb, overlay_alpha, 0)

        cv2.imwrite(str(masks_dir / f"{stem}_pred.png"), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlays_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        if logger is not None and index % 25 == 0:
            logger.info("prediction saved index=%d/%d stem=%s", index, len(dataset), stem)


def main() -> None:
    args = parse_args()
    if not args.data_path:
        raise ValueError("Provide --data-path or data_path in --config")
    if not args.checkpoint:
        raise ValueError("Provide --checkpoint or checkpoint in --config")
    log_path = Path(args.log_file) if args.log_file else Path(args.output_json).with_suffix(".log")
    logger = setup_logging(log_path, "lcms_unetpp.test")
    log_environment(logger)
    logger.info("command args: %s", json.dumps(vars(args), indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    logger.info("loading checkpoint=%s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})
    model_in_channels = model_args.get("in_channels", args.in_channels)
    model_num_classes = model_args.get("num_classes", args.num_classes)
    model_args = {**vars(args), **model_args, "in_channels": model_in_channels, "num_classes": model_num_classes}
    model = build_model_from_args(model_args, pretrained=False).to(device)
    logger.info(
        "model architecture=%s in_channels=%d num_classes=%d deep_supervision=%s",
        model_args.get("model", "unetpp"),
        model_in_channels,
        model_num_classes,
        model_args.get("deep_supervision", True),
    )
    state = checkpoint.get("ema_model") or checkpoint["model"]
    model.load_state_dict(state)

    ds = LCMSCrackDataset(args.data_path, args.split, EvalTransform(args.height, args.width), mask_mode=args.mask_mode)
    logger.info("dataset split=%s samples=%d", args.split, len(ds))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=LCMSCrackDataset.collate_fn)
    loss_class_weights = parse_loss_class_weights(model_args.get("loss_class_weights", ""), model_num_classes)
    criterion = CrackSegmentationLoss(
        class_weights=loss_class_weights,
        dice_weight=model_args.get("dice_weight", 0.5),
        region_loss=model_args.get("region_loss", "dice"),
        tversky_alpha=model_args.get("tversky_alpha", 0.3),
        tversky_beta=model_args.get("tversky_beta", 0.7),
        tversky_gamma=model_args.get("tversky_gamma", 1.0),
        focal_weight=model_args.get("focal_weight", 0.25),
        boundary_weight=model_args.get("boundary_weight", 0.1),
        edge_weight=model_args.get("edge_weight", 0.2),
        aux_weight=model_args.get("aux_weight", 0.4),
    ).to(device)
    metric_classes = 2 if args.mask_mode == "binary" or model_num_classes == 1 else model_num_classes
    logger.info("evaluation start")
    metrics = evaluate(
        model,
        loader,
        criterion,
        device,
        metric_classes=metric_classes,
        use_tta=args.tta,
        progress_callback=lambda step, loss: logger.info("eval step=%d/%d loss=%.6f", step, len(loader), loss)
        if step % 25 == 0
        else None,
    )
    logger.info("evaluation done")
    Path(args.output_json).write_text(json.dumps(metrics, indent=2))
    logger.info("wrote metrics=%s", args.output_json)
    if args.save_predictions:
        logger.info("saving predictions to %s", args.prediction_dir)
        save_predictions(
            model,
            ds,
            loader,
            device,
            Path(args.prediction_dir),
            metric_classes,
            args.height,
            args.width,
            args.overlay_alpha,
            args.tta,
            logger,
        )
        logger.info("saved predictions to %s", args.prediction_dir)
        print(f"Saved predictions to {args.prediction_dir}")
    print(metrics)


if __name__ == "__main__":
    main()



