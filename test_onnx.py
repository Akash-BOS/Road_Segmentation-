"""Evaluate an exported ONNX segmentation model on LCMS splits."""
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

from lcms_unetpp.data import COLOR_MAP, EvalTransform, LCMSCrackDataset
from lcms_unetpp.logging_utils import log_environment, setup_logging
from lcms_unetpp.metrics import SegmentationMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test LCMS ONNX segmentation model")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--split", default="TEST", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=419, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--num-classes", default=6, type=int)
    parser.add_argument("--mask-mode", default="color", choices=["color", "index", "binary"])
    parser.add_argument("--output-json", default="onnx_test_metrics.json")
    parser.add_argument("--save-predictions", action="store_true", help="Save predicted masks and overlays as PNG files")
    parser.add_argument("--prediction-dir", default="onnx_predictions", help="Folder for saved prediction images")
    parser.add_argument("--overlay-alpha", default=0.45, type=float, help="Prediction overlay opacity")
    parser.add_argument("--providers", default="", help="Comma-separated ONNX Runtime providers. Empty uses available defaults.")
    parser.add_argument("--log-file", default="", help="Append console output and tracebacks to this file")
    return parser.parse_args()


def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
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


def make_session(onnx_path: str, providers_arg: str, logger: logging.Logger):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for test_onnx.py. Install it in this environment, "
            "for example: pip install onnxruntime-gpu"
        ) from exc

    available = ort.get_available_providers()
    providers = [item.strip() for item in providers_arg.split(",") if item.strip()]
    if not providers:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [provider for provider in preferred if provider in available]
        if not providers:
            providers = available
    logger.info("onnxruntime available_providers=%s selected_providers=%s", available, providers)
    return ort.InferenceSession(onnx_path, providers=providers)


def run_onnx(session, image: torch.Tensor) -> np.ndarray:
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    logits = session.run([output_info.name], {input_info.name: image.numpy()})[0]
    return logits


def save_prediction(
    dataset: LCMSCrackDataset,
    index: int,
    pred_mask: np.ndarray,
    output_dir: Path,
    num_classes: int,
    height: int,
    width: int,
    overlay_alpha: float,
) -> None:
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset.image_paths[index].stem
    pred_rgb = colorize_mask(pred_mask, num_classes)
    image_rgb = prepare_overlay_image(dataset.image_paths[index], height, width)
    overlay_rgb = cv2.addWeighted(image_rgb, 1.0 - overlay_alpha, pred_rgb, overlay_alpha, 0)
    cv2.imwrite(str(masks_dir / f"{stem}_pred.png"), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlays_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file) if args.log_file else Path(args.output_json).with_suffix(".log")
    logger = setup_logging(log_path, "lcms_unetpp.test_onnx")
    log_environment(logger)
    logger.info("command args: %s", json.dumps(vars(args), indent=2))

    session = make_session(args.onnx_path, args.providers, logger)
    ds = LCMSCrackDataset(args.data_path, args.split, EvalTransform(args.height, args.width), mask_mode=args.mask_mode)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=LCMSCrackDataset.collate_fn)
    metric_classes = 2 if args.mask_mode == "binary" or args.num_classes == 1 else args.num_classes
    metrics = SegmentationMetrics(metric_classes)
    output_dir = Path(args.prediction_dir)

    logger.info("evaluation start split=%s samples=%d", args.split, len(ds))
    for index, (images, targets) in enumerate(loader):
        logits = run_onnx(session, images.float())
        logits_t = torch.from_numpy(logits)
        if logits_t.shape[1] == 1:
            pred = (torch.sigmoid(logits_t[:, 0]) > 0.5).long()
            pred_mask = pred[0].numpy().astype(np.uint8)
            metrics.update(logits_t, targets)
            color_classes = 2
        else:
            pred = logits_t.argmax(1)
            pred_mask = pred[0].numpy().astype(np.uint8)
            metrics.update(logits_t, targets)
            color_classes = metric_classes
        if args.save_predictions:
            save_prediction(ds, index, pred_mask, output_dir, color_classes, args.height, args.width, args.overlay_alpha)
        if index % 25 == 0:
            logger.info("eval step=%d/%d", index, len(loader))

    result = metrics.compute()
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("evaluation done wrote metrics=%s", args.output_json)
    if args.save_predictions:
        logger.info("saved predictions to %s", args.prediction_dir)
        print(f"Saved predictions to {args.prediction_dir}")
    print(result)


if __name__ == "__main__":
    main()
