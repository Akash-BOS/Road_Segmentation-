"""Run inference with a PyTorch .pth segmentation checkpoint without masks."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lcms_unetpp.config import apply_config_defaults, load_yaml_config
from lcms_unetpp.logging_utils import log_environment, setup_logging
from lcms_unetpp.models import build_model_from_args


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_COLOR_MAP: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 0, 0): 2,
    (0, 0, 0): 3,
    (0, 0, 0): 4,
    (0, 0, 0): 5,
}


def _main_logits(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["out"] if isinstance(outputs, dict) else outputs


def _predict_logits(model: torch.nn.Module, images: torch.Tensor, use_tta: bool) -> torch.Tensor:
    logits = _main_logits(model(images))
    if not use_tta:
        return logits
    flip_logits = _main_logits(model(torch.flip(images, dims=[3])))
    return (logits + torch.flip(flip_logits, dims=[3])) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LCMS PyTorch checkpoint inference without masks")
    parser.add_argument("--config", default="", help="Optional YAML config. CLI arguments override config defaults.")
    parser.add_argument("--data-path", default="", help="Dataset root, split folder, IMAGES folder, or image file")
    parser.add_argument("--checkpoint", default="", help="Path to .pth checkpoint")
    parser.add_argument("--split", default="TEST", help="Used when --data-path is a dataset root containing split/IMAGES")
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=419, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--in-channels", default=3, type=int)
    parser.add_argument("--num-classes", default=6, type=int)
    parser.add_argument(
        "--model",
        default="unetpp",
        choices=["unetpp", "unet++", "unet3plus", "unet3+", "unet_3plus", "unet_3_plus", "hrnet_ocr", "hrnet+ocr", "hrnetocr"],
    )
    parser.add_argument("--attention", default="gate", choices=["gate", "cbam", "se", "none"])
    parser.add_argument("--context", default="aspp", choices=["aspp", "pyramid", "identity", "none"])
    parser.add_argument("--norm", default="group", choices=["batch", "group", "instance"])
    parser.add_argument("--activation", default="silu", choices=["relu", "silu", "gelu", "leaky_relu"])
    parser.add_argument("--deep-supervision", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--prediction-dir", default="predictions", help="Folder for saved prediction images")
    parser.add_argument("--overlay-alpha", default=0.45, type=float, help="Prediction overlay opacity")
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip test-time augmentation")
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for auto")
    parser.add_argument("--use-model-args", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-file", default="", help="Append console output and tracebacks to this file")
    args = parser.parse_args()
    if args.config:
        args = apply_config_defaults(parser, args, load_yaml_config(args.config))
    return args


class ImageInferenceDataset(Dataset):
    """Image-only dataset for inference from common LCMS folder layouts."""

    def __init__(self, data_path: str, split: str, height: int, width: int, mean: float = 0.5, std: float = 0.25) -> None:
        self.data_path = Path(data_path)
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.image_paths = self._find_images(self.data_path, split)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under: {self.data_path}")

    @staticmethod
    def _find_images(data_path: Path, split: str) -> list[Path]:
        if data_path.is_file():
            if data_path.suffix.lower() in VALID_IMAGE_EXTS:
                return [data_path]
            raise FileNotFoundError(f"Unsupported image extension: {data_path}")

        candidates = [
            data_path / split.upper() / "IMAGES",
            data_path / split / "IMAGES",
            data_path / "IMAGES",
            data_path,
        ]
        for folder in candidates:
            if folder.exists() and folder.is_dir():
                images = sorted(p for p in folder.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS)
                if images:
                    return images
        return []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image = read_rgb_image(self.image_paths[index])
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        max_value = 65535.0 if image.max() > 255 else 255.0
        image = image / max_value
        image = (image - self.mean) / self.std
        image_t = torch.from_numpy(image).permute(2, 0, 1).float()
        return image_t, str(self.image_paths[index])


def read_rgb_image(image_path: Path) -> np.ndarray:
    try:
        with Image.open(image_path) as image_img:
            image = np.array(image_img.convert("RGB"), dtype=np.uint8, copy=True)
            return np.ascontiguousarray(image)
    except FileNotFoundError:
        raise FileNotFoundError(image_path) from None


def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    palette = np.zeros((max(num_classes, 1), 3), dtype=np.uint8)
    for color, class_id in DEFAULT_COLOR_MAP.items():
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


def load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Expected checkpoint dict in {path}, got {type(checkpoint).__name__}")


def build_model(args: argparse.Namespace, checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint.get("args", {}), dict) else {}
    if args.use_model_args:
        model_args = {**vars(args), **checkpoint_args}
    else:
        model_args = vars(args).copy()
    model_args["in_channels"] = int(model_args.get("in_channels", args.in_channels))
    model_args["num_classes"] = int(model_args.get("num_classes", args.num_classes))
    model = build_model_from_args(model_args, pretrained=False).to(device)

    if args.use_ema and "ema_model" in checkpoint:
        state = checkpoint["ema_model"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    return model


def collate_fn(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
    images, image_paths = zip(*batch)
    return torch.stack(images, dim=0), list(image_paths)


def save_prediction(
    image_path: Path,
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

    pred_rgb = colorize_mask(pred_mask, num_classes)
    image_rgb = prepare_overlay_image(image_path, height, width)
    overlay_rgb = cv2.addWeighted(image_rgb, 1.0 - overlay_alpha, pred_rgb, overlay_alpha, 0)

    cv2.imwrite(str(masks_dir / f"{image_path.stem}.png"), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlays_dir / f"{image_path.stem}_overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_classes: int,
    height: int,
    width: int,
    overlay_alpha: float,
    use_tta: bool,
    logger: logging.Logger,
) -> None:
    model.eval()
    for index, (images, image_paths) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        logits = _predict_logits(model, images, use_tta)
        if logits.shape[1] == 1:
            pred = (torch.sigmoid(logits[:, 0]) > 0.5).long()
            color_classes = 2
        else:
            pred = logits.argmax(1)
            color_classes = num_classes

        for batch_index, image_path_str in enumerate(image_paths):
            image_path = Path(image_path_str)
            pred_mask = pred[batch_index].cpu().numpy().astype(np.uint8)
            save_prediction(image_path, pred_mask, output_dir, color_classes, height, width, overlay_alpha)

        if index % 25 == 0:
            logger.info("inference step=%d/%d stem=%s", index, len(loader), Path(image_paths[0]).stem)


def main() -> None:
    args = parse_args()
    if not args.data_path:
        raise ValueError("Provide --data-path or data_path in --config")
    if not args.checkpoint:
        raise ValueError("Provide --checkpoint or checkpoint in --config")

    log_path = Path(args.log_file) if args.log_file else Path(args.prediction_dir) / "inference.log"
    logger = setup_logging(log_path, "lcms_unetpp.inference")
    log_environment(logger)
    logger.info("command args: %s", json.dumps(vars(args), indent=2))

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    logger.info("device=%s", device)

    logger.info("loading checkpoint=%s", args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(args, checkpoint, device)

    model_args = checkpoint.get("args", {}) if args.use_model_args and isinstance(checkpoint.get("args", {}), dict) else vars(args)
    model_num_classes = int(model_args.get("num_classes", args.num_classes))
    dataset = ImageInferenceDataset(args.data_path, args.split, args.height, args.width)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    logger.info("inference start samples=%d", len(dataset))
    run_inference(
        model,
        loader,
        device,
        Path(args.prediction_dir),
        model_num_classes,
        args.height,
        args.width,
        args.overlay_alpha,
        args.tta,
        logger,
    )
    logger.info("inference done saved predictions to %s", args.prediction_dir)
    print(f"Saved predictions to {args.prediction_dir}")


if __name__ == "__main__":
    main()
