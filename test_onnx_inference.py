"""Run inference with an exported ONNX segmentation model without masks."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lcms_unetpp.data import COLOR_MAP
from lcms_unetpp.logging_utils import log_environment, setup_logging


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LCMS ONNX segmentation inference without masks")
    parser.add_argument("--data-path", required=True, help="Dataset root, split folder, IMAGES folder, or image file")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--split", default="TEST", help="Used when --data-path is a dataset root containing split/IMAGES")
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=419, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--num-classes", default=6, type=int)
    parser.add_argument("--prediction-dir", default="onnx_predictions", help="Folder for saved prediction images")
    parser.add_argument("--overlay-alpha", default=0.45, type=float, help="Prediction overlay opacity")
    parser.add_argument("--providers", default="", help="Comma-separated ONNX Runtime providers. Empty uses available defaults.")
    parser.add_argument("--log-file", default="", help="Append console output and tracebacks to this file")
    return parser.parse_args()


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
            "onnxruntime is required for test_onnx_inference.py. Install it in this environment, "
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


def run_onnx(session, images: torch.Tensor) -> np.ndarray:
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    return session.run([output_info.name], {input_info.name: images.numpy()})[0]


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

    cv2.imwrite(str(masks_dir / f"{image_path.stem}_pred.png"), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlays_dir / f"{image_path.stem}_overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


def collate_fn(batch):
    images, image_paths = zip(*batch)
    return torch.stack(images, dim=0), list(image_paths)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file) if args.log_file else Path(args.prediction_dir) / "onnx_inference.log"
    logger = setup_logging(log_path, "lcms_unetpp.test_onnx_inference")
    log_environment(logger)
    logger.info("command args: %s", json.dumps(vars(args), indent=2))

    session = make_session(args.onnx_path, args.providers, logger)
    dataset = ImageInferenceDataset(args.data_path, args.split, args.height, args.width)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    output_dir = Path(args.prediction_dir)

    logger.info("inference start samples=%d", len(dataset))
    for index, (images, image_paths) in enumerate(loader):
        logits = torch.from_numpy(run_onnx(session, images.float()))
        if logits.shape[1] == 1:
            pred_mask = (torch.sigmoid(logits[0, 0]) > 0.5).numpy().astype(np.uint8)
            color_classes = 2
        else:
            pred_mask = logits.argmax(1)[0].numpy().astype(np.uint8)
            color_classes = args.num_classes

        image_path = Path(image_paths[0])
        save_prediction(image_path, pred_mask, output_dir, color_classes, args.height, args.width, args.overlay_alpha)
        if index % 25 == 0:
            logger.info("inference step=%d/%d stem=%s", index, len(loader), image_path.stem)

    logger.info("inference done saved predictions to %s", args.prediction_dir)
    print(f"Saved predictions to {args.prediction_dir}")


if __name__ == "__main__":
    main()
