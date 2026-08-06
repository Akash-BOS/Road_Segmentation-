"""Create a binary foreground/background copy of an LCMS dataset.

The source dataset keeps the same TRAIN/VAL/TEST layout as the multiclass data:

    root/TRAIN/IMAGES
    root/TRAIN/MASKS
    root/VAL/IMAGES
    root/VAL/MASKS
    root/TEST/IMAGES
    root/TEST/MASKS

Images are copied unchanged. Masks are converted to 8-bit grayscale PNGs where
background is 0 and any known foreground class is 255. Unknown/ignore pixels are
written as 0 by default so the binary dataset can be used with mask_mode: binary.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


COLOR_MAP = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 0, 255): 2,
    (0, 255, 0): 3,
    (139, 69, 19): 4,
    (255, 165, 0): 5,
}


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LCMS multiclass masks to binary masks")
    parser.add_argument("--src", required=True, help="Source dataset root")
    parser.add_argument("--dst", required=True, help="Destination dataset root")
    parser.add_argument("--splits", default="TRAIN,VAL,TEST", help="Comma-separated splits to convert")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing destination")
    parser.add_argument(
        "--ignore-as-foreground",
        action="store_true",
        help="Treat unknown mask colors as foreground instead of background",
    )
    return parser.parse_args()


def color_mask_to_binary(mask_path: Path, ignore_as_foreground: bool) -> Image.Image:
    with Image.open(mask_path) as mask_img:
        mask = np.array(mask_img.convert("RGB"), dtype=np.uint8)

    class_map = np.full(mask.shape[:2], 255, dtype=np.uint8)
    for color, class_id in COLOR_MAP.items():
        matches = (
            (mask[:, :, 0] == color[0])
            & (mask[:, :, 1] == color[1])
            & (mask[:, :, 2] == color[2])
        )
        class_map[matches] = class_id

    foreground = (class_map > 0) & (class_map != 255)
    if ignore_as_foreground:
        foreground |= class_map == 255
    binary = np.where(foreground, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def convert_split(src_root: Path, dst_root: Path, split: str, overwrite: bool, ignore_as_foreground: bool) -> None:
    src_split = src_root / split
    src_images = src_split / "IMAGES"
    src_masks = src_split / "MASKS"
    if not src_images.exists() and not src_masks.exists():
        print(f"skip missing split={split}")
        return
    if not src_images.exists() or not src_masks.exists():
        raise FileNotFoundError(f"Expected IMAGES and MASKS under {src_split}")

    dst_images = dst_root / split / "IMAGES"
    dst_masks = dst_root / split / "MASKS"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in src_images.iterdir() if path.suffix.lower() in VALID_EXTS)
    copied = 0
    converted = 0
    for image_path in image_paths:
        mask_path = src_masks / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        image_dst = dst_images / image_path.name
        mask_dst = dst_masks / mask_path.name
        if not overwrite and (image_dst.exists() or mask_dst.exists()):
            raise FileExistsError(f"Destination exists; use --overwrite: {image_dst} or {mask_dst}")

        shutil.copy2(image_path, image_dst)
        color_mask_to_binary(mask_path, ignore_as_foreground).save(mask_dst)
        copied += 1
        converted += 1

    print(f"converted split={split} images={copied} masks={converted}")


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.exists():
        raise FileNotFoundError(src_root)
    if dst_root.exists() and any(dst_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Destination is not empty; use --overwrite: {dst_root}")

    splits = [split.strip().upper() for split in args.splits.split(",") if split.strip()]
    for split in splits:
        convert_split(src_root, dst_root, split, args.overwrite, args.ignore_as_foreground)


if __name__ == "__main__":
    main()
