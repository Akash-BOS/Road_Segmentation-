"""Compute LCMS segmentation metrics from GT and predicted mask folders."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lcms_unetpp.data import COLOR_MAP, IGNORE_INDEX
from lcms_unetpp.metrics import SegmentationMetrics


VALID_MASK_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PRED_SUFFIXES = (
    "_crack_pred",
    "_pred",
    "_prediction",
    "_mask",
    "_seg",
    "_crack",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure test.py-style metrics from GT masks and prediction masks")
    parser.add_argument("--gt-dir", required=True, help="Folder containing ground-truth masks")
    parser.add_argument("--pred-dir", required=True, help="Folder containing predicted masks")
    parser.add_argument("--num-classes", default=6, type=int, help="Number of metric classes")
    parser.add_argument("--gt-mask-mode", default="color", choices=["auto", "color", "index", "binary"])
    parser.add_argument("--pred-mask-mode", default="auto", choices=["auto", "color", "index", "binary"])
    parser.add_argument("--recursive", action="store_true", help="Search both folders recursively")
    parser.add_argument("--resize-pred-to-gt", action="store_true", help="Resize prediction masks to GT mask size with nearest neighbor")
    parser.add_argument("--output-json", default="mask_metrics.json")
    parser.add_argument("--output-csv", default="", help="Optional single-row CSV output")
    parser.add_argument("--per-image-csv", default="", help="Optional per-image metric CSV output")
    return parser.parse_args()


def iter_mask_paths(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in folder.glob(pattern) if path.is_file() and path.suffix.lower() in VALID_MASK_EXTS)


def canonical_stem(path: Path) -> str:
    stem = path.stem
    changed = True
    while changed:
        changed = False
        lower = stem.lower()
        for suffix in PRED_SUFFIXES:
            if lower.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
                break
    return stem


def index_paths(paths: list[Path]) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in paths:
        key = canonical_stem(path)
        if key in indexed:
            duplicates.setdefault(key, [indexed[key]]).append(path)
        else:
            indexed[key] = path
    if duplicates:
        preview = "; ".join(f"{key}: {', '.join(str(p) for p in values[:3])}" for key, values in list(duplicates.items())[:5])
        raise ValueError(f"Duplicate mask stems after suffix normalization: {preview}")
    return indexed


def read_mask(path: Path, mode: str, num_classes: int) -> np.ndarray:
    with Image.open(path) as image:
        array = np.array(image, dtype=np.uint8, copy=True)

    if mode == "auto":
        mode = infer_mask_mode(array, num_classes)
    if mode == "binary":
        if array.ndim == 3:
            return (array.sum(axis=2) > 0).astype(np.uint8)
        return (array > 0).astype(np.uint8)
    if mode == "index":
        if array.ndim == 3:
            array = array[:, :, 0]
        return array.astype(np.uint8)
    if mode == "color":
        if array.ndim == 2:
            return array.astype(np.uint8)
        if array.shape[2] == 4:
            array = array[:, :, :3]
        class_map = np.full(array.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        for color, class_id in COLOR_MAP.items():
            if class_id < num_classes:
                matches = (array[:, :, 0] == color[0]) & (array[:, :, 1] == color[1]) & (array[:, :, 2] == color[2])
                class_map[matches] = class_id
        return class_map
    raise ValueError(f"Unsupported mask mode: {mode}")


def infer_mask_mode(array: np.ndarray, num_classes: int) -> str:
    if array.ndim == 2:
        values = np.unique(array)
        if len(values) <= 2 and values.max(initial=0) > 1:
            return "binary"
        return "index"

    rgb = array[:, :, :3] if array.shape[2] >= 3 else array
    if rgb.ndim == 3 and np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 0], rgb[:, :, 2]):
        values = np.unique(rgb[:, :, 0])
        if len(values) <= 2 and values.max(initial=0) > 1:
            return "binary"
        return "index"

    flat_colors = np.unique(rgb.reshape(-1, 3), axis=0)
    known_colors = {color for color, class_id in COLOR_MAP.items() if class_id < num_classes}
    if all(tuple(int(channel) for channel in color) in known_colors for color in flat_colors):
        return "color"
    return "binary"


def ensure_same_shape(gt: np.ndarray, pred: np.ndarray, pred_path: Path, resize_pred: bool) -> np.ndarray:
    if gt.shape == pred.shape:
        return pred
    if resize_pred:
        return cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    raise ValueError(f"Shape mismatch for {pred_path}: gt={gt.shape} pred={pred.shape}. Use --resize-pred-to-gt to resize.")


def compute_pair_metrics(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> dict[str, float]:
    metrics = SegmentationMetrics(num_classes)
    metrics.update_labels(torch.from_numpy(pred)[None, :, :], torch.from_numpy(gt)[None, :, :])
    return metrics.compute()


def write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_per_image_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    gt_paths = index_paths(iter_mask_paths(gt_dir, args.recursive))
    pred_paths = index_paths(iter_mask_paths(pred_dir, args.recursive))

    common_keys = sorted(gt_paths.keys() & pred_paths.keys())
    if not common_keys:
        raise FileNotFoundError(f"No matching mask stems found between {gt_dir} and {pred_dir}")

    missing_preds = sorted(gt_paths.keys() - pred_paths.keys())
    extra_preds = sorted(pred_paths.keys() - gt_paths.keys())
    metrics = SegmentationMetrics(args.num_classes)
    per_image_rows: list[dict[str, object]] = []

    for index, key in enumerate(common_keys, start=1):
        gt = read_mask(gt_paths[key], args.gt_mask_mode, args.num_classes)
        pred = read_mask(pred_paths[key], args.pred_mask_mode, args.num_classes)
        pred = ensure_same_shape(gt, pred, pred_paths[key], args.resize_pred_to_gt)
        metrics.update_labels(torch.from_numpy(pred)[None, :, :], torch.from_numpy(gt)[None, :, :])
        if args.per_image_csv:
            per_image_rows.append({"image": key, **compute_pair_metrics(gt, pred, args.num_classes)})
        if index % 25 == 0:
            print(f"processed {index}/{len(common_keys)}")

    result: dict[str, object] = {
        "matched_masks": len(common_keys),
        "missing_predictions": len(missing_preds),
        "extra_predictions": len(extra_preds),
        **metrics.compute(),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_csv:
        write_csv(Path(args.output_csv), result)
    if args.per_image_csv:
        write_per_image_csv(Path(args.per_image_csv), per_image_rows)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
