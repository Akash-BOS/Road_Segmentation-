"""Count per-class instances in color ground-truth masks.

The script expects LCMS color masks that use the same RGB palette as data.py.
For each foreground class it counts connected components in every mask, then
writes dataset-level totals plus optional per-mask details.
"""

from __future__ import annotations



import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import COLOR_MAP, IGNORE_INDEX


VALID_MASK_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

CLASS_NAMES = {
    0: "BACKGROUND",
    1: "ALLIGATOR",
    2: "TRANSVERSE",
    3: "LONGITUDINAL",
    4: "POTHOLE",
    5: "PATCH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count per-class connected-component instances in GT color masks")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mask-dir", help="Folder containing GT mask images")
    source.add_argument("--data-path", help="Dataset root containing SPLIT/MASKS folders")
    parser.add_argument("--splits", default="TRAIN,VAL,TEST", help="Comma-separated splits used with --data-path")
    parser.add_argument("--recursive", action="store_true", help="Search --mask-dir recursively")
    parser.add_argument("--num-classes", default=6, type=int, help="Total classes including background")
    parser.add_argument("--connectivity", default=8, type=int, choices=[4, 8], help="Connected-component connectivity")
    parser.add_argument(
        "--min-instance-pixels",
        default=1,
        type=int,
        help="Ignore connected components smaller than this many pixels",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Also count class 0/background connected components",
    )
    parser.add_argument("--output-json", default="gt_mask_instance_counts.json")
    parser.add_argument("--output-csv", default="", help="Optional dataset-level CSV output")
    parser.add_argument("--per-mask-csv", default="", help="Optional per-mask per-class CSV output")
    return parser.parse_args()


def iter_mask_paths(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in folder.glob(pattern) if path.is_file() and path.suffix.lower() in VALID_MASK_EXTS)


def collect_mask_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
        if not mask_dir.exists():
            raise FileNotFoundError(mask_dir)
        return [(mask_dir.name, path) for path in iter_mask_paths(mask_dir, args.recursive)]

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    mask_paths: list[tuple[str, Path]] = []
    for split in [item.strip().upper() for item in args.splits.split(",") if item.strip()]:
        split_mask_dir = data_path / split / "MASKS"
        if not split_mask_dir.exists():
            print(f"skip missing split masks: {split_mask_dir}")
            continue
        mask_paths.extend((split, path) for path in iter_mask_paths(split_mask_dir, False))
    return mask_paths


def color_mask_to_labels(mask_path: Path, num_classes: int) -> tuple[np.ndarray, dict[str, int]]:
    with Image.open(mask_path) as image:
        mask = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)

    labels = np.full(mask.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    known = np.zeros(mask.shape[:2], dtype=bool)
    for color, class_id in COLOR_MAP.items():
        if class_id >= num_classes:
            continue
        matches = (mask[:, :, 0] == color[0]) & (mask[:, :, 1] == color[1]) & (mask[:, :, 2] == color[2])
        labels[matches] = class_id
        known |= matches

    unknown_pixels = int(np.count_nonzero(~known))
    unknown_colors: dict[str, int] = {}
    if unknown_pixels:
        colors, counts = np.unique(mask[~known].reshape(-1, 3), axis=0, return_counts=True)
        unknown_colors = {
            f"{int(color[0])},{int(color[1])},{int(color[2])}": int(count)
            for color, count in zip(colors, counts)
        }
    return labels, unknown_colors


def count_class_instances(labels: np.ndarray, class_id: int, connectivity: int, min_pixels: int) -> tuple[int, int]:
    class_mask = (labels == class_id).astype(np.uint8)
    pixel_count = int(class_mask.sum())
    if pixel_count == 0:
        return 0, 0

    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=connectivity)
    kept = 0
    kept_pixels = 0
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_pixels:
            kept += 1
            kept_pixels += area
    return kept, kept_pixels


def build_class_info(num_classes: int) -> dict[int, dict[str, object]]:
    by_class_id = {class_id: color for color, class_id in COLOR_MAP.items()}
    return {
        class_id: {
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"CLASS_{class_id}"),
            "color_rgb": list(by_class_id.get(class_id, ())),
            "instances": 0,
            "pixels": 0,
            "masks_with_class": 0,
        }
        for class_id in range(num_classes)
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    min_pixels = max(1, args.min_instance_pixels)
    mask_paths = collect_mask_paths(args)
    if not mask_paths:
        raise FileNotFoundError("No GT mask files found")

    class_ids = list(range(args.num_classes)) if args.include_background else list(range(1, args.num_classes))
    totals = build_class_info(args.num_classes)
    per_mask_rows: list[dict[str, object]] = []
    unknown_colors_total: dict[str, int] = {}

    for index, (split, mask_path) in enumerate(mask_paths, start=1):
        labels, unknown_colors = color_mask_to_labels(mask_path, args.num_classes)
        for color, count in unknown_colors.items():
            unknown_colors_total[color] = unknown_colors_total.get(color, 0) + count

        for class_id in class_ids:
            instances, pixels = count_class_instances(labels, class_id, args.connectivity, min_pixels)
            totals[class_id]["instances"] = int(totals[class_id]["instances"]) + instances
            totals[class_id]["pixels"] = int(totals[class_id]["pixels"]) + pixels
            if pixels > 0:
                totals[class_id]["masks_with_class"] = int(totals[class_id]["masks_with_class"]) + 1
            per_mask_rows.append(
                {
                    "split": split,
                    "mask": mask_path.name,
                    "class_id": class_id,
                    "class_name": totals[class_id]["class_name"],
                    "color_rgb": ",".join(str(value) for value in totals[class_id]["color_rgb"]),
                    "instances": instances,
                    "pixels": pixels,
                }
            )

        if index % 100 == 0:
            print(f"processed {index}/{len(mask_paths)} masks")

    summary_rows = [totals[class_id] for class_id in class_ids]
    result = {
        "mask_count": len(mask_paths),
        "connectivity": args.connectivity,
        "min_instance_pixels": min_pixels,
        "classes": summary_rows,
        "unknown_color_pixels": int(sum(unknown_colors_total.values())),
        "unknown_colors": unknown_colors_total,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.output_csv:
        write_csv(Path(args.output_csv), summary_rows)
    if args.per_mask_csv:
        write_csv(Path(args.per_mask_csv), per_mask_rows)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
