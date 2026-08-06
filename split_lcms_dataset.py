"""Split an LCMS image/mask folder into TRAIN/VAL/TEST folders.

The source layout is expected to be:

    root/IMAGES/*.jpg
    root/MASKS/*.png

or an already-split layout:

    root/TRAIN/IMAGES
    root/TRAIN/MASKS
    root/VAL/IMAGES
    root/VAL/MASKS
    root/TEST/IMAGES
    root/TEST/MASKS

The output layout is:

    root/TRAIN/IMAGES
    root/TRAIN/MASKS
    root/VAL/IMAGES
    root/VAL/MASKS
    root/TEST/IMAGES
    root/TEST/MASKS
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lcms_unetpp.logging_utils import setup_logging


CLASS_NAMES = {
    0: "background",
    1: "alligator",
    2: "transverse",
    3: "longitudinal",
    4: "pothole",
    5: "patch",
}

COLOR_MAP = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 0, 255): 2,
    (0, 255, 0): 3,
    (139, 69, 19): 4,
    (255, 165, 0): 5,
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    mask_path: Path
    classes: frozenset[int]
    pixels: dict[int, int]
    instances: dict[int, int]


def mask_to_class_ids(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 2:
        return mask.astype(np.uint8)

    rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    class_map = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in COLOR_MAP.items():
        class_map[np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)] = class_id
    return class_map


def connected_instances(class_map: np.ndarray, class_id: int) -> int:
    if class_id == 0:
        return 1 if np.any(class_map == 0) else 0
    binary = (class_map == class_id).astype(np.uint8)
    count, _ = cv2.connectedComponents(binary, connectivity=8)
    return max(0, count - 1)


def collect_samples_from_dirs(images_dir: Path, masks_dir: Path, logger=None) -> list[Sample]:
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Expected IMAGES and MASKS folders: {images_dir}, {masks_dir}")

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    samples: list[Sample] = []
    missing_masks: list[Path] = []
    for index, image_path in enumerate(image_paths):
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            missing_masks.append(mask_path)
            continue

        class_map = mask_to_class_ids(mask_path)
        pixels = {
            class_id: int(np.count_nonzero(class_map == class_id))
            for class_id in CLASS_NAMES
        }
        classes = frozenset(class_id for class_id, count in pixels.items() if count > 0)
        instances = {
            class_id: connected_instances(class_map, class_id)
            for class_id in CLASS_NAMES
        }
        samples.append(Sample(image_path.stem, image_path, mask_path, classes, pixels, instances))
        if logger is not None and index % 50 == 0:
            logger.info("collected sample=%d/%d stem=%s", index, len(image_paths), image_path.stem)

    if missing_masks:
        preview = ", ".join(str(p) for p in missing_masks[:5])
        raise FileNotFoundError(f"Missing {len(missing_masks)} masks. First missing: {preview}")
    return samples


def collect_samples(root: Path, logger=None) -> list[Sample]:
    return collect_samples_from_dirs(root / "IMAGES", root / "MASKS", logger)


def collect_all_samples(root: Path, logger=None) -> list[Sample]:
    if (root / "IMAGES").exists() and (root / "MASKS").exists():
        return collect_samples(root, logger)

    samples_by_stem: dict[str, Sample] = {}
    found_split = False
    for split in ("TRAIN", "VAL", "TEST"):
        images_dir = root / split / "IMAGES"
        masks_dir = root / split / "MASKS"
        if not images_dir.exists() and not masks_dir.exists():
            continue
        found_split = True
        for sample in collect_samples_from_dirs(images_dir, masks_dir, logger):
            if sample.stem in samples_by_stem:
                raise ValueError(f"Duplicate sample stem across split folders: {sample.stem}")
            samples_by_stem[sample.stem] = sample
    if not found_split:
        raise FileNotFoundError(f"Expected either root/IMAGES+MASKS or split folders under {root}")
    return list(samples_by_stem.values())


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    total = len(samples)
    target_val_total = max(1, round(total * val_ratio))
    class_totals = Counter()
    for sample in samples:
        class_totals.update(sample.classes)

    target_val_by_class: dict[int, int] = {}
    for class_id in CLASS_NAMES:
        class_total = class_totals[class_id]
        if class_total >= 2:
            target_val_by_class[class_id] = min(class_total - 1, max(1, round(class_total * val_ratio)))
        else:
            target_val_by_class[class_id] = 0

    rng = random.Random(seed)
    remaining = samples[:]
    rng.shuffle(remaining)
    val: list[Sample] = []
    val_counts = Counter()
    train_counts = Counter(class_totals)

    def loss(counts: Counter) -> float:
        value = 0.0
        for class_id, target in target_val_by_class.items():
            denom = max(1, target)
            value += abs(counts[class_id] - target) / denom
        return value

    while len(val) < target_val_total and remaining:
        current_loss = loss(val_counts)

        def score(sample: Sample) -> tuple[float, float, int]:
            proposed = val_counts.copy()
            proposed.update(sample.classes)
            improvement = current_loss - loss(proposed)
            overshoot = sum(
                max(0, proposed[class_id] - target_val_by_class[class_id])
                for class_id in sample.classes
            )
            return improvement, -overshoot, len(sample.classes)

        eligible = [
            sample
            for sample in remaining
            if all(train_counts[class_id] > 1 for class_id in sample.classes if class_totals[class_id] >= 2)
        ]
        if not eligible:
            break
        candidate = max(eligible, key=score)
        remaining.remove(candidate)
        val.append(candidate)
        val_counts.update(candidate.classes)
        for class_id in candidate.classes:
            train_counts[class_id] -= 1

    val_stems = {sample.stem for sample in val}
    train = [sample for sample in samples if sample.stem not in val_stems]
    return train, val


def split_samples_by_pixels(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    priority_classes: set[int] | None = None,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    priority_classes = priority_classes or {2, 3}
    total_count = len(samples)
    total_pixels = Counter()
    for sample in samples:
        for class_id, pixels in sample.pixels.items():
            total_pixels[class_id] += pixels

    rng = random.Random(seed)
    remaining = samples[:]
    rng.shuffle(remaining)
    buckets: dict[str, list[Sample]] = {"VAL": [], "TEST": []}
    bucket_pixels: dict[str, Counter] = {"VAL": Counter(), "TEST": Counter()}
    target_counts = {
        "VAL": max(1, round(total_count * val_ratio)),
        "TEST": max(1, round(total_count * test_ratio)),
    }
    ratios = {"VAL": val_ratio, "TEST": test_ratio}
    pixel_targets = {
        split_name: {class_id: float(total_pixels[class_id]) * ratios[split_name] for class_id in CLASS_NAMES}
        for split_name in ("VAL", "TEST")
    }

    def score(split_name: str, sample: Sample) -> tuple[float, float, float]:
        if len(buckets[split_name]) >= target_counts[split_name]:
            return (-1e9, -1e9, -1e9)
        gain = 0.0
        overflow = 0.0
        for class_id in CLASS_NAMES:
            target = pixel_targets[split_name][class_id]
            if target <= 0:
                continue
            before = bucket_pixels[split_name][class_id]
            after = before + sample.pixels[class_id]
            useful = max(0.0, min(after, target) - min(before, target))
            extra = max(0.0, after - target) - max(0.0, before - target)
            class_weight = 10.0 if class_id in priority_classes else 1.0
            gain += class_weight * useful / target
            overflow += class_weight * extra / target
        size_fill = 1.0 - abs((len(buckets[split_name]) + 1) - target_counts[split_name]) / max(target_counts[split_name], 1)
        return (gain - overflow * 2.0, size_fill, rng.random())

    while remaining and (len(buckets["VAL"]) < target_counts["VAL"] or len(buckets["TEST"]) < target_counts["TEST"]):
        best: tuple[tuple[float, float, float], str, Sample] | None = None
        for sample in remaining:
            for split_name in ("VAL", "TEST"):
                candidate = (score(split_name, sample), split_name, sample)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None or best[0][0] <= -1e8:
            break
        _, split_name, sample = best
        remaining.remove(sample)
        buckets[split_name].append(sample)
        for class_id, pixels in sample.pixels.items():
            bucket_pixels[split_name][class_id] += pixels

    train = remaining
    return train, buckets["VAL"], buckets["TEST"]


def clean_split_dirs(root: Path, logger=None) -> None:
    for split in ("TRAIN", "VAL", "TEST"):
        for kind in ("IMAGES", "MASKS"):
            folder = root / split / kind
            folder.mkdir(parents=True, exist_ok=True)
            removed = 0
            for path in folder.iterdir():
                if path.is_file():
                    path.unlink()
                    removed += 1
            if logger is not None:
                logger.info("cleaned folder=%s removed_files=%d", folder, removed)


def copy_split(samples: list[Sample], root: Path, split: str, logger=None) -> None:
    images_out = root / split / "IMAGES"
    masks_out = root / split / "MASKS"
    for index, sample in enumerate(samples):
        shutil.copy2(sample.image_path, images_out / sample.image_path.name)
        shutil.copy2(sample.mask_path, masks_out / sample.mask_path.name)
        if logger is not None and index % 50 == 0:
            logger.info("copied split=%s sample=%d/%d stem=%s", split, index, len(samples), sample.stem)


def stage_split(samples: list[Sample], staging_root: Path, split: str, logger=None) -> None:
    images_out = staging_root / split / "IMAGES"
    masks_out = staging_root / split / "MASKS"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples):
        shutil.copy2(sample.image_path, images_out / sample.image_path.name)
        shutil.copy2(sample.mask_path, masks_out / sample.mask_path.name)
        if logger is not None and index % 50 == 0:
            logger.info("staged split=%s sample=%d/%d stem=%s", split, index, len(samples), sample.stem)


def install_staged_split(staging_root: Path, root: Path, split: str, logger=None) -> None:
    staged_images = staging_root / split / "IMAGES"
    staged_masks = staging_root / split / "MASKS"
    images_out = root / split / "IMAGES"
    masks_out = root / split / "MASKS"
    for index, image_path in enumerate(staged_images.iterdir()):
        shutil.copy2(image_path, images_out / image_path.name)
        if logger is not None and index % 50 == 0:
            logger.info("installed images split=%s file=%d path=%s", split, index, image_path.name)
    for index, mask_path in enumerate(staged_masks.iterdir()):
        shutil.copy2(mask_path, masks_out / mask_path.name)
        if logger is not None and index % 50 == 0:
            logger.info("installed masks split=%s file=%d path=%s", split, index, mask_path.name)


def summarize(samples: list[Sample]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for class_id, name in CLASS_NAMES.items():
        summary[name] = {
            "masks_containing_class": sum(class_id in sample.classes for sample in samples),
            "connected_instances": sum(sample.instances[class_id] for sample in samples),
            "pixels": sum(sample.pixels[class_id] for sample in samples),
        }
    return summary


def write_summary(root: Path, train: list[Sample], val: list[Sample], test: list[Sample], all_samples: list[Sample]) -> None:
    payload = {
        "total_images": len(all_samples),
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
        "classes": {
            "all": summarize(all_samples),
            "train": summarize(train),
            "val": summarize(val),
            "test": summarize(test),
        },
    }
    (root / "split_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (root / "split_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "masks_containing_class", "connected_instances", "pixels"])
        for split_name, split_samples in (("ALL", all_samples), ("TRAIN", train), ("VAL", val), ("TEST", test)):
            split_summary = summarize(split_samples)
            for class_name, counts in split_summary.items():
                writer.writerow([
                    split_name,
                    class_name,
                    counts["masks_containing_class"],
                    counts["connected_instances"],
                    counts["pixels"],
                ])


def print_summary(train: list[Sample], val: list[Sample], test: list[Sample], all_samples: list[Sample]) -> None:
    print(f"Images: total={len(all_samples)} train={len(train)} val={len(val)} test={len(test)}")
    print("Split  Class         Masks  Instances  Pixels")
    for split_name, split_samples in (("ALL", all_samples), ("TRAIN", train), ("VAL", val), ("TEST", test)):
        summary = summarize(split_samples)
        for class_name, counts in summary.items():
            print(
                f"{split_name:<6} {class_name:<12} "
                f"{counts['masks_containing_class']:>5} "
                f"{counts['connected_instances']:>10} "
                f"{counts['pixels']:>10}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--test-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--pixel-balanced", action="store_true", help="Build TRAIN/VAL/TEST using class pixel targets")
    parser.add_argument("--log-file", default="", help="Append console output and tracebacks to this file")
    args = parser.parse_args()

    log_path = Path(args.log_file) if args.log_file else args.root / "split.log"
    logger = setup_logging(log_path, "lcms_unetpp.split")
    logger.info(
        "split start root=%s val_ratio=%s test_ratio=%s pixel_balanced=%s seed=%s",
        args.root,
        args.val_ratio,
        args.test_ratio,
        args.pixel_balanced,
        args.seed,
    )

    samples = collect_all_samples(args.root, logger)
    if not samples:
        raise RuntimeError("No paired image/mask samples found")
    logger.info("collected total_samples=%d", len(samples))

    if args.pixel_balanced:
        train, val, test = split_samples_by_pixels(samples, args.val_ratio, args.test_ratio, args.seed)
    else:
        train, val = split_samples(samples, args.val_ratio, args.seed)
        test = []
    logger.info("split selected train=%d val=%d test=%d", len(train), len(val), len(test))

    staging_root = args.root / ".split_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    stage_split(train, staging_root, "TRAIN", logger)
    stage_split(val, staging_root, "VAL", logger)
    if test:
        stage_split(test, staging_root, "TEST", logger)

    clean_split_dirs(args.root, logger)
    install_staged_split(staging_root, args.root, "TRAIN", logger)
    install_staged_split(staging_root, args.root, "VAL", logger)
    if test:
        install_staged_split(staging_root, args.root, "TEST", logger)
    shutil.rmtree(staging_root)
    write_summary(args.root, train, val, test, samples)
    logger.info("wrote summary files")
    print_summary(train, val, test, samples)
    logger.info("split done")


if __name__ == "__main__":
    main()
