"""Metric accumulation and epoch-history plotting."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

IGNORE_INDEX = 255
LOGGER = logging.getLogger("lcms_unetpp.metrics")


class SegmentationMetrics:
    """Confusion-matrix metrics for multi-class semantic segmentation."""

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(1) if logits.shape[1] > 1 else (torch.sigmoid(logits[:, 0]) > 0.5).long()
        self.update_labels(pred, target)

    def update_labels(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update metrics from already-decoded class-id prediction masks."""
        target = target.detach().cpu().long()
        pred = pred.detach().cpu().long()
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
            & (pred >= 0)
            & (pred < self.num_classes)
        )
        inds = self.num_classes * target[valid] + pred[valid]
        self.mat += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes).double()

    def compute(self) -> Dict[str, float]:
        h = self.mat
        eps = 1e-7
        tp = torch.diag(h)
        precision = tp / (h.sum(0) + eps)
        recall = tp / (h.sum(1) + eps)
        iou = tp / (h.sum(1) + h.sum(0) - tp + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        acc = tp.sum() / (h.sum() + eps)
        fg = slice(1, None) if self.num_classes > 1 else slice(0, None)
        result = {
            "accuracy": acc.item(),
            "mean_precision": precision[fg].mean().item(),
            "mean_recall": recall[fg].mean().item(),
            "mean_f1": f1[fg].mean().item(),
            "mean_iou": iou[fg].mean().item(),
            "mean_dice": f1[fg].mean().item(),
        }
        for class_idx in range(self.num_classes):
            result[f"class_{class_idx}_precision"] = precision[class_idx].item()
            result[f"class_{class_idx}_recall"] = recall[class_idx].item()
            result[f"class_{class_idx}_f1"] = f1[class_idx].item()
            result[f"class_{class_idx}_iou"] = iou[class_idx].item()
        return result


def append_history(csv_path: str | Path, row: Dict[str, float]) -> None:
    """Append an epoch row to a metrics CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    keys = list(row.keys())
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    LOGGER.info("appended metrics row csv=%s epoch=%s", csv_path, row.get("epoch"))


def read_history(csv_path: str | Path) -> List[Dict[str, float]]:
    with Path(csv_path).open("r", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def truncate_history(csv_path: str | Path, keep_epochs_before: int) -> None:
    """Drop metric rows at or after ``keep_epochs_before`` when resuming."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if int(float(row.get("epoch", -1))) < keep_epochs_before]
        fieldnames = reader.fieldnames
    if not fieldnames:
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("truncated metrics csv=%s keep_epochs_before=%d rows=%d", csv_path, keep_epochs_before, len(rows))


def plot_history(csv_path: str | Path, output_path: str | Path) -> None:
    """Plot top-level metrics over epochs to one PNG."""
    rows = read_history(csv_path)
    if not rows:
        LOGGER.warning("no history rows found csv=%s", csv_path)
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row.get("epoch", i) for i, row in enumerate(rows)]
    preferred = ["train_loss", "val_loss", "mean_dice", "mean_iou", "mean_precision", "mean_recall", "accuracy", "lr"]
    keys = [key for key in preferred if key in rows[0]]
    cols = 2
    rows_count = max((len(keys) + 1) // cols, 1)
    fig = plt.figure(figsize=(12, 4 * rows_count))
    try:
        for idx, key in enumerate(keys, 1):
            ax = fig.add_subplot(rows_count, cols, idx)
            ax.plot(epochs, [row[key] for row in rows], marker="o", linewidth=1.5)
            ax.set_title(key)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)
    LOGGER.info("saved history plot=%s rows=%d", output_path, len(rows))
