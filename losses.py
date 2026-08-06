"""Loss functions for multi-class LCMS semantic segmentation."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


def _valid_mask(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return target != ignore_index


def foreground_target(target: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Return binary foreground mask used only for boundary/edge supervision."""
    valid = _valid_mask(target, ignore_index)
    return ((target > 0) & valid).float()


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class logits.

    Background is excluded from the mean when ``include_background=False``.
    """

    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        smooth: float = 1.0,
        include_background: bool = False,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.include_background = include_background
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits[:, 0])
            target_f = foreground_target(target, self.ignore_index)
            probs = probs[valid]
            target_f = target_f[valid]
            inter = (probs * target_f).sum()
            denom = probs.sum() + target_f.sum()
            return 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)

        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        safe_target = target.clone()
        safe_target[~valid] = 0
        one_hot = F.one_hot(safe_target.long(), num_classes).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        one_hot = one_hot * valid_f
        if not self.include_background and num_classes > 1:
            probs = probs[:, 1:]
            one_hot = one_hot[:, 1:]
        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            if not self.include_background and num_classes > 1:
                weights = weights[1:]
            loss = loss * weights
            return loss.sum() / weights.sum().clamp_min(1e-7)
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss for imbalanced multi-class segmentation.

    ``alpha`` penalizes false positives and ``beta`` penalizes false negatives.
    For thin crack recall, a common starting point is ``alpha=0.3, beta=0.7``.
    ``gamma > 1`` makes the loss focus harder on poorly segmented classes.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        ignore_index: int = IGNORE_INDEX,
        smooth: float = 1.0,
        include_background: bool = False,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.include_background = include_background
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits[:, 0])
            target_f = foreground_target(target, self.ignore_index)
            probs = probs[valid]
            target_f = target_f[valid]
            tp = (probs * target_f).sum()
            fp = (probs * (1.0 - target_f)).sum()
            fn = ((1.0 - probs) * target_f).sum()
            score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            return (1.0 - score).pow(self.gamma)

        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        safe_target = target.clone()
        safe_target[~valid] = 0
        one_hot = F.one_hot(safe_target.long(), num_classes).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        one_hot = one_hot * valid_f
        if not self.include_background and num_classes > 1:
            probs = probs[:, 1:]
            one_hot = one_hot[:, 1:]

        tp = (probs * one_hot).sum(dim=(0, 2, 3))
        fp = (probs * (1.0 - one_hot)).sum(dim=(0, 2, 3))
        fn = ((1.0 - probs) * one_hot).sum(dim=(0, 2, 3))
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = (1.0 - score).pow(self.gamma)
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            if not self.include_background and num_classes > 1:
                weights = weights[1:]
            loss = loss * weights
            return loss.sum() / weights.sum().clamp_min(1e-7)
        return loss.mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss with ignore-index support."""

    def __init__(
        self,
        gamma: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index)
        if logits.shape[1] == 1:
            target_f = foreground_target(target, self.ignore_index)
            bce = F.binary_cross_entropy_with_logits(logits[:, 0], target_f, reduction="none")
            prob = torch.sigmoid(logits[:, 0])
            pt = prob * target_f + (1.0 - prob) * (1.0 - target_f)
            return ((1.0 - pt).pow(self.gamma) * bce)[valid].mean()

        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, target.long(), ignore_index=self.ignore_index, reduction="none", weight=weights)
        safe_target = target.clone()
        safe_target[~valid] = 0
        pt = torch.softmax(logits, dim=1).gather(1, safe_target.unsqueeze(1)).squeeze(1)
        loss = (1.0 - pt).pow(self.gamma) * ce
        return loss[valid].mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension with respect to sorted errors."""
    num_pixels = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-7)
    if num_pixels > 1:
        jaccard[1:num_pixels] = jaccard[1:num_pixels] - jaccard[0 : num_pixels - 1]
    return jaccard


class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax loss for directly optimizing a convex IoU surrogate."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, include_background: bool = False) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits[:, 0])
            target_f = foreground_target(target, self.ignore_index)
            return self._binary_lovasz(probs[valid], target_f[valid])

        probs = torch.softmax(logits, dim=1).permute(0, 2, 3, 1)[valid]
        target_flat = target[valid].long()
        losses = []
        start_class = 0 if self.include_background else 1
        for class_idx in range(start_class, logits.shape[1]):
            fg = (target_flat == class_idx).float()
            if fg.sum() == 0:
                continue
            class_pred = probs[:, class_idx]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _binary_lovasz(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.numel() == 0 or target.sum() == 0:
            return probs.sum() * 0.0
        errors = (target - probs).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        target_sorted = target[perm]
        return torch.dot(errors_sorted, _lovasz_grad(target_sorted))


class CrossEntropyRegionLoss(nn.Module):
    """CrossEntropy plus Dice or Tversky for multi-class segmentation."""

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        ignore_index: int = IGNORE_INDEX,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        region_loss: str = "dice",
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 1.0,
        ohem_ratio: float = 1.0,
        ohem_min_kept: int = 4096,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        region_loss = region_loss.lower()
        if region_loss == "dice":
            self.region = DiceLoss(ignore_index=ignore_index, class_weights=class_weights)
        elif region_loss == "tversky":
            self.region = TverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=tversky_gamma,
                ignore_index=ignore_index,
                class_weights=class_weights,
            )
        else:
            raise ValueError(f"Unsupported region_loss: {region_loss}")
        self.ignore_index = ignore_index
        self.ohem_ratio = min(max(ohem_ratio, 0.0), 1.0)
        self.ohem_min_kept = max(1, int(ohem_min_kept))
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index)
        if logits.shape[1] == 1:
            target_f = foreground_target(target, self.ignore_index)
            ce_loss = F.binary_cross_entropy_with_logits(logits[:, 0], target_f, reduction="none")[valid]
        else:
            weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
            ce_map = F.cross_entropy(logits, target.long(), ignore_index=self.ignore_index, weight=weights, reduction="none")
            ce_loss = ce_map[valid]
        ce = self._reduce_ohem(ce_loss)
        return self.ce_weight * ce + self.dice_weight * self.region(logits, target)

    def _reduce_ohem(self, losses: torch.Tensor) -> torch.Tensor:
        if losses.numel() == 0:
            return losses.sum() * 0.0
        if self.ohem_ratio >= 1.0:
            return losses.mean()
        keep = max(self.ohem_min_kept, int(losses.numel() * self.ohem_ratio))
        keep = min(keep, losses.numel())
        return torch.topk(losses, keep, sorted=False).values.mean()


class BoundaryLoss(nn.Module):
    """Foreground boundary loss for thin structures in a multi-class mask."""

    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    @staticmethod
    def edges(mask: torch.Tensor) -> torch.Tensor:
        """Return a compact boundary map for ``[B, 1, H, W]`` masks/probabilities."""
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_fg = foreground_target(target, self.ignore_index).unsqueeze(1)
        if logits.shape[1] == 1:
            pred_fg = torch.sigmoid(logits[:, :1])
        else:
            pred_fg = torch.softmax(logits, dim=1)[:, 1:].sum(1, keepdim=True)
        pred_edge = self.edges(pred_fg)
        target_edge = self.edges(target_fg)
        valid = _valid_mask(target, self.ignore_index).unsqueeze(1)
        pred_edge = pred_edge.float().clamp(1e-4, 1 - 1e-4)
        target_edge = target_edge.float()
        loss = -(target_edge * pred_edge.log() + (1.0 - target_edge) * (1.0 - pred_edge).log())
        return loss[valid].mean()


class CrackSegmentationLoss(nn.Module):
    """Combined multi-class crack segmentation loss.

    The segmentation logits are optimized with CE+Dice/Tversky, focal loss, and optional
    foreground-boundary supervision. Deep supervision heads are weighted by
    ``aux_weight``. The optional edge head predicts foreground boundaries across
    all non-background classes.
    """

    def __init__(
        self,
        main_weight: float = 1.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.25,
        boundary_weight: float = 0.1,
        edge_weight: float = 0.2,
        aux_weight: float = 0.4,
        ignore_index: int = IGNORE_INDEX,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        region_loss: str = "dice",
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 1.0,
        lovasz_weight: float = 0.0,
        lovasz_start_epoch: int = -1,
        ohem_start_epoch: int = -1,
        ohem_ratio: float = 1.0,
        ohem_min_kept: int = 4096,
    ) -> None:
        super().__init__()
        self.current_epoch = 0
        self.ohem_start_epoch = ohem_start_epoch
        self.ohem_ratio = min(max(ohem_ratio, 0.0), 1.0)
        self.lovasz_weight = max(0.0, lovasz_weight)
        self.lovasz_start_epoch = lovasz_start_epoch
        self.seg = CrossEntropyRegionLoss(
            1.0 - dice_weight,
            dice_weight,
            ignore_index,
            class_weights,
            region_loss=region_loss,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_gamma=tversky_gamma,
            ohem_ratio=1.0,
            ohem_min_kept=ohem_min_kept,
        )
        self.focal = FocalLoss(ignore_index=ignore_index, class_weights=class_weights)
        self.boundary = BoundaryLoss(ignore_index=ignore_index)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index)
        self.main_weight = main_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.edge_weight = edge_weight
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        if self.ohem_start_epoch >= 0 and epoch >= self.ohem_start_epoch:
            self.seg.ohem_ratio = self.ohem_ratio
        else:
            self.seg.ohem_ratio = 1.0

    def _lovasz_active(self) -> bool:
        return self.lovasz_weight > 0 and self.lovasz_start_epoch >= 0 and self.current_epoch >= self.lovasz_start_epoch

    def _seg_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.seg(logits, target)
        loss = loss + self.focal_weight * self.focal(logits, target)
        loss = loss + self.boundary_weight * self.boundary(logits, target)
        if self._lovasz_active():
            loss = loss + self.lovasz_weight * self.lovasz(logits, target)
        return loss

    def forward(self, outputs: torch.Tensor | Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if not isinstance(outputs, dict):
            return self._seg_loss(outputs, target)
        loss = self.main_weight * self._seg_loss(outputs["out"], target)
        aux_names = [name for name in ("aux0", "aux1", "aux2") if name in outputs]
        if aux_names:
            aux_loss = sum(self._seg_loss(outputs[name], target) for name in aux_names) / len(aux_names)
            loss = loss + self.aux_weight * aux_loss
        if "edge" in outputs and self.edge_weight > 0:
            edge_target = BoundaryLoss.edges(foreground_target(target, self.ignore_index).unsqueeze(1))
            valid = _valid_mask(target, self.ignore_index).unsqueeze(1)
            edge_loss = F.binary_cross_entropy_with_logits(outputs["edge"], edge_target, reduction="none")
            loss = loss + self.edge_weight * edge_loss[valid].mean()
        return loss
