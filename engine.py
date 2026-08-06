"""Training and evaluation engine for LCMS UNet++."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping

import torch
from torch import nn

from .metrics import SegmentationMetrics


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(model_state[key].detach(), alpha=1.0 - self.decay)
            else:
                value.copy_(model_state[key])


def create_warmup_cosine_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int = 5):
    """Linear warmup followed by cosine decay."""
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(steps_per_epoch * warmup_epochs, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _main_logits(outputs: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    return outputs["out"] if isinstance(outputs, Mapping) else outputs


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler,
    scheduler=None,
    ema: ModelEMA | None = None,
    grad_clip: float = 1.0,
    log_interval: int = 20,
    progress_callback: Callable[[int, float, float], None] | None = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        if progress_callback is not None:
            progress_callback(step, loss.item(), optimizer.param_groups[0]["lr"])
        if log_interval and step % log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"step {step:04d}/{len(loader)} loss={loss.item():.4f} lr={lr:.6g}")
    return {"train_loss": total_loss / max(len(loader), 1), "lr": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    metric_classes: int = 2,
    use_tta: bool = False,
    progress_callback: Callable[[int, float], None] | None = None,
) -> Dict[str, float]:
    model.eval()
    metrics = SegmentationMetrics(metric_classes)
    total_loss = 0.0
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        logits = predict_tta(model, images) if use_tta else _main_logits(outputs)
        total_loss += loss.item()
        metrics.update(logits, targets)
        if progress_callback is not None:
            progress_callback(step, loss.item())
    result = metrics.compute()
    result["val_loss"] = total_loss / max(len(loader), 1)
    return result


@torch.no_grad()
def predict_tta(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Horizontal-flip TTA for evaluation/testing."""
    out = _main_logits(model(images))
    flip_out = _main_logits(model(torch.flip(images, dims=[3])))
    flip_out = torch.flip(flip_out, dims=[3])
    return (out + flip_out) / 2.0
