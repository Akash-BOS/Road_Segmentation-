"""Train LCMS UNet++ for multi-class road-crack semantic segmentation."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import apply_config_defaults, load_yaml_config
from clearml_utils import ClearMLRun
from data import (
    BiasedPatchTrainTransform,
    EvalTransform,
    LCMSCrackDataset,
    LCMSCrackPatchDataset,
    MixedFullPatchDataset,
    TrainTransform,
)
from engine import ModelEMA, create_warmup_cosine_scheduler, evaluate, train_one_epoch
from logging_utils import log_environment, setup_logging, timestamp, write_json
from losses import CrackSegmentationLoss
from metrics import append_history, plot_history, truncate_history
from models import build_model_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LCMS semantic segmentation")
    parser.add_argument("--config", default="", help="Optional YAML config. CLI arguments override config defaults.")
    parser.add_argument("--data-path", default="", help="Dataset root containing TRAIN/VAL folders")
    parser.add_argument("--output-dir", default="weights/lcms_unetpp", help="Checkpoint and metric output folder")
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader workers. Use 0 on Windows to avoid worker deadlocks.")
    parser.add_argument("--height", default=992, type=int)
    parser.add_argument("--width", default=416, type=int)
    parser.add_argument("--no-patch-training", action="store_true", help="Train on resized full images instead of biased online patches")
    parser.add_argument("--patch-height", default=512, type=int)
    parser.add_argument("--patch-width", default=384, type=int)
    parser.add_argument("--samples-per-image", default=8, type=int, help="Virtual patch samples per source image per epoch")
    parser.add_argument("--anomaly-ratio", default=0.8, type=float, help="Approximate fraction of anomaly-centered training patches")
    parser.add_argument("--random-crop-ratio", default=0.1, type=float, help="Fraction of patch samples taken as fully random crops before anomaly/background selection")
    parser.add_argument("--full-image-ratio", default=0.1, type=float, help="Fraction of patch-training samples replaced by resized full-image training samples")
    parser.add_argument("--min-target-pixels", default=32, type=int, help="Minimum target-class pixels preferred in anomaly-centered patch crops")
    parser.add_argument("--target-crop-attempts", default=32, type=int, help="Number of candidate anomaly crops to try before keeping the best crop")
    parser.add_argument("--foreground-extension-ratio", default=0.0, type=float, help="Crop extra target-centered context and resize back to patch size")
    parser.add_argument(
        "--class-sampling-weights",
        default="1:2,2:8,3:14,4:2,5:2",
        help="Foreground patch sampling weights, e.g. alligator/transverse/longitudinal/pothole/patch as 1:2,2:6,3:5,4:2,5:2",
    )
    parser.add_argument(
        "--loss-class-weights",
        default="0:0.05,1:2,2:100,3:180,4:4,5:2",
        help="Per-pixel loss weights by class id. Use high values for rare crack pixels, e.g. 0:0.05,1:2,2:100,3:180,4:4,5:2",
    )
    parser.add_argument("--in-channels", default=3, type=int)
    parser.add_argument("--num-classes", default=6, type=int, help="Total classes including background")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-epochs", default=5, type=int)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--model", default="unetpp", choices=["unetpp", "unet3plus", "hrnet_ocr"], help="Segmentation architecture")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dilated-encoder", action="store_true")
    parser.add_argument("--attention", default="gate", choices=["cbam", "se", "gate", "none"])
    parser.add_argument("--context", default="aspp", choices=["aspp", "pyramid", "identity", "none"])
    parser.add_argument("--norm", default="group", choices=["batch", "group", "instance"])
    parser.add_argument("--activation", default="silu", choices=["relu", "silu", "gelu", "leaky_relu"])
    parser.add_argument("--hrnet-width", default=32, type=int)
    parser.add_argument("--hrnet-norm", default="batch", choices=["batch", "group", "instance"])
    parser.add_argument("--hrnet-activation", default="relu", choices=["relu", "silu", "gelu", "leaky_relu"])
    parser.add_argument("--ocr-mid-channels", default=512, type=int)
    parser.add_argument("--ocr-key-channels", default=256, type=int)
    parser.add_argument("--unet3plus-base-channels", default=32, type=int)
    parser.add_argument("--unet3plus-cat-channels", default=32, type=int)
    parser.add_argument("--unet3plus-dropout", default=0.1, type=float)
    parser.add_argument("--deep-supervision", action="store_true", default=True)
    parser.add_argument("--no-edge-head", action="store_true")
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--mask-mode", default="color", choices=["color", "index", "binary"], help="How masks encode classes")
    parser.add_argument("--resume", default="", help="Checkpoint path")
    parser.add_argument("--init-checkpoint", default="", help="Load only model weights from this checkpoint and start a fresh fine-tuning run")
    parser.add_argument("--resume-lr", default=None, type=float, help="Override optimizer LR after loading a checkpoint")
    parser.add_argument("--region-loss", default="dice", choices=["dice", "tversky"], help="Region overlap loss used with cross entropy")
    parser.add_argument("--dice-weight", default=0.5, type=float, help="Weight of Dice/Tversky region loss inside CE+region loss")
    parser.add_argument("--tversky-alpha", default=0.3, type=float, help="False-positive penalty for Tversky loss")
    parser.add_argument("--tversky-beta", default=0.7, type=float, help="False-negative penalty for Tversky loss")
    parser.add_argument("--tversky-gamma", default=1.0, type=float, help="Focal exponent for Tversky loss")
    parser.add_argument("--focal-weight", default=0.25, type=float)
    parser.add_argument("--boundary-weight", default=0.1, type=float)
    parser.add_argument("--edge-weight", default=0.2, type=float)
    parser.add_argument("--aux-weight", default=0.4, type=float)
    parser.add_argument("--ohem-start-epoch", default=-1, type=int, help="Enable online hard example mining from this epoch; -1 disables it")
    parser.add_argument("--ohem-ratio", default=1.0, type=float, help="Fraction of hardest valid pixels kept for CE/BCE after OHEM starts")
    parser.add_argument("--ohem-min-kept", default=4096, type=int, help="Minimum valid pixels kept by OHEM")
    parser.add_argument("--lovasz-weight", default=0.0, type=float, help="Lovasz-Softmax loss weight for late IoU fine-tuning")
    parser.add_argument("--lovasz-start-epoch", default=-1, type=int, help="Enable Lovasz-Softmax from this epoch; -1 disables it")
    parser.add_argument("--overwrite-output", action="store_true", help="Allow starting from scratch in an existing checkpoint folder")
    parser.add_argument("--log-file", default="", help="Append console output and tracebacks to this file")
    parser.add_argument("--plot-interval", default=0, type=int, help="Plot metrics every N epochs. Use 0 to disable plotting during training.")
    parser.add_argument("--no-clearml", action="store_true", help="Disable ClearML experiment logging")
    parser.add_argument("--clearml-project", default="LCMS Crack Segmentation", help="ClearML project name")
    parser.add_argument("--clearml-task-name", default="", help="ClearML task name. Defaults to the output folder name.")
    parser.add_argument("--clearml-tags", default="", help="Comma-separated ClearML tags")
    parser.add_argument("--clearml-config-file", default="clearml.conf", help="ClearML SDK config file")
    parser.add_argument("--clearml-output-uri", default="", help="Optional ClearML artifact/model output URI")
    parser.add_argument("--clearml-offline", action="store_true", help="Run ClearML in offline mode")
    parser.add_argument("--no-clearml-models", action="store_true", help="Do not upload checkpoint weights as ClearML output models")
    parser.add_argument("--early-stopping-patience", default=20, type=int, help="Stop after N epochs without validation mean_dice improvement. Use 0 to disable.")
    parser.add_argument(
        "--monitor-metric",
        default="mean_dice",
        help="Metric key used to save an additional best_<metric>.pth checkpoint, e.g. mean_iou, val_loss, class_3_precision.",
    )
    parser.add_argument(
        "--monitor-mode",
        default="auto",
        choices=["auto", "max", "min"],
        help="Whether the monitor metric should increase or decrease. Auto uses min for loss metrics and max otherwise.",
    )
    args = parser.parse_args()
    if args.config:
        args = apply_config_defaults(parser, args, load_yaml_config(args.config))
    return args


def parse_class_sampling_weights(value: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    if not value.strip():
        return weights
    for item in value.split(","):
        class_id, weight = item.split(":", 1)
        weights[int(class_id.strip())] = float(weight.strip())
    return weights


def parse_loss_class_weights(value: str, num_classes: int) -> list[float] | None:
    if not value.strip():
        return None
    weights = [1.0] * num_classes
    for item in value.split(","):
        class_id, weight = item.split(":", 1)
        class_idx = int(class_id.strip())
        if class_idx < 0 or class_idx >= num_classes:
            raise ValueError(f"loss class weight id {class_idx} is outside num_classes={num_classes}")
        weights[class_idx] = float(weight.strip())
    return weights


def resolve_monitor_mode(metric_name: str, mode: str) -> str:
    if mode != "auto":
        return mode
    return "min" if "loss" in metric_name.lower() else "max"


def is_monitor_improved(value: float, best_value: float | None, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    if best_value is None or not math.isfinite(best_value):
        return True
    if mode == "min":
        return value < best_value
    return value > best_value


def checkpoint_metric_name(metric_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_name).strip("._") or "metric"


def load_compatible_model_weights(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> tuple[int, list[str], list[str]]:
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    missing = [key for key in model_state if key not in compatible]
    skipped = [key for key, value in state.items() if key not in model_state or tuple(value.shape) != tuple(model_state[key].shape)]
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return len(compatible), missing, skipped


def main() -> None:
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    if not args.data_path:
        raise ValueError("Provide --data-path or data_path in --config")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_run_files = [output_dir / name for name in ("latest.pth", "best.pth", "metrics.csv")]
    if not args.resume and not args.overwrite_output and any(path.exists() for path in existing_run_files):
        existing = ", ".join(str(path) for path in existing_run_files if path.exists())
        raise RuntimeError(
            "Refusing to start from scratch in an existing output folder because it contains "
            f"{existing}. Use --resume to continue, choose a new --output-dir, or pass "
            "--overwrite-output if you intentionally want to replace this run."
        )
    log_path = Path(args.log_file) if args.log_file else output_dir / "train.log"
    logger = setup_logging(log_path, "lcms_unetpp.train")
    log_environment(logger)
    logger.info("command args: %s", json.dumps(vars(args), indent=2))
    config_json_path = output_dir / "config.json"
    config_json_path.write_text(json.dumps(vars(args), indent=2))

    clearml_run = ClearMLRun(
        enabled=not args.no_clearml,
        project_name=args.clearml_project,
        task_name=args.clearml_task_name or output_dir.name,
        config_file=args.clearml_config_file,
        output_uri=args.clearml_output_uri,
        tags=args.clearml_tags,
        offline=args.clearml_offline,
    )
    clearml_run.connect_parameters("hyperparameters", vars(args))
    clearml_run.upload_artifact("resolved_config_json", config_json_path)
    if args.config:
        clearml_run.connect_configuration(args.config, "training_yaml")
        clearml_run.upload_artifact("training_yaml", args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    class_sampling_weights = parse_class_sampling_weights(args.class_sampling_weights)
    loss_class_weights = parse_loss_class_weights(args.loss_class_weights, args.num_classes)
    if args.no_patch_training:
        train_ds = LCMSCrackDataset(args.data_path, "TRAIN", TrainTransform(args.height, args.width), mask_mode=args.mask_mode)
    else:
        patch_train_ds = LCMSCrackPatchDataset(
            args.data_path,
            "TRAIN",
            BiasedPatchTrainTransform(args.patch_height, args.patch_width),
            mask_mode=args.mask_mode,
            samples_per_image=args.samples_per_image,
            anomaly_ratio=args.anomaly_ratio,
            random_crop_ratio=args.random_crop_ratio,
            class_sampling_weights=class_sampling_weights,
            min_target_pixels=args.min_target_pixels,
            target_crop_attempts=args.target_crop_attempts,
            foreground_extension_ratio=args.foreground_extension_ratio,
        )
        if args.full_image_ratio > 0:
            full_train_ds = LCMSCrackDataset(
                args.data_path,
                "TRAIN",
                TrainTransform(args.height, args.width),
                mask_mode=args.mask_mode,
            )
            train_ds = MixedFullPatchDataset(patch_train_ds, full_train_ds, args.full_image_ratio)
        else:
            train_ds = patch_train_ds
    val_ds = LCMSCrackDataset(args.data_path, "VAL", EvalTransform(args.height, args.width), mask_mode=args.mask_mode)
    logger.info(
        "dataset train_samples=%d val_samples=%d patch_training=%s patch_size=%sx%s anomaly_ratio=%.2f random_crop_ratio=%.2f full_image_ratio=%.2f class_sampling_weights=%s loss_class_weights=%s",
        len(train_ds),
        len(val_ds),
        not args.no_patch_training,
        args.patch_height,
        args.patch_width,
        args.anomaly_ratio,
        args.random_crop_ratio,
        args.full_image_ratio,
        class_sampling_weights,
        loss_class_weights,
    )
    clearml_run.connect_parameters(
        "dataset",
        {
            "data_path": args.data_path,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "patch_training": not args.no_patch_training,
            "patch_height": args.patch_height,
            "patch_width": args.patch_width,
            "class_sampling_weights": class_sampling_weights,
            "loss_class_weights": loss_class_weights,
            "foreground_extension_ratio": args.foreground_extension_ratio,
            "mask_mode": args.mask_mode,
            "num_classes": args.num_classes,
        },
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": LCMSCrackDataset.collate_fn,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    )

    checkpoint = None
    init_checkpoint = None
    resume_model_args = {}
    if args.resume:
        logger.info("loading checkpoint metadata=%s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        resume_model_args = checkpoint.get("args", {})
    elif args.init_checkpoint:
        logger.info("loading init checkpoint metadata=%s", args.init_checkpoint)
        init_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        resume_model_args = init_checkpoint.get("args", {})

    model_args = {**vars(args)}
    if args.resume:
        model_args.update(resume_model_args)
    model = build_model_from_args(
        model_args,
        pretrained=False if args.resume or args.init_checkpoint else not args.no_pretrained,
    ).to(device)
    logger.info("model architecture=%s args=%s", model_args.get("model", "unetpp"), json.dumps(model_args, indent=2))

    criterion = CrackSegmentationLoss(
        class_weights=loss_class_weights,
        dice_weight=args.dice_weight,
        region_loss=args.region_loss,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
        focal_weight=args.focal_weight,
        boundary_weight=args.boundary_weight,
        edge_weight=args.edge_weight,
        aux_weight=args.aux_weight,
        ohem_start_epoch=args.ohem_start_epoch,
        ohem_ratio=args.ohem_ratio,
        ohem_min_kept=args.ohem_min_kept,
        lovasz_weight=args.lovasz_weight,
        lovasz_start_epoch=args.lovasz_start_epoch,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_warmup_cosine_scheduler(optimizer, len(train_loader), args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    ema = ModelEMA(model) if args.ema else None

    start_epoch = 0
    best_dice = 0.0
    monitor_mode = resolve_monitor_mode(args.monitor_metric, args.monitor_mode)
    monitor_best = None
    epochs_without_improvement = 0
    if args.resume:
        logger.info("loading checkpoint=%s", args.resume)
        model.load_state_dict(checkpoint["model"])
        if ema is not None:
            if checkpoint.get("ema_model") is not None:
                ema.module.load_state_dict(checkpoint["ema_model"])
                logger.info("loaded ema_model from checkpoint")
            else:
                ema.module.load_state_dict(model.state_dict())
                logger.warning("checkpoint has no ema_model; initialized EMA from model weights")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if args.resume_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.resume_lr
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [args.resume_lr for _ in scheduler.base_lrs]
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = [args.resume_lr for _ in optimizer.param_groups]
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_dice = checkpoint.get("best_dice", 0.0)
        if checkpoint.get("monitor_metric") == args.monitor_metric and checkpoint.get("monitor_mode", monitor_mode) == monitor_mode:
            monitor_best = checkpoint.get("monitor_best")
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        logger.info(
            "resumed start_epoch=%d best_dice=%.6f monitor_metric=%s monitor_best=%s monitor_mode=%s epochs_without_improvement=%d",
            start_epoch,
            best_dice,
            args.monitor_metric,
            monitor_best,
            monitor_mode,
            epochs_without_improvement,
        )
    elif init_checkpoint is not None:
        loaded_count, missing, skipped = load_compatible_model_weights(model, init_checkpoint["model"])
        logger.info(
            "initialized compatible model weights from %s loaded_keys=%d missing_keys=%d skipped_keys=%d",
            args.init_checkpoint,
            loaded_count,
            len(missing),
            len(skipped),
        )
        if missing:
            logger.info("init missing_keys=%s", missing)
        if skipped:
            logger.info("init skipped_keys=%s", skipped)
    monitor_ckpt_name = f"best_{checkpoint_metric_name(args.monitor_metric)}.pth"
    logger.info("monitor checkpoint metric=%s mode=%s output=%s", args.monitor_metric, monitor_mode, output_dir / monitor_ckpt_name)

    history_path = output_dir / "metrics.csv"
    heartbeat_path = output_dir / "heartbeat.json"
    if args.resume:
        truncate_history(history_path, start_epoch)
    metric_classes = 2 if args.mask_mode == "binary" or args.num_classes == 1 else args.num_classes
    for epoch in range(start_epoch, args.epochs):
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        print(f"[{timestamp()}] Epoch {epoch}/{args.epochs - 1}", flush=True)
        print(f"[{timestamp()}]   train: start", flush=True)

        def write_heartbeat(step: int, loss: float, lr: float) -> None:
            write_json(
                heartbeat_path,
                {
                    "time": timestamp(),
                    "phase": "train",
                    "epoch": epoch,
                    "step": step,
                    "steps_per_epoch": len(train_loader),
                    "loss": loss,
                    "lr": lr,
                },
            )

        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            scheduler,
            ema=ema,
            grad_clip=args.grad_clip,
            progress_callback=write_heartbeat,
        )
        print(f"[{timestamp()}]   train: done", flush=True)
        eval_model = ema.module if ema is not None else model
        print(f"[{timestamp()}]   val: start", flush=True)
        write_json(heartbeat_path, {"time": timestamp(), "phase": "val", "epoch": epoch})

        def write_val_heartbeat(step: int, loss: float) -> None:
            write_json(
                heartbeat_path,
                {
                    "time": timestamp(),
                    "phase": "val",
                    "epoch": epoch,
                    "step": step,
                    "steps_per_epoch": len(val_loader),
                    "loss": loss,
                },
            )

        val_stats = evaluate(
            eval_model,
            val_loader,
            criterion,
            device,
            metric_classes=metric_classes,
            progress_callback=write_val_heartbeat,
        )
        print(f"[{timestamp()}]   val: done", flush=True)
        row = {"epoch": epoch, **train_stats, **val_stats}
        print(f"[{timestamp()}]   metrics: writing", flush=True)
        write_json(heartbeat_path, {"time": timestamp(), "phase": "metrics", "epoch": epoch})
        append_history(history_path, row)
        clearml_run.report_metrics(row, iteration=epoch)
        print(row, flush=True)
        if args.monitor_metric not in row:
            available = ", ".join(sorted(row.keys()))
            raise KeyError(f"Unknown --monitor-metric '{args.monitor_metric}'. Available metrics: {available}")

        improved = val_stats["mean_dice"] > best_dice
        monitor_value = float(row[args.monitor_metric])
        monitor_improved = is_monitor_improved(monitor_value, monitor_best, monitor_mode)
        if improved:
            best_dice = val_stats["mean_dice"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if monitor_improved:
            monitor_best = monitor_value
        logger.info(
            "early_stopping epoch=%d improved=%s best_dice=%.6f current_mean_dice=%.6f monitor_metric=%s monitor_value=%.6f monitor_improved=%s monitor_best=%.6f wait=%d patience=%d",
            epoch,
            improved,
            best_dice,
            val_stats["mean_dice"],
            args.monitor_metric,
            monitor_value,
            monitor_improved,
            monitor_best if monitor_best is not None else float("nan"),
            epochs_without_improvement,
            args.early_stopping_patience,
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema.module.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
            "monitor_metric": args.monitor_metric,
            "monitor_mode": monitor_mode,
            "monitor_best": monitor_best,
            "monitor_value": monitor_value,
            "epochs_without_improvement": epochs_without_improvement,
            "args": vars(args),
        }
        print(f"[{timestamp()}]   checkpoint: saving latest", flush=True)
        write_json(heartbeat_path, {"time": timestamp(), "phase": "checkpoint_latest", "epoch": epoch})
        torch.save(checkpoint, output_dir / "latest.pth")
        if not args.no_clearml_models:
            clearml_run.update_model(
                "latest",
                output_dir / "latest.pth",
                iteration=epoch,
                metadata={"epoch": epoch, "mean_dice": val_stats["mean_dice"], args.monitor_metric: monitor_value},
            )
        clearml_run.upload_artifact("latest_checkpoint", output_dir / "latest.pth")
        clearml_run.upload_artifact("metrics_csv", history_path)
        clearml_run.upload_artifact("heartbeat_json", heartbeat_path)
        if improved:
            print(f"[{timestamp()}]   checkpoint: saving best", flush=True)
            write_json(heartbeat_path, {"time": timestamp(), "phase": "checkpoint_best", "epoch": epoch})
            torch.save(checkpoint, output_dir / "best.pth")
            if not args.no_clearml_models:
                clearml_run.update_model(
                    "best_mean_dice",
                    output_dir / "best.pth",
                    iteration=epoch,
                    metadata={"epoch": epoch, "mean_dice": best_dice},
                )
            clearml_run.upload_artifact("best_checkpoint", output_dir / "best.pth")
            print(f"[{timestamp()}] Best checkpoint saved: mean_dice={best_dice:.4f}", flush=True)
        if monitor_improved:
            print(f"[{timestamp()}]   checkpoint: saving {monitor_ckpt_name}", flush=True)
            write_json(
                heartbeat_path,
                {
                    "time": timestamp(),
                    "phase": "checkpoint_monitor",
                    "epoch": epoch,
                    "monitor_metric": args.monitor_metric,
                    "monitor_value": monitor_value,
                },
            )
            torch.save(checkpoint, output_dir / monitor_ckpt_name)
            if not args.no_clearml_models:
                clearml_run.update_model(
                    f"best_{checkpoint_metric_name(args.monitor_metric)}",
                    output_dir / monitor_ckpt_name,
                    iteration=epoch,
                    metadata={"epoch": epoch, args.monitor_metric: monitor_value, "monitor_mode": monitor_mode},
                )
            clearml_run.upload_artifact(f"best_{checkpoint_metric_name(args.monitor_metric)}_checkpoint", output_dir / monitor_ckpt_name)
            print(
                f"[{timestamp()}] Monitor checkpoint saved: {args.monitor_metric}={monitor_value:.4f} mode={monitor_mode}",
                flush=True,
            )
        if args.plot_interval > 0 and (epoch + 1) % args.plot_interval == 0:
            print(f"[{timestamp()}]   metrics: plotting", flush=True)
            write_json(heartbeat_path, {"time": timestamp(), "phase": "plot", "epoch": epoch})
            try:
                plot_history(history_path, output_dir / "metrics.png")
                clearml_run.upload_artifact("metrics_plot", output_dir / "metrics.png")
            except Exception:
                logger.exception("plot_history failed epoch=%d", epoch)
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"[{timestamp()}] Early stopping: no mean_dice improvement for "
                f"{epochs_without_improvement} epochs; best_dice={best_dice:.4f}",
                flush=True,
            )
            write_json(
                heartbeat_path,
                {
                    "time": timestamp(),
                    "phase": "early_stopping",
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "epochs_without_improvement": epochs_without_improvement,
                },
            )
            break
    clearml_run.upload_artifact("train_log", log_path)
    clearml_run.upload_artifact("final_metrics_csv", history_path)
    clearml_run.upload_artifact("final_heartbeat_json", heartbeat_path)
    clearml_run.close()


if __name__ == "__main__":
    main()
