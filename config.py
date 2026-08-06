"""Configuration helpers for LCMS training/evaluation scripts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    section_prefix = {
        "model": "",
        "data": "",
        "training": "",
        "loss": "",
        "patch": "",
        "checkpoint": "",
        "logging": "",
        "clearml": "",
    }
    aliases = {
        "name": "model",
        "model_name": "model",
        "path": "data_path",
        "checkpoint_path": "checkpoint",
        "init": "init_checkpoint",
    }
    for key, value in data.items():
        key_norm = str(key).replace("-", "_")
        if isinstance(value, dict) and key_norm in section_prefix:
            for child_key, child_value in value.items():
                child_norm = str(child_key).replace("-", "_")
                flat[aliases.get(child_norm, child_norm)] = child_value
        else:
            flat[aliases.get(key_norm, key_norm)] = value
    return flat


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for --config. Install pyyaml or omit --config.") from exc
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top of {config_path}")
    return _flatten_config(data)


def apply_config_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    """Apply config values only for arguments left at parser defaults."""
    merged = vars(args).copy()
    for key, value in config.items():
        if key not in merged:
            continue
        default = parser.get_default(key)
        current = merged[key]
        if current == default or current in (None, ""):
            merged[key] = value
    return argparse.Namespace(**merged)
