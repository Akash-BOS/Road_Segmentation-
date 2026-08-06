"""Optional ClearML experiment tracking helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("lcms_unetpp.clearml")


class ClearMLRun:
    """Thin wrapper that keeps ClearML optional for local development."""

    def __init__(
        self,
        *,
        enabled: bool,
        project_name: str,
        task_name: str,
        config_file: str | Path = "",
        output_uri: str = "",
        tags: str = "",
        offline: bool = False,
    ) -> None:
        self.enabled = enabled
        self.task = None
        self.logger = None
        self.output_models: dict[str, Any] = {}
        self._clearml = None
        if not enabled:
            LOGGER.info("ClearML logging disabled")
            return

        try:
            from clearml import OutputModel, Task
        except ImportError:
            LOGGER.warning("ClearML package is not installed; continuing without ClearML logging")
            return

        self._clearml = {"OutputModel": OutputModel, "Task": Task}
        config_path = Path(config_file) if config_file else Path("clearml.conf")
        if config_path.exists() and "CLEARML_CONFIG_FILE" not in os.environ:
            os.environ["CLEARML_CONFIG_FILE"] = str(config_path.resolve())
            LOGGER.info("using ClearML config file %s", config_path)
        if offline:
            Task.set_offline(offline_mode=True)
            LOGGER.info("ClearML offline mode enabled")

        try:
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                output_uri=output_uri or None,
                auto_connect_arg_parser=False,
                auto_connect_frameworks=False,
            )
            self.logger = self.task.get_logger()
            parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
            if parsed_tags:
                self.task.add_tags(parsed_tags)
            LOGGER.info("ClearML task initialized project=%s task=%s id=%s", project_name, task_name, self.task.id)
        except Exception:
            LOGGER.exception("ClearML initialization failed; continuing without ClearML logging")
            self.task = None
            self.logger = None

    @property
    def active(self) -> bool:
        return self.task is not None

    def connect_parameters(self, name: str, params: dict[str, Any]) -> None:
        if not self.active:
            return
        try:
            self.task.connect(_sanitize(params), name=name)
        except Exception:
            LOGGER.exception("ClearML failed to connect parameters name=%s", name)

    def connect_configuration(self, path: str | Path, name: str) -> None:
        if not self.active:
            return
        path = Path(path)
        if not path.exists():
            return
        try:
            self.task.connect_configuration(str(path), name=name)
        except Exception:
            LOGGER.exception("ClearML failed to connect configuration path=%s", path)

    def upload_artifact(self, name: str, path: str | Path) -> None:
        if not self.active:
            return
        path = Path(path)
        if not path.exists():
            return
        try:
            self.task.upload_artifact(name=name, artifact_object=str(path))
        except Exception:
            LOGGER.exception("ClearML failed to upload artifact name=%s path=%s", name, path)

    def report_metrics(self, metrics: dict[str, Any], iteration: int) -> None:
        if self.logger is None:
            return
        for key, value in metrics.items():
            if key == "epoch" or not isinstance(value, (int, float)):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            title = _metric_title(key)
            series = key.removeprefix("train_").removeprefix("val_")
            try:
                self.logger.report_scalar(title=title, series=series, value=numeric, iteration=iteration)
            except Exception:
                LOGGER.exception("ClearML failed to report metric key=%s epoch=%d", key, iteration)

    def update_model(self, name: str, path: str | Path, *, iteration: int, metadata: dict[str, Any] | None = None) -> None:
        if not self.active or self._clearml is None:
            return
        path = Path(path)
        if not path.exists():
            return
        try:
            model = self.output_models.get(name)
            if model is None:
                model = self._clearml["OutputModel"](task=self.task, name=name, framework="PyTorch")
                self.output_models[name] = model
            if metadata:
                for key, value in metadata.items():
                    model.set_metadata(str(key), str(value))
            model.update_weights(weights_filename=str(path), iteration=iteration, auto_delete_file=False)
        except Exception:
            LOGGER.exception("ClearML failed to update model name=%s path=%s", name, path)

    def close(self) -> None:
        if not self.active:
            return
        try:
            self.task.close()
        except Exception:
            LOGGER.exception("ClearML task close failed")


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): ("***" if _is_secret_key(str(key)) else _sanitize(child)) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(child) for child in value]
    return value


def _is_secret_key(key: str) -> bool:
    key = key.lower()
    return any(marker in key for marker in ("secret", "password", "token", "access_key"))


def _metric_title(key: str) -> str:
    if key.startswith("train_") or key == "lr":
        return "train"
    if key.startswith("val_"):
        return "validation"
    if key.startswith("class_"):
        parts = key.split("_")
        if len(parts) >= 3:
            return f"class_{parts[1]}"
    return "validation"
