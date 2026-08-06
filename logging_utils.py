"""Shared logging helpers for LCMS training and inference scripts."""
from __future__ import annotations

import atexit
import faulthandler
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


class Tee:
    """Write stream output to multiple destinations."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    @property
    def closed(self) -> bool:
        return all(getattr(stream, "closed", False) for stream in self.streams)

    def write(self, data: str) -> int:
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            if not getattr(stream, "closed", False):
                stream.flush()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def setup_logging(log_path: str | Path, logger_name: str = "lcms_unetpp") -> logging.Logger:
    """Mirror stdout/stderr to a log file and install exception/fault logging."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)

    faulthandler.enable(file=log_file, all_threads=True)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    atexit.register(log_file.close)

    base_logger = logging.getLogger("lcms_unetpp")
    base_logger.setLevel(logging.INFO)
    base_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    for stream in (sys.stdout,):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        base_logger.addHandler(handler)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    def log_exception(exc_type, exc, tb):
        logger.critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = log_exception
    logger.info("logging to %s", log_path)
    return logger


def log_environment(logger: logging.Logger) -> None:
    """Record runtime details useful for debugging crashes and restarts."""
    logger.info("python=%s", sys.version.replace("\n", " "))
    logger.info("platform=%s", platform.platform())
    logger.info("torch=%s cuda_available=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            logger.info(
                "cuda_device[%d]=%s total_memory_gb=%.2f capability=%s",
                idx,
                props.name,
                props.total_memory / (1024**3),
                f"{props.major}.{props.minor}",
            )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
