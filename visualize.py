"""Visualize LCMS UNet++ metrics CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lcms_unetpp.logging_utils import setup_logging
from lcms_unetpp.metrics import plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all metrics from training CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="metrics.png")
    parser.add_argument("--log-file", default="", help="Append console output and tracebacks to this file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file) if args.log_file else Path(args.out).with_suffix(".log")
    logger = setup_logging(log_path, "lcms_unetpp.visualize")
    logger.info("plot start csv=%s out=%s", args.csv, args.out)
    plot_history(args.csv, args.out)
    logger.info("plot done out=%s", args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
