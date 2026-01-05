#!/usr/bin/env python
"""Train a simple price-move classifier on processed data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from myproj.ml.pipeline_price import run_price_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train price move prototype model.")
    parser.add_argument("--symbol", help="Symbol to filter (default: all)", default=None)
    parser.add_argument("--bucket", help="Time bucket (e.g., 1s)", default="1s")
    parser.add_argument("--horizon", help="Horizon in buckets (default: 5)", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (default 0.2)")
    args = parser.parse_args()

    metrics, feature_cols = run_price_pipeline(
        symbol=args.symbol,
        bucket=args.bucket,
        horizon_steps=args.horizon,
        val_fraction=args.val_fraction,
    )
    print("Price-move metrics:", metrics)
    print(f"Used {len(feature_cols)} features.")


if __name__ == "__main__":
    main()
