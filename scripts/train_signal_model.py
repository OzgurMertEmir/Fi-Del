#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

"""
train_signal_model.py

Train a model to predict whether a signal will result in a fill.

This script uses the parsed log data (signals + fills) to train a classifier
that predicts:
- had_fill: Whether a fill occurred within the time window after a signal

Example:

```bash
python scripts/train_signal_model.py
python scripts/train_signal_model.py --symbol BTCUSDT --window 30
```
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from myproj.config import PROCESSED_DATA_DIR
from myproj.data import (
    build_log_ml_dataset,
    get_dataset_summary,
    DatasetConfig,
)
from myproj.ml.models import train_price_move_model, evaluate_classifier


def train_and_evaluate(
    data_dir: Path,
    symbol: str | None = None,
    fill_window_sec: float = 60.0,
    val_fraction: float = 0.2,
    output_dir: Path | None = None,
) -> dict:
    """Train and evaluate a signal model.

    Args:
        data_dir: Directory with processed signals and fills
        symbol: Filter to specific symbol (None = all)
        fill_window_sec: Time window to look for fills after signal
        val_fraction: Fraction of data for validation
        output_dir: Directory to save results

    Returns:
        Dictionary with metrics and model info
    """
    config = DatasetConfig(
        fill_window_sec=fill_window_sec,
        val_fraction=val_fraction,
    )

    print("Building dataset...")
    print(f"  Data dir: {data_dir}")
    print(f"  Symbol: {symbol or 'all'}")
    print(f"  Fill window: {fill_window_sec}s")

    dataset = build_log_ml_dataset(
        data_dir=data_dir,
        config=config,
        symbol=symbol,
        label_col="had_fill",
    )

    summary = get_dataset_summary(dataset)

    print("\nDataset summary:")
    print(f"  Total signals: {summary['total_rows']:,}")
    print(f"  Features: {summary['n_features']}")
    print(f"  Train size: {summary['train_size']:,}")
    print(f"  Val size: {summary['val_size']:,}")
    print(f"  Label distribution: {summary['label_distribution']}")

    # Compute class balance
    total_train = sum(summary['label_distribution'].values())
    fill_rate = summary['label_distribution'].get(1, 0) / total_train
    print(f"  Fill rate: {fill_rate:.1%}")

    print("\nTraining model...")
    model = train_price_move_model(dataset.X_train, dataset.y_train)

    print("Evaluating on validation set...")
    metrics = evaluate_classifier(model, dataset.X_val, dataset.y_val)

    print("\nValidation metrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    # Feature importance
    if hasattr(model, 'named_steps'):
        lr = model.named_steps.get('logisticregression')
        if lr is not None and hasattr(lr, 'coef_'):
            coef = lr.coef_.ravel()
            importance = sorted(
                zip(dataset.feature_cols, np.abs(coef)),
                key=lambda x: -x[1]
            )
            print("\nTop features by importance:")
            for name, imp in importance[:10]:
                print(f"  {name}: {imp:.4f}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "symbol": symbol or "all",
            "fill_window_sec": fill_window_sec,
            "val_fraction": val_fraction,
            "summary": summary,
            "metrics": metrics,
        }

        results_path = output_dir / "signal_model_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

    return {
        "summary": summary,
        "metrics": metrics,
        "model": model,
        "dataset": dataset,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a signal prediction model")
    parser.add_argument(
        "--data-dir",
        default=str(PROCESSED_DATA_DIR),
        help="Directory with processed signals and fills",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Filter to specific symbol (default: all symbols)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=60.0,
        help="Time window in seconds to look for fills after signal (default: 60)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to save results (default: artifacts)",
    )
    args = parser.parse_args()

    train_and_evaluate(
        data_dir=Path(args.data_dir),
        symbol=args.symbol,
        fill_window_sec=args.window,
        val_fraction=args.val_fraction,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
