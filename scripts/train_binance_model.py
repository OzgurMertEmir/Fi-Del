#!/usr/bin/env python3
"""Train and evaluate a baseline model on Binance trade data.

This script demonstrates the ML integration with Binance data:
1. Downloads trade data from Binance
2. Computes order flow and microstructure features
3. Trains a logistic regression baseline
4. Evaluates on held-out validation data
5. Shows feature importance

Usage:
    python scripts/train_binance_model.py
"""

import logging

from myproj.binance_data import (
    build_binance_ml_dataset,
    run_binance_baseline,
    get_feature_importance,
)
from myproj.ml.models import train_price_move_model, evaluate_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    """Run the training pipeline."""

    # Configuration
    symbol = "BTCUSDT"
    start_date = "2024-12-01"
    end_date = "2024-12-07"  # Use a week of data for demo
    interval = "1s"
    label_horizon = 30  # Predict 30 seconds ahead
    deadzone_bps = 2.0  # 2 bps deadzone for classification

    print("=" * 70)
    print("Binance ML Pipeline Demo")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print(f"Label horizon: {label_horizon} bars")
    print(f"Deadzone: {deadzone_bps} bps")
    print()

    # Option 1: Quick baseline run
    print("Running baseline model...")
    print("-" * 40)

    metrics, dataset = run_binance_baseline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        label_horizon=label_horizon,
        deadzone_bps=deadzone_bps,
        include_advanced_features=True,
    )

    print("\nDataset Summary:")
    summary = dataset.summary()
    for k, v in summary.items():
        if k == "train_label_dist":
            print(f"  {k}:")
            for label, count in sorted(v.items()):
                print(f"    class {label}: {count:,}")
        else:
            print(f"  {k}: {v}")

    print("\nValidation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Feature importance
    print("\nTop 20 Most Important Features:")
    print("-" * 40)

    # Re-train model to get coefficients
    model = train_price_move_model(dataset.X_train, dataset.y_train)
    importance = get_feature_importance(model, dataset.feature_cols, top_k=20)

    for i, (name, imp) in enumerate(importance, 1):
        print(f"  {i:2d}. {name:<30s} {imp:.4f}")

    # Option 2: Custom dataset building with more control
    print("\n" + "=" * 70)
    print("Custom Dataset Building Example")
    print("=" * 70)

    # Build dataset with custom parameters
    custom_dataset = build_binance_ml_dataset(
        symbol="BTCUSDT",
        start_date=start_date,
        end_date=end_date,
        interval="5s",  # 5-second bars
        label_horizon=12,  # 12 bars = 60 seconds ahead
        deadzone_bps=3.0,
        rolling_windows=[6, 12, 60, 180],  # Custom windows for 5s bars
        include_advanced_features=True,
        val_fraction=0.15,
        test_fraction=0.15,
        embargo_bars=24,  # 2 minutes gap
    )

    print("\nCustom Dataset Summary:")
    for k, v in custom_dataset.summary().items():
        if k == "train_label_dist":
            print(f"  {k}:")
            for label, count in sorted(v.items()):
                print(f"    class {label}: {count:,}")
        else:
            print(f"  {k}: {v}")

    # Train and evaluate
    model = train_price_move_model(custom_dataset.X_train, custom_dataset.y_train)
    val_metrics = evaluate_classifier(model, custom_dataset.X_val, custom_dataset.y_val)

    print("\nValidation Metrics (5s bars):")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    if custom_dataset.X_test is not None:
        test_metrics = evaluate_classifier(model, custom_dataset.X_test, custom_dataset.y_test)
        print("\nTest Metrics (5s bars):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
