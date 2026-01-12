#!/usr/bin/env python3
"""Build ML-ready dataset from Binance public data.

This script demonstrates the full pipeline:
1. Downloads trade data from Binance
2. Aggregates into time bars
3. Computes order flow features
4. Adds labels for price direction prediction
5. Splits into train/val/test sets
6. Saves to parquet format

Usage:
    python scripts/build_binance_dataset.py

Modify the configuration below to customize the dataset.
"""

import logging
from datetime import date
from pathlib import Path

from myproj.binance_data import BinancePipelineConfig, run_binance_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    """Build the dataset."""

    # Configure the pipeline
    config = BinancePipelineConfig(
        # Symbols to process
        symbols=["BTCUSDT", "ETHUSDT"],

        # Date range
        start_date=date(2024, 12, 1),
        end_date=date(2024, 12, 31),

        # Market type: "spot", "um" (USD-M futures), "cm" (COIN-M futures)
        market_type="spot",

        # Data type: "trades" or "aggTrades"
        data_type="trades",

        # Aggregation interval
        interval="1s",

        # Rolling window sizes for feature computation
        rolling_windows=[5, 15, 60, 300],

        # Prediction horizons (in bars)
        horizons=[5, 30, 60],

        # Deadzone for direction classification (basis points)
        deadzone_bps=5.0,

        # Train/val/test split ratios
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,

        # Embargo period between splits (bars)
        embargo_bars=60,

        # Paths
        raw_data_dir=Path("data/binance"),
        output_dir=Path("data/processed"),

        # Download data if not present
        download_if_missing=True,

        # Enable cross-asset features (merge all symbols)
        enable_cross_asset=False,
    )

    # Run the pipeline
    output_paths = run_binance_pipeline(config)

    # Print results
    print("\n" + "=" * 60)
    print("Dataset build complete!")
    print("=" * 60)

    for symbol, paths in output_paths.items():
        print(f"\n{symbol}:")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
