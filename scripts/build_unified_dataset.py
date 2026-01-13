#!/usr/bin/env python
# ruff: noqa: E402
"""
build_unified_dataset.py

Build unified ML datasets from multiple data sources:
- Order book snapshots (.store files)
- Trading signals (parsed logs)
- Trade fills (parsed logs)

This script aligns all data sources by timestamp and symbol,
creates features, and outputs ML-ready datasets.

Example:
    python scripts/build_unified_dataset.py
    python scripts/build_unified_dataset.py --symbol BTCUSDT --interval 1s
    python scripts/build_unified_dataset.py --all-symbols
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from myproj.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from myproj.unified_data import (
    MultiSourceLoader,
    MLDatasetConfig,
    build_ml_dataset,
    get_dataset_summary,
    save_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Build unified multi-source ML datasets")
    parser.add_argument(
        "--raw-dir",
        default=str(RAW_DATA_DIR),
        help="Directory with raw data (.store files)",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(PROCESSED_DATA_DIR),
        help="Directory with processed data (signals, fills)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Single symbol to process (default: first available)",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Process all available symbols",
    )
    parser.add_argument(
        "--interval",
        default="1s",
        help="Resampling interval (default: 1s)",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=30,
        help="Look-ahead periods for label (default: 30)",
    )
    parser.add_argument(
        "--label-type",
        default="fill_occurred",
        choices=["fill_occurred", "price_direction"],
        help="Type of prediction label (default: fill_occurred)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/unified",
        help="Output directory for datasets (default: data/unified)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save datasets to disk",
    )
    parser.add_argument(
        "--include-binance",
        action="store_true",
        help="Include Binance public data (downloads if needed)",
    )
    parser.add_argument(
        "--binance-dir",
        default="data/binance",
        help="Directory for Binance data (default: data/binance)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("UNIFIED DATASET BUILDER")
    print("=" * 60)
    print(f"Raw data dir: {raw_dir}")
    print(f"Processed dir: {processed_dir}")
    print(f"Output dir: {output_dir}")
    print()

    # Initialize loader
    binance_dir = Path(args.binance_dir) if args.include_binance else None
    loader = MultiSourceLoader(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        binance_dir=binance_dir,
    )

    # Load available data sources
    print("Loading data sources...")

    # Try to load store data
    try:
        store_df = loader.load_store_data()
        print(f"  Store data: {len(store_df):,} rows")
        print(f"    Symbols: {store_df['symbol'].unique().to_list()}")
    except Exception as e:
        print(f"  Store data: Not available ({e})")

    # Load signals
    try:
        signals_df = loader.load_signals()
        print(f"  Signals: {len(signals_df):,} rows")
        print(f"    Symbols: {signals_df['symbol'].unique().to_list()}")
    except Exception as e:
        print(f"  Signals: Not available ({e})")

    # Load fills
    try:
        fills_df = loader.load_fills()
        print(f"  Fills: {len(fills_df):,} rows")
        print(f"    Symbols: {fills_df['symbol'].unique().to_list()}")
    except Exception as e:
        print(f"  Fills: Not available ({e})")

    print()

    # Get time ranges
    time_ranges = loader.get_time_range()
    print("Time ranges:")
    for source, times in time_ranges.items():
        print(f"  {source}: {times['start']} to {times['end']}")
    print()

    # Get available symbols
    available_symbols = loader.get_available_symbols()
    print(f"Available symbols: {available_symbols}")
    print()

    # Configure ML dataset
    config = MLDatasetConfig(
        interval=args.interval,
        label_type=args.label_type,
        label_horizon=args.label_horizon,
        lag_features=True,
        lag_periods=[1, 5, 10, 30],
        rolling_features=True,
        rolling_windows=[5, 10, 30, 60],
        val_fraction=0.2,
    )

    # Determine symbols to process
    if args.all_symbols:
        symbols = available_symbols
    elif args.symbol:
        symbols = [args.symbol]
    else:
        # Default to first symbol that has store data (for richer features)
        store_symbols = loader._store_data["symbol"].unique().to_list() if loader._store_data is not None else []
        if store_symbols:
            symbols = [store_symbols[0]]
        elif available_symbols:
            symbols = [available_symbols[0]]
        else:
            print("Error: No symbols available")
            return

    print(f"Processing symbols: {symbols}")
    print("=" * 60)

    # Load Binance data if requested
    if args.include_binance:
        print("\nLoading Binance public data...")
        for symbol in symbols:
            try:
                loader.load_binance_data(symbol, interval=args.interval)
            except Exception as e:
                print(f"  Warning: Could not load Binance data for {symbol}: {e}")

    # Build datasets
    results = {}
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Building dataset for {symbol}")
        print("=" * 60)

        try:
            dataset = build_ml_dataset(loader, symbol, config)
            summary = get_dataset_summary(dataset)

            results[symbol] = {
                "summary": summary,
                "dataset": dataset,
            }

            print(f"\nDataset Summary for {symbol}:")
            print(f"  Total rows: {summary['total_rows']:,}")
            print(f"  Features: {summary['n_features']}")
            print(f"  Train size: {summary['train_size']:,}")
            print(f"  Val size: {summary['val_size']:,}")
            print(f"  Label distribution: {summary['label_distribution']}")

            # Save if requested
            if not args.no_save:
                paths = save_dataset(dataset, output_dir, prefix="unified")
                print("\n  Saved to:")
                for name, path in paths.items():
                    print(f"    {name}: {path}")

        except Exception as e:
            print(f"Error building dataset for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Save overall summary
    if not args.no_save and results:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        summary_data = {
            symbol: data["summary"]
            for symbol, data in results.items()
        }
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"\nOverall summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
