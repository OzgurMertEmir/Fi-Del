"""Pipeline integration to build ML-ready datasets from Binance data.

This module provides the high-level interface for:
1. Downloading raw data from Binance
2. Parsing and validating the data
3. Computing features (OHLCV + order flow)
4. Adding labels for supervised learning
5. Splitting into train/val/test sets
6. Saving to parquet format
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl

from .config import BinanceDataConfig
from .downloader import BinanceDownloader
from .parsers import scan_downloaded_files, load_multiple_files
from .aggregators import (
    compute_trade_flow_features,
    add_rolling_flow_features,
    add_return_labels,
    add_direction_labels,
    merge_multi_symbol,
)

logger = logging.getLogger(__name__)


@dataclass
class BinancePipelineConfig:
    """Configuration for the full Binance data pipeline."""

    # Data source
    symbols: list[str]
    start_date: date
    end_date: date
    market_type: str = "spot"
    data_type: str = "trades"  # trades or aggTrades

    # Aggregation
    interval: str = "1s"
    rolling_windows: list[int] = field(default_factory=lambda: [5, 15, 60, 300])

    # Labels
    horizons: list[int] = field(default_factory=lambda: [5, 30, 60])
    deadzone_bps: float = 5.0

    # Splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_bars: int = 60  # Gap between splits to prevent leakage

    # Paths
    raw_data_dir: Path = field(default_factory=lambda: Path("data/binance"))
    output_dir: Path = field(default_factory=lambda: Path("data/processed"))

    # Options
    download_if_missing: bool = True
    enable_cross_asset: bool = False  # Merge features from multiple symbols


def run_binance_pipeline(config: BinancePipelineConfig) -> dict[str, Path]:
    """Run the full Binance data pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        Dictionary with paths to output files
    """
    logger.info(f"Starting Binance pipeline for {config.symbols}")
    logger.info(f"Date range: {config.start_date} to {config.end_date}")

    # Step 1: Download data if needed
    if config.download_if_missing:
        _ensure_data_downloaded(config)

    # Step 2: Load and parse data
    symbol_dfs = {}
    for symbol in config.symbols:
        df = _load_symbol_data(config, symbol)
        if df is not None and len(df) > 0:
            symbol_dfs[symbol] = df
            logger.info(f"Loaded {len(df):,} rows for {symbol}")

    if not symbol_dfs:
        raise ValueError("No data could be loaded for any symbol")

    # Step 3: Aggregate and compute features
    processed_dfs = {}
    for symbol, df in symbol_dfs.items():
        logger.info(f"Processing {symbol}...")

        # Aggregate trades to bars with order flow features
        agg = compute_trade_flow_features(df, interval=config.interval)

        # Add rolling features
        agg = add_rolling_flow_features(agg, windows=config.rolling_windows)

        # Add labels
        agg = add_return_labels(agg, horizons=config.horizons)
        agg = add_direction_labels(agg, horizons=config.horizons, deadzone_bps=config.deadzone_bps)

        processed_dfs[symbol] = agg
        logger.info(f"Processed {symbol}: {len(agg):,} bars")

    # Step 4: Optionally merge cross-asset features
    if config.enable_cross_asset and len(processed_dfs) > 1:
        # Use first symbol as target
        target_symbol = config.symbols[0]
        merged = merge_multi_symbol(processed_dfs, target_symbol)
        datasets = {target_symbol: merged}
    else:
        datasets = processed_dfs

    # Step 5: Split and save each symbol
    output_paths = {}
    for symbol, df in datasets.items():
        symbol_output_dir = config.output_dir / f"binance_{symbol}_{config.interval}"
        paths = _split_and_save(df, symbol_output_dir, config)
        output_paths[symbol] = paths

    logger.info("Pipeline complete!")
    return output_paths


def _ensure_data_downloaded(config: BinancePipelineConfig) -> None:
    """Download data if not already present."""
    for symbol in config.symbols:
        files = scan_downloaded_files(
            config.raw_data_dir,
            symbol,
            config.data_type,
            config.market_type,
        )

        if not files:
            logger.info(f"Downloading {config.data_type} for {symbol}...")
            download_config = BinanceDataConfig(
                symbols=[symbol],
                data_type=config.data_type,
                market_type=config.market_type,
                start_date=config.start_date,
                end_date=config.end_date,
                period="daily",
                output_dir=config.raw_data_dir,
            )
            downloader = BinanceDownloader(download_config)
            downloader.download_all()


def _load_symbol_data(config: BinancePipelineConfig, symbol: str) -> pl.DataFrame | None:
    """Load all data files for a symbol."""
    files = scan_downloaded_files(
        config.raw_data_dir,
        symbol,
        config.data_type,
        config.market_type,
    )

    if not files:
        logger.warning(f"No data files found for {symbol}")
        return None

    # Filter by date range
    filtered_files = []
    for f in files:
        # Extract date from filename
        try:
            stem = f.stem
            parts = stem.split("-")
            file_date = date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
            if config.start_date <= file_date <= config.end_date:
                filtered_files.append(f)
        except (ValueError, IndexError):
            continue

    if not filtered_files:
        logger.warning(f"No files in date range for {symbol}")
        return None

    return load_multiple_files(
        filtered_files,
        config.data_type,
        config.market_type,
        symbol=symbol,
    )


def _split_and_save(
    df: pl.DataFrame,
    output_dir: Path,
    config: BinancePipelineConfig,
) -> dict[str, Path]:
    """Split data and save to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove rows with NaN labels (at the end due to future lookups)
    label_cols = [c for c in df.columns if c.startswith("y_")]
    df = df.drop_nulls(subset=label_cols)

    n = len(df)
    train_end = int(n * config.train_ratio)
    val_end = train_end + config.embargo_bars + int(n * config.val_ratio)

    # Apply embargo gaps
    train_df = df.slice(0, train_end)
    val_df = df.slice(train_end + config.embargo_bars, int(n * config.val_ratio))
    test_df = df.slice(val_end + config.embargo_bars, n - val_end - config.embargo_bars)

    # Save
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)
    test_df.write_parquet(test_path)

    # Save metadata
    feature_cols = [
        c for c in df.columns
        if not c.startswith("y_") and c not in ["ts", "symbol"]
    ]
    label_cols = [c for c in df.columns if c.startswith("y_")]

    metadata = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "feature_columns": feature_cols,
        "label_columns": label_cols,
        "interval": config.interval,
        "start_date": str(config.start_date),
        "end_date": str(config.end_date),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved to {output_dir}:")
    logger.info(f"  Train: {len(train_df):,} rows")
    logger.info(f"  Val:   {len(val_df):,} rows")
    logger.info(f"  Test:  {len(test_df):,} rows")

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "metadata": output_dir / "metadata.json",
    }


# Convenience function for quick experimentation
def quick_load(
    symbol: str = "BTCUSDT",
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
    interval: str = "1s",
    data_dir: Path | str = "data/binance",
) -> pl.DataFrame:
    """Quickly load and process Binance data for experimentation.

    Args:
        symbol: Trading pair symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Aggregation interval
        data_dir: Directory with downloaded data

    Returns:
        Processed DataFrame with features and labels
    """
    from datetime import datetime

    config = BinancePipelineConfig(
        symbols=[symbol],
        start_date=datetime.strptime(start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(end_date, "%Y-%m-%d").date(),
        interval=interval,
        raw_data_dir=Path(data_dir),
        download_if_missing=True,
    )

    # Load data
    df = _load_symbol_data(config, symbol)
    if df is None:
        raise ValueError(f"No data found for {symbol}")

    # Process
    agg = compute_trade_flow_features(df, interval=interval)
    agg = add_rolling_flow_features(agg, windows=[5, 15, 60, 300])
    agg = add_return_labels(agg, horizons=[5, 30, 60])
    agg = add_direction_labels(agg, horizons=[5, 30, 60], deadzone_bps=5.0)

    return agg
