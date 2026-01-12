"""Integration utilities for using Binance data with ML pipelines.

This module bridges the Binance data pipeline with existing ML utilities,
providing convenient functions for model training and evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import polars as pl

from ..ml.datasets import time_train_val_split, to_numpy
from ..ml.models import train_price_move_model, evaluate_classifier
from .aggregators import (
    compute_trade_flow_features,
    add_rolling_flow_features,
    add_advanced_flow_features,
    add_return_labels,
    add_direction_labels,
)
from .pipeline import BinancePipelineConfig, _load_symbol_data

logger = logging.getLogger(__name__)


def _select_numeric_features(df: pl.DataFrame, exclude: set[str]) -> list[str]:
    """Select numeric feature columns, excluding specified columns."""
    cols: list[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in exclude:
            continue
        if dtype.is_numeric():
            cols.append(name)
    return cols


@dataclass
class BinanceMLDataset:
    """Container for ML-ready dataset from Binance data.

    Attributes:
        df: Full processed DataFrame
        X_train: Training features (numpy)
        y_train: Training labels (numpy)
        X_val: Validation features (numpy)
        y_val: Validation labels (numpy)
        X_test: Test features (numpy)
        y_test: Test labels (numpy)
        feature_cols: List of feature column names
        label_col: Name of the label column
    """
    df: pl.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray | None
    y_test: np.ndarray | None
    feature_cols: list[str]
    label_col: str

    def summary(self) -> dict:
        """Return summary statistics about the dataset."""
        result = {
            "total_rows": len(self.df),
            "n_features": len(self.feature_cols),
            "train_size": len(self.X_train),
            "val_size": len(self.X_val),
            "label_col": self.label_col,
        }
        if self.X_test is not None:
            result["test_size"] = len(self.X_test)

        # Label distribution
        train_unique, train_counts = np.unique(self.y_train, return_counts=True)
        result["train_label_dist"] = dict(zip(train_unique.tolist(), train_counts.tolist()))

        return result


def build_binance_ml_dataset(
    symbol: str = "BTCUSDT",
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
    interval: str = "1s",
    data_dir: str | Path = "data/binance",
    label_horizon: int = 30,
    deadzone_bps: float = 5.0,
    rolling_windows: list[int] | None = None,
    include_advanced_features: bool = True,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    embargo_bars: int = 60,
) -> BinanceMLDataset:
    """Build an ML-ready dataset from Binance trade data.

    This is a convenient function that combines data loading, feature engineering,
    and train/val/test splitting into a single call.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Aggregation interval (e.g., "1s", "5s", "1m")
        data_dir: Directory with downloaded Binance data
        label_horizon: Number of bars ahead for prediction target
        deadzone_bps: Deadzone threshold in basis points for direction labels
        rolling_windows: Window sizes for rolling features (default: [5, 15, 60, 300])
        include_advanced_features: Whether to add advanced order flow features
        val_fraction: Fraction of data for validation
        test_fraction: Fraction of data for testing (0 for no test set)
        embargo_bars: Gap between train/val/test to prevent leakage

    Returns:
        BinanceMLDataset with features, labels, and train/val/test splits
    """
    from datetime import datetime

    if rolling_windows is None:
        rolling_windows = [5, 15, 60, 300]

    # Create config for loading
    config = BinancePipelineConfig(
        symbols=[symbol],
        start_date=datetime.strptime(start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(end_date, "%Y-%m-%d").date(),
        interval=interval,
        raw_data_dir=Path(data_dir),
        download_if_missing=True,
    )

    # Load raw data
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    df = _load_symbol_data(config, symbol)
    if df is None or len(df) == 0:
        raise ValueError(f"No data found for {symbol}")

    logger.info(f"Loaded {len(df):,} raw trades")

    # Process features
    logger.info(f"Computing features with {interval} bars")
    agg = compute_trade_flow_features(df, interval=interval)
    agg = add_rolling_flow_features(agg, windows=rolling_windows)

    if include_advanced_features:
        logger.info("Adding advanced order flow features")
        agg = add_advanced_flow_features(agg, windows=rolling_windows)

    # Add labels
    logger.info(f"Adding labels for horizon={label_horizon}")
    agg = add_return_labels(agg, horizons=[label_horizon])
    agg = add_direction_labels(agg, horizons=[label_horizon], deadzone_bps=deadzone_bps)

    # Define label column
    label_col = f"y_dir_{label_horizon}"

    # Drop rows with null labels (from future lookups at the end)
    agg = agg.drop_nulls(subset=[label_col])

    # Fill any remaining null features with 0
    agg = agg.fill_null(0)

    # Select feature columns
    exclude = {"ts", "symbol"} | {c for c in agg.columns if c.startswith("y_")}
    feature_cols = _select_numeric_features(agg, exclude)

    logger.info(f"Dataset: {len(agg):,} rows, {len(feature_cols)} features")

    # Time-based split with embargo
    n = len(agg)
    agg = agg.sort("ts")

    if test_fraction > 0:
        # Three-way split
        train_end = int(n * (1 - val_fraction - test_fraction))
        val_end = train_end + embargo_bars + int(n * val_fraction)

        train_df = agg.slice(0, train_end)
        val_df = agg.slice(train_end + embargo_bars, int(n * val_fraction))
        test_df = agg.slice(val_end + embargo_bars, n - val_end - embargo_bars)

        X_test, y_test = to_numpy(test_df, feature_cols, label_col)
    else:
        # Two-way split
        train_df, val_df = time_train_val_split(agg, val_fraction=val_fraction)
        X_test, y_test = None, None

    X_train, y_train = to_numpy(train_df, feature_cols, label_col)
    X_val, y_val = to_numpy(val_df, feature_cols, label_col)

    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}" +
                (f", Test: {len(X_test):,}" if X_test is not None else ""))

    return BinanceMLDataset(
        df=agg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        label_col=label_col,
    )


def run_binance_baseline(
    symbol: str = "BTCUSDT",
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
    interval: str = "1s",
    data_dir: str | Path = "data/binance",
    label_horizon: int = 30,
    deadzone_bps: float = 5.0,
    include_advanced_features: bool = True,
) -> Tuple[dict, BinanceMLDataset]:
    """Run a baseline logistic regression model on Binance data.

    This provides a quick way to establish baseline performance on the
    price direction prediction task.

    Args:
        symbol: Trading pair symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Aggregation interval
        data_dir: Directory with downloaded data
        label_horizon: Prediction horizon in bars
        deadzone_bps: Deadzone for direction classification
        include_advanced_features: Whether to use advanced features

    Returns:
        Tuple of (metrics dict, BinanceMLDataset)
    """
    dataset = build_binance_ml_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        data_dir=data_dir,
        label_horizon=label_horizon,
        deadzone_bps=deadzone_bps,
        include_advanced_features=include_advanced_features,
    )

    logger.info("Training baseline logistic regression model")
    model = train_price_move_model(dataset.X_train, dataset.y_train)

    metrics = evaluate_classifier(model, dataset.X_val, dataset.y_val)

    # Add dataset info to metrics
    metrics["n_train"] = len(dataset.X_train)
    metrics["n_val"] = len(dataset.X_val)
    metrics["n_features"] = len(dataset.feature_cols)

    logger.info(f"Validation metrics: {metrics}")

    return metrics, dataset


def load_processed_binance_dataset(
    dataset_dir: str | Path,
    split: Literal["train", "val", "test"] = "train",
) -> pl.DataFrame:
    """Load a pre-processed Binance dataset from parquet files.

    Args:
        dataset_dir: Directory containing train.parquet, val.parquet, test.parquet
        split: Which split to load

    Returns:
        DataFrame with features and labels
    """
    path = Path(dataset_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return pl.read_parquet(path)


def get_feature_importance(
    model,
    feature_cols: list[str],
    top_k: int = 20,
) -> list[Tuple[str, float]]:
    """Extract feature importance from a trained sklearn pipeline.

    Args:
        model: Trained sklearn pipeline with LogisticRegression
        feature_cols: List of feature names
        top_k: Number of top features to return

    Returns:
        List of (feature_name, importance) tuples sorted by importance
    """
    # Get the logistic regression coefficients
    lr = model.named_steps.get("logisticregression", model[-1])

    # For multiclass, use mean absolute coefficient across classes
    if lr.coef_.ndim == 2 and lr.coef_.shape[0] > 1:
        importance = np.abs(lr.coef_).mean(axis=0)
    else:
        importance = np.abs(lr.coef_.ravel())

    # Pair with feature names and sort
    feature_importance = list(zip(feature_cols, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance[:top_k]
