"""Build ML-ready datasets from unified multi-source data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, NamedTuple

import numpy as np
import polars as pl

from .multi_source import (
    MultiSourceLoader,
    UnifiedDataConfig,
    build_unified_dataset,
    add_lagged_features,
    add_rolling_features,
    create_labels,
)


class UnifiedMLDataset(NamedTuple):
    """ML-ready dataset from unified multi-source data."""
    symbol: str
    df: pl.DataFrame
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    feature_cols: List[str]
    label_col: str
    timestamps_train: np.ndarray
    timestamps_val: np.ndarray


@dataclass
class MLDatasetConfig:
    """Configuration for ML dataset creation."""
    # Base config
    interval: str = "1s"
    symbols: Optional[List[str]] = None

    # Feature engineering
    lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 30])
    rolling_features: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 30, 60])

    # Label settings
    label_type: str = "fill_occurred"  # or "price_direction"
    label_horizon: int = 30

    # Train/val split
    val_fraction: float = 0.2
    time_based_split: bool = True  # Use time-based split vs random

    # Feature selection
    drop_cols: List[str] = field(default_factory=lambda: [
        "timestamp", "timestamp_ms", "symbol", "label",
        "hour", "minute", "second", "weekday",  # Keep encoded versions
    ])


def get_feature_columns(
    df: pl.DataFrame,
    config: MLDatasetConfig,
) -> List[str]:
    """Get list of feature columns from DataFrame."""
    exclude = set(config.drop_cols)
    exclude.add("label")

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            feature_cols.append(col)

    return feature_cols


def prepare_ml_features(
    df: pl.DataFrame,
    config: MLDatasetConfig,
) -> pl.DataFrame:
    """Prepare features for ML training.

    Args:
        df: Unified DataFrame
        config: ML configuration

    Returns:
        DataFrame with engineered features
    """
    # Identify numeric columns for feature engineering
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in [pl.Float64, pl.Float32] and c not in ["timestamp", "label"]
    ]

    # Price-related columns for extra features
    price_cols = [c for c in numeric_cols if any(x in c.lower() for x in ["price", "mid", "close", "open"])]

    # Volume-related columns
    volume_cols = [c for c in numeric_cols if any(x in c.lower() for x in ["volume", "qty", "fill"])]

    # Add lagged features
    if config.lag_features and config.lag_periods:
        lag_cols = price_cols[:3] + volume_cols[:2]  # Limit to key columns
        if lag_cols:
            df = add_lagged_features(df, lag_cols, config.lag_periods)

    # Add rolling features
    if config.rolling_features and config.rolling_windows:
        roll_cols = price_cols[:2] + volume_cols[:1]  # Limit to key columns
        if roll_cols:
            df = add_rolling_features(df, roll_cols, config.rolling_windows)

    # Fill NaN from lagged/rolling features
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]
    df = df.with_columns([
        pl.col(c).fill_null(0).fill_nan(0) for c in numeric_cols
    ])

    return df


def build_ml_dataset(
    loader: MultiSourceLoader,
    symbol: str,
    config: Optional[MLDatasetConfig] = None,
) -> UnifiedMLDataset:
    """Build ML-ready dataset for a single symbol.

    Args:
        loader: MultiSourceLoader with data loaded
        symbol: Symbol to build dataset for
        config: ML configuration

    Returns:
        UnifiedMLDataset ready for training
    """
    config = config or MLDatasetConfig()

    # Build unified config
    unified_config = UnifiedDataConfig(
        interval=config.interval,
        symbols=[symbol],
        include_time_features=True,
        label_horizon=config.label_horizon,
        label_type=config.label_type,
    )

    # Build unified dataset
    df = build_unified_dataset(loader, symbol, unified_config)
    print(f"Unified dataset: {len(df)} rows, {len(df.columns)} columns")

    # Create labels
    df = create_labels(df, config.label_type, config.label_horizon)

    # Prepare features
    df = prepare_ml_features(df, config)

    # Drop rows with null labels (end of dataset due to look-ahead)
    df = df.filter(pl.col("label").is_not_null())
    print(f"After label creation: {len(df)} rows")

    # Get feature columns
    feature_cols = get_feature_columns(df, config)
    print(f"Feature columns: {len(feature_cols)}")

    # Train/val split
    if config.time_based_split:
        n_train = int(len(df) * (1 - config.val_fraction))
        train_df = df.head(n_train)
        val_df = df.tail(len(df) - n_train)
    else:
        # Random split (not recommended for time series)
        n_train = int(len(df) * (1 - config.val_fraction))
        indices = np.random.permutation(len(df))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        train_df = df[train_idx]
        val_df = df[val_idx]

    # Extract arrays
    X_train = train_df.select(feature_cols).to_numpy()
    X_val = val_df.select(feature_cols).to_numpy()
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    timestamps_train = train_df["timestamp"].to_numpy()
    timestamps_val = val_df["timestamp"].to_numpy()

    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"Label distribution (train): {np.bincount(y_train.astype(int))}")

    return UnifiedMLDataset(
        symbol=symbol,
        df=df,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_cols=feature_cols,
        label_col="label",
        timestamps_train=timestamps_train,
        timestamps_val=timestamps_val,
    )


def build_multi_symbol_ml_datasets(
    loader: MultiSourceLoader,
    symbols: Optional[List[str]] = None,
    config: Optional[MLDatasetConfig] = None,
) -> Dict[str, UnifiedMLDataset]:
    """Build ML datasets for multiple symbols.

    Args:
        loader: MultiSourceLoader with data loaded
        symbols: List of symbols (None = all available)
        config: ML configuration

    Returns:
        Dictionary mapping symbol -> UnifiedMLDataset
    """
    config = config or MLDatasetConfig()
    symbols = symbols or config.symbols or loader.get_available_symbols()

    datasets = {}
    for symbol in symbols:
        try:
            print(f"\n=== Building dataset for {symbol} ===")
            datasets[symbol] = build_ml_dataset(loader, symbol, config)
        except Exception as e:
            print(f"Warning: Could not build ML dataset for {symbol}: {e}")

    return datasets


def get_dataset_summary(dataset: UnifiedMLDataset) -> dict:
    """Get summary statistics for a dataset."""
    return {
        "symbol": dataset.symbol,
        "total_rows": len(dataset.df),
        "n_features": len(dataset.feature_cols),
        "train_size": len(dataset.X_train),
        "val_size": len(dataset.X_val),
        "label_distribution": {
            int(k): int(v) for k, v in
            zip(*np.unique(dataset.y_train, return_counts=True))
        },
        "feature_cols": dataset.feature_cols,
    }


def save_dataset(
    dataset: UnifiedMLDataset,
    output_dir: Path,
    prefix: str = "unified",
) -> Dict[str, Path]:
    """Save dataset to disk.

    Args:
        dataset: UnifiedMLDataset to save
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Dictionary of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save full DataFrame
    df_path = output_dir / f"{prefix}_{dataset.symbol}_full.parquet"
    dataset.df.write_parquet(df_path)
    paths["df"] = df_path

    # Save numpy arrays
    np.savez(
        output_dir / f"{prefix}_{dataset.symbol}_arrays.npz",
        X_train=dataset.X_train,
        X_val=dataset.X_val,
        y_train=dataset.y_train,
        y_val=dataset.y_val,
        timestamps_train=dataset.timestamps_train,
        timestamps_val=dataset.timestamps_val,
        feature_cols=np.array(dataset.feature_cols),
    )
    paths["arrays"] = output_dir / f"{prefix}_{dataset.symbol}_arrays.npz"

    return paths
