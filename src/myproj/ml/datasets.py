"""Dataset utilities for simple time-based splits."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import polars as pl


def time_train_val_split(df: pl.DataFrame, val_fraction: float = 0.2) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Split by time order, keeping the last fraction for validation."""
    df_sorted = df.sort("ts")
    n = df_sorted.height
    n_val = max(1, int(n * val_fraction))
    return df_sorted.slice(0, n - n_val), df_sorted.slice(n - n_val, n_val)


def to_numpy(df: pl.DataFrame, feature_cols: Iterable[str], label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert selected columns to numpy arrays."""
    X = df.select(feature_cols).to_numpy()
    y = df.get_column(label_col).to_numpy()
    return X, y
