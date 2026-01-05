"""End-to-end execution fill prototype pipeline."""

from __future__ import annotations

from typing import Optional, Tuple

import polars as pl

from .datasets import time_train_val_split, to_numpy
from .features import build_feature_matrix, bucket_strategy_updates, bucket_trade_fills
from .ingest import load_processed_data
from .labels import make_execution_labels
from .models import evaluate_classifier, train_execution_model
import numpy as np


def _select_numeric_features(df: pl.DataFrame, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in exclude:
            continue
        if dtype.is_numeric():
            cols.append(name)
    return cols


def run_execution_pipeline(
    symbol: Optional[str] = None,
    bucket: str = "1s",
    horizon_steps: int = 5,
    val_fraction: float = 0.2,
) -> Tuple[dict, list[str]]:
    """Run ingestion → features → labels → train/eval for execution fill probability."""
    strat_df, fills_df = load_processed_data()
    if symbol:
        strat_df = strat_df.filter(pl.col("symbol") == symbol)
        fills_df = fills_df.filter(pl.col("symbol") == symbol)
    if strat_df.is_empty() or fills_df.is_empty():
        raise ValueError("No data found for the requested filters; ensure processed CSVs exist.")

    strat_bucket = bucket_strategy_updates(strat_df, every=bucket)
    fills_bucket = bucket_trade_fills(fills_df, every=bucket)
    feat = build_feature_matrix(strat_bucket, fills_bucket, every=bucket, forward_fill=False)
    # Replace nulls with zero for quantity-based features
    feat = feat.fill_null(0)

    # Combine strategy and trade-based fills for label generation
    feat = feat.with_columns(
        (
            pl.coalesce(pl.col("sum_filled_qty"), pl.lit(0.0))
            + pl.coalesce(pl.col("sum_fill_qty"), pl.lit(0.0))
        ).alias("total_fill_qty")
    )

    feat = make_execution_labels(feat, horizon_steps=horizon_steps, source_col="total_fill_qty")
    feat = feat.drop_nulls(subset=["future_fill_sum"])

    train_df, val_df = time_train_val_split(feat, val_fraction=val_fraction)
    exclude = {"symbol", "ts", "future_fill_sum", "will_fill"}
    feature_cols = _select_numeric_features(train_df, exclude)

    X_train, y_train = to_numpy(train_df, feature_cols, "will_fill")
    X_val, y_val = to_numpy(val_df, feature_cols, "will_fill")

    # Guard against single-class splits
    train_classes = np.unique(y_train)
    val_classes = np.unique(y_val)
    if train_classes.size < 2:
        raise ValueError(f"Training labels contain only one class: {train_classes}. Try --force, a different horizon, or more data.")
    if val_classes.size < 2:
        raise ValueError(f"Validation labels contain only one class: {val_classes}. Adjust horizon or provide more data.")

    model = train_execution_model(X_train, y_train)
    metrics = evaluate_classifier(model, X_val, y_val)
    return metrics, feature_cols
