"""End-to-end price move prototype pipeline."""

from __future__ import annotations

from typing import Optional, Tuple

import polars as pl

from .datasets import time_train_val_split, to_numpy
from .features import build_feature_matrix, bucket_strategy_updates, bucket_trade_fills
from .ingest import load_processed_data
from .labels import make_price_move_labels
from .models import evaluate_classifier, train_price_move_model


def _select_numeric_features(df: pl.DataFrame, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in exclude:
            continue
        if dtype.is_numeric():
            cols.append(name)
    return cols


def run_price_pipeline(
    symbol: Optional[str] = None,
    bucket: str = "1s",
    horizon_steps: int = 5,
    val_fraction: float = 0.2,
) -> Tuple[dict, list[str]]:
    """Run ingestion → features → labels → train/eval for price move prediction."""
    strat_df, fills_df = load_processed_data()
    if symbol:
        strat_df = strat_df.filter(pl.col("symbol") == symbol)
        fills_df = fills_df.filter(pl.col("symbol") == symbol)
    if strat_df.is_empty() or fills_df.is_empty():
        raise ValueError("No data found for the requested filters; ensure processed CSVs exist.")

    strat_bucket = bucket_strategy_updates(strat_df, every=bucket)
    fills_bucket = bucket_trade_fills(fills_df, every=bucket)
    feat = build_feature_matrix(strat_bucket, fills_bucket, every=bucket, forward_fill=True)

    # Require price info
    feat = feat.drop_nulls(subset=["last_fill_price"])
    feat = make_price_move_labels(feat, horizon_steps=horizon_steps)
    feat = feat.drop_nulls(subset=["future_price"])

    # Fill remaining nulls in features
    feat = feat.fill_null(0)

    train_df, val_df = time_train_val_split(feat, val_fraction=val_fraction)
    exclude = {"symbol", "ts", "future_price", "future_return", "price_move_cls"}
    feature_cols = _select_numeric_features(train_df, exclude)

    X_train, y_train = to_numpy(train_df, feature_cols, "price_move_cls")
    X_val, y_val = to_numpy(val_df, feature_cols, "price_move_cls")

    model = train_price_move_model(X_train, y_train)
    metrics = evaluate_classifier(model, X_val, y_val)
    return metrics, feature_cols
