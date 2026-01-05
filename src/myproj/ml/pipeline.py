"""Shared dataset builder for price pipeline (used by tests and scripts)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import polars as pl

from .features import bucket_strategy_updates, bucket_trade_fills, build_feature_matrix
from .ingest import load_processed_data


def _select_numeric_features(df: pl.DataFrame, exclude: set[str]) -> List[str]:
    cols: List[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in exclude:
            continue
        if dtype.is_numeric():
            cols.append(name)
    return cols


def build_dataset(
    symbol: Optional[str] = None,
    bucket: str = "1s",
    horizon: int = 5,
    deadband: float = 5e-4,
    limit_rows: Optional[int] = None,
) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Build a feature/label dataset for price-move modeling.

    Returns
    -------
    df : pl.DataFrame
        DataFrame containing at least columns ['ts', 'mid', 'future_mid', 'ret_h', 'y'].
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels array.
    feature_names : list[str]
        Names corresponding to columns of X.
    """
    strat_df, fills_df = load_processed_data()
    if symbol:
        strat_df = strat_df.filter(pl.col("symbol") == symbol)
        fills_df = fills_df.filter(pl.col("symbol") == symbol)
    if strat_df.is_empty() or fills_df.is_empty():
        raise ValueError("No data found for the requested filters; ensure processed CSVs exist.")

    strat_bucket = bucket_strategy_updates(strat_df, every=bucket)
    fills_bucket = bucket_trade_fills(fills_df, every=bucket)
    feat = build_feature_matrix(strat_bucket, fills_bucket, every=bucket, forward_fill=True)

    # Require a price proxy
    feat = feat.drop_nulls(subset=["last_fill_price"])
    feat = feat.with_columns(
        pl.col("last_fill_price").alias("mid"),
        pl.col("last_fill_price").shift(-horizon).over("symbol").alias("future_mid"),
    )
    feat = feat.with_columns(
        ((pl.col("future_mid") - pl.col("mid")) / pl.col("mid")).alias("ret_h")
    )
    label_expr = (
        pl.when(pl.col("ret_h") > deadband)
        .then(pl.lit(1))
        .when(pl.col("ret_h") < -deadband)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
    )
    feat = feat.with_columns(label_expr.alias("y"))
    feat = feat.drop_nulls(subset=["future_mid"])

    feat = feat.sort(["symbol", "ts"]).fill_null(0)
    if limit_rows is not None:
        feat = feat.head(limit_rows)

    exclude = {"symbol", "ts", "future_mid", "ret_h", "y"}
    feature_cols = _select_numeric_features(feat, exclude)

    X = feat.select(feature_cols).to_numpy()
    y = feat.get_column("y").to_numpy()
    return feat, X, y, feature_cols
