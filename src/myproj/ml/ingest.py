"""Ingestion helpers for processed strategy updates and trade fills."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import polars as pl

from myproj.config import PROCESSED_DATA_DIR


def load_processed_data(
    strategy_files: Iterable[Path] | None = None,
    fill_files: Iterable[Path] | None = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load processed strategy_updates and trade_fills CSVs as Polars DataFrames.

    Parameters
    ----------
    strategy_files : iterable of Path | None
        If provided, uses these files; otherwise loads all CSVs in
        `data/processed/strategy_updates/`.
    fill_files : iterable of Path | None
        If provided, uses these files; otherwise loads all CSVs in
        `data/processed/trade_fills/`.
    """
    strat_dir = PROCESSED_DATA_DIR / "strategy_updates"
    fills_dir = PROCESSED_DATA_DIR / "trade_fills"

    strategy_paths = list(strategy_files) if strategy_files else sorted(strat_dir.glob("*.csv"))
    fill_paths = list(fill_files) if fill_files else sorted(fills_dir.glob("*.csv"))

    if not strategy_paths:
        raise FileNotFoundError(f"No strategy update CSVs found in {strat_dir}")
    if not fill_paths:
        raise FileNotFoundError(f"No trade fill CSVs found in {fills_dir}")

    strat_dtypes = {
        "timestamp": pl.Utf8,
        "symbol": pl.Utf8,
        "strategy_id": pl.Int64,
        "filled_qty": pl.Float64,
        "remaining_qty": pl.Float64,
        "open_qty": pl.Float64,
        "new_open_qty": pl.Float64,
    }
    fills_dtypes = {
        "timestamp": pl.Utf8,
        "symbol": pl.Utf8,
        "side": pl.Utf8,
        "fill_qty": pl.Float64,
        "fill_price": pl.Float64,
    }

    strat_df = pl.concat(
        [
            pl.read_csv(
                path,
                null_values=["", "null", "NaN"],
                infer_schema_length=0,
                schema_overrides=strat_dtypes,
            )
            for path in strategy_paths
        ],
        how="diagonal",
    )
    fills_df = pl.concat(
        [
            pl.read_csv(
                path,
                null_values=["", "null", "NaN"],
                infer_schema_length=0,
                schema_overrides=fills_dtypes,
            )
            for path in fill_paths
        ],
        how="diagonal",
    )

    strat_df = strat_df.with_columns(
        pl.col("timestamp").str.to_datetime(strict=False).alias("ts")
    ).drop_nulls(subset=["ts"])

    fills_df = fills_df.with_columns(
        pl.col("timestamp").str.to_datetime(strict=False).alias("ts")
    ).drop_nulls(subset=["ts"])

    return strat_df, fills_df
