"""Timestamp alignment and resampling for multi-source data."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Tuple

import polars as pl


def normalize_timestamp_precision(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    target_precision: str = "us",
) -> pl.DataFrame:
    """Normalize timestamp column to consistent precision.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        target_precision: Target precision ("us", "ms", "ns")

    Returns:
        DataFrame with normalized timestamp precision
    """
    if timestamp_col not in df.columns:
        return df

    col = df[timestamp_col]

    # If string, parse it first
    if col.dtype == pl.Utf8:
        df = df.with_columns(
            pl.col(timestamp_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f")
        )
        col = df[timestamp_col]

    # If already a datetime, cast to target precision
    if col.dtype in [pl.Datetime("ns"), pl.Datetime("us"), pl.Datetime("ms")]:
        df = df.with_columns(
            pl.col(timestamp_col).cast(pl.Datetime(target_precision))
        )

    return df


def create_time_grid(
    start: datetime,
    end: datetime,
    interval: str = "1s",
    symbols: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Create a regular time grid for data alignment.

    Args:
        start: Start datetime
        end: End datetime
        interval: Time interval (e.g., "1s", "100ms", "1m")
        symbols: Optional list of symbols to create grid for

    Returns:
        DataFrame with timestamp column (and symbol column if symbols provided)
    """
    # Create timestamp range
    timestamps = pl.datetime_range(
        start,
        end,
        interval,
        eager=True,
    ).alias("timestamp")

    grid = pl.DataFrame({"timestamp": timestamps})

    if symbols:
        # Cross join with symbols
        symbol_df = pl.DataFrame({"symbol": symbols})
        grid = grid.join(symbol_df, how="cross")

    return grid.sort(["timestamp"] if not symbols else ["symbol", "timestamp"])


def align_to_grid(
    df: pl.DataFrame,
    grid: pl.DataFrame,
    timestamp_col: str = "timestamp",
    symbol_col: Optional[str] = "symbol",
    fill_strategy: str = "forward",
) -> pl.DataFrame:
    """Align data to a regular time grid using asof join.

    Args:
        df: Source DataFrame with timestamps
        grid: Regular time grid from create_time_grid
        timestamp_col: Name of timestamp column
        symbol_col: Name of symbol column (None for single-symbol data)
        fill_strategy: How to fill gaps ("forward", "backward", "null")

    Returns:
        DataFrame aligned to the grid
    """
    df = df.sort(timestamp_col if not symbol_col else [symbol_col, timestamp_col])
    grid = grid.sort("timestamp" if not symbol_col else [symbol_col, "timestamp"])

    if symbol_col and symbol_col in df.columns:
        # Per-symbol asof join
        result = grid.join_asof(
            df.rename({timestamp_col: "timestamp"}) if timestamp_col != "timestamp" else df,
            on="timestamp",
            by=symbol_col,
            strategy="backward" if fill_strategy == "forward" else fill_strategy,
        )
    else:
        # Single asof join
        result = grid.join_asof(
            df.rename({timestamp_col: "timestamp"}) if timestamp_col != "timestamp" else df,
            on="timestamp",
            strategy="backward" if fill_strategy == "forward" else fill_strategy,
        )

    return result


def find_overlapping_range(
    *dataframes: pl.DataFrame,
    timestamp_col: str = "timestamp",
) -> Tuple[datetime, datetime]:
    """Find the overlapping time range across multiple DataFrames.

    Args:
        dataframes: DataFrames to find overlap for
        timestamp_col: Name of timestamp column

    Returns:
        Tuple of (start, end) datetimes for overlap
    """
    starts = []
    ends = []

    for df in dataframes:
        if df is not None and len(df) > 0:
            starts.append(df[timestamp_col].min())
            ends.append(df[timestamp_col].max())

    if not starts:
        raise ValueError("No valid DataFrames provided")

    return max(starts), min(ends)


def resample_signals(
    signals_df: pl.DataFrame,
    interval: str = "1s",
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Resample signal data to fixed intervals.

    Aggregates signals by taking the last signal value in each interval
    and computing signal statistics.

    Args:
        signals_df: DataFrame with signal data
        interval: Resampling interval
        timestamp_col: Name of timestamp column
        symbol_col: Name of symbol column

    Returns:
        Resampled signals DataFrame
    """
    # Parse timestamp if string
    if signals_df[timestamp_col].dtype == pl.Utf8:
        signals_df = signals_df.with_columns(
            pl.col(timestamp_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f").alias(timestamp_col)
        )

    return signals_df.group_by_dynamic(
        timestamp_col,
        every=interval,
        group_by=symbol_col,
    ).agg([
        # Signal aggregations
        pl.col("signal_value").last().alias("last_signal"),
        pl.col("signal_value").mean().alias("avg_signal"),
        pl.col("signal_value").sum().alias("sum_signal"),
        pl.col("signal_value").count().alias("n_signals"),

        # Signal extremes
        pl.col("signal_value").max().alias("max_signal"),
        pl.col("signal_value").min().alias("min_signal"),

        # Strategy diversity
        pl.col("strategy_id").n_unique().alias("n_strategies"),

        # Entry stats
        pl.col("entry_allowed").sum().alias("n_entries_allowed"),

        # Mid price from signals
        pl.col("mid_price").last().alias("signal_mid_price"),
    ]).sort([symbol_col, timestamp_col])


def resample_fills(
    fills_df: pl.DataFrame,
    interval: str = "1s",
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Resample fill data to fixed intervals.

    Args:
        fills_df: DataFrame with fill data
        interval: Resampling interval
        timestamp_col: Name of timestamp column
        symbol_col: Name of symbol column

    Returns:
        Resampled fills DataFrame
    """
    # Parse timestamp if string
    if fills_df[timestamp_col].dtype == pl.Utf8:
        fills_df = fills_df.with_columns(
            pl.col(timestamp_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f").alias(timestamp_col)
        )

    # Ensure qty and price are numeric
    fills_df = fills_df.with_columns([
        pl.col("qty").cast(pl.Float64),
        pl.col("price").cast(pl.Float64),
    ])

    return fills_df.group_by_dynamic(
        timestamp_col,
        every=interval,
        group_by=symbol_col,
    ).agg([
        # Volume stats
        pl.col("qty").sum().alias("fill_volume"),
        pl.col("qty").count().alias("n_fills"),

        # Price stats
        pl.col("price").mean().alias("avg_fill_price"),
        pl.col("price").last().alias("last_fill_price"),

        # VWAP
        (pl.col("qty") * pl.col("price")).sum().alias("_dollar_volume"),

        # Side breakdown
        (pl.col("side") == "BUY").sum().alias("n_buys"),
        (pl.col("side") == "SELL").sum().alias("n_sells"),

        # Buy/Sell volume
        pl.when(pl.col("side") == "BUY")
        .then(pl.col("qty"))
        .otherwise(0)
        .sum()
        .alias("buy_volume"),

        pl.when(pl.col("side") == "SELL")
        .then(pl.col("qty"))
        .otherwise(0)
        .sum()
        .alias("sell_volume"),

        # Unique orders
        pl.col("order_id").n_unique().alias("n_orders"),
    ]).with_columns([
        # VWAP calculation
        (pl.col("_dollar_volume") / pl.col("fill_volume")).alias("vwap"),
        # Volume imbalance
        ((pl.col("buy_volume") - pl.col("sell_volume")) /
         (pl.col("buy_volume") + pl.col("sell_volume") + 1e-10)).alias("fill_imbalance"),
    ]).drop("_dollar_volume").sort([symbol_col, timestamp_col])


def compute_time_features(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Add time-based features to a DataFrame.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with added time features
    """
    return df.with_columns([
        pl.col(timestamp_col).dt.hour().alias("hour"),
        pl.col(timestamp_col).dt.minute().alias("minute"),
        pl.col(timestamp_col).dt.second().alias("second"),
        pl.col(timestamp_col).dt.weekday().alias("weekday"),

        # Cyclic encoding for hour
        (pl.col(timestamp_col).dt.hour() * 2 * 3.14159 / 24).sin().alias("hour_sin"),
        (pl.col(timestamp_col).dt.hour() * 2 * 3.14159 / 24).cos().alias("hour_cos"),
    ])
