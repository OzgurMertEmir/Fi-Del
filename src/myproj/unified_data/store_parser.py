"""Parser for .store order book snapshot files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl


def parse_store_file(
    file_path: Path,
    symbol: Optional[str] = None,
) -> pl.DataFrame:
    """Parse a .store order book file into a Polars DataFrame.

    The .store format is CSV-like with columns:
    - ignored (always 1)
    - timestamp (epoch milliseconds)
    - bid_price
    - bid_qty
    - ask_price
    - ask_qty
    - volume

    Args:
        file_path: Path to the .store file
        symbol: Symbol name (extracted from filename if not provided)

    Returns:
        Polars DataFrame with order book snapshots
    """
    file_path = Path(file_path)

    if symbol is None:
        symbol = file_path.stem

    df = pl.read_csv(
        file_path,
        comment_prefix="#",
        has_header=False,
        new_columns=["_ignore", "timestamp_ms", "bid_price", "bid_qty", "ask_price", "ask_qty", "volume"],
    )

    df = df.drop("_ignore").with_columns([
        pl.col("timestamp_ms").cast(pl.Int64),
        pl.col("bid_price").cast(pl.Float64),
        pl.col("bid_qty").cast(pl.Float64),
        pl.col("ask_price").cast(pl.Float64),
        pl.col("ask_qty").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.lit(symbol).alias("symbol"),
    ])

    # Convert timestamp to datetime
    df = df.with_columns([
        (pl.col("timestamp_ms") * 1_000_000).cast(pl.Datetime("ns")).alias("timestamp"),
    ])

    # Compute derived features (in two steps since spread_bps depends on mid_price)
    df = df.with_columns([
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price"),
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
        ((pl.col("bid_qty") - pl.col("ask_qty")) / (pl.col("bid_qty") + pl.col("ask_qty"))).alias("book_imbalance"),
    ])

    # Now compute spread_bps using mid_price
    df = df.with_columns([
        (pl.col("spread") / pl.col("mid_price") * 10000).alias("spread_bps"),
    ])

    return df.select([
        "timestamp",
        "timestamp_ms",
        "symbol",
        "bid_price",
        "bid_qty",
        "ask_price",
        "ask_qty",
        "mid_price",
        "spread",
        "spread_bps",
        "book_imbalance",
        "volume",
    ]).sort("timestamp")


def load_all_store_files(
    data_dir: Path,
    symbols: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Load all .store files from a directory.

    Args:
        data_dir: Directory containing .store files
        symbols: Optional filter for specific symbols

    Returns:
        Combined DataFrame with all order book data
    """
    data_dir = Path(data_dir)
    store_files = list(data_dir.glob("*.store"))

    if not store_files:
        raise FileNotFoundError(f"No .store files found in {data_dir}")

    dfs = []
    for store_file in store_files:
        symbol = store_file.stem
        if symbols is not None and symbol not in symbols:
            continue

        df = parse_store_file(store_file, symbol=symbol)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No matching .store files for symbols: {symbols}")

    return pl.concat(dfs).sort(["symbol", "timestamp"])


def resample_store_data(
    df: pl.DataFrame,
    interval: str = "1s",
) -> pl.DataFrame:
    """Resample order book data to fixed intervals.

    Args:
        df: Order book DataFrame from parse_store_file
        interval: Resampling interval (e.g., "1s", "100ms", "1m")

    Returns:
        Resampled DataFrame with OHLC-style aggregation
    """
    return df.group_by_dynamic(
        "timestamp",
        every=interval,
        group_by="symbol",
    ).agg([
        pl.col("mid_price").first().alias("open"),
        pl.col("mid_price").max().alias("high"),
        pl.col("mid_price").min().alias("low"),
        pl.col("mid_price").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("bid_price").last().alias("bid_price"),
        pl.col("bid_qty").last().alias("bid_qty"),
        pl.col("ask_price").last().alias("ask_price"),
        pl.col("ask_qty").last().alias("ask_qty"),
        pl.col("spread").mean().alias("avg_spread"),
        pl.col("spread_bps").mean().alias("avg_spread_bps"),
        pl.col("book_imbalance").mean().alias("avg_book_imbalance"),
        pl.len().alias("tick_count"),
    ]).sort(["symbol", "timestamp"])


def get_store_summary(df: pl.DataFrame) -> dict:
    """Get summary statistics for order book data."""
    return {
        "n_rows": len(df),
        "symbols": df["symbol"].unique().to_list(),
        "time_range": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "price_stats": {
            "min_mid": float(df["mid_price"].min()),
            "max_mid": float(df["mid_price"].max()),
            "avg_spread_bps": float(df["spread_bps"].mean()),
        },
    }
