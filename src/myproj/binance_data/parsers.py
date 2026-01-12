"""Parsers for reading Binance public data files."""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl

from .config import MICROSECONDS_START_DATE
from .schemas import get_schema

logger = logging.getLogger(__name__)


def _read_zip_csv(
    zip_path: Path,
    columns: list[str],
    dtypes: dict,
    has_header: bool = False,
) -> pl.DataFrame:
    """Read CSV from a ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Get the CSV file name (should be only one file)
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found in {zip_path}")

        with zf.open(csv_names[0]) as f:
            content = io.BytesIO(f.read())

            df = pl.read_csv(
                content,
                has_header=has_header,
                new_columns=columns,
                schema_overrides=dtypes,
                ignore_errors=True,
            )

    return df


def _convert_timestamp(
    df: pl.DataFrame,
    time_col: str,
    file_date: date | None = None,
    market_type: str = "spot",
) -> pl.DataFrame:
    """Convert epoch timestamp to datetime.

    Note: From 2025-01-01, SPOT data timestamps are in microseconds.
    """
    # Determine if we need to handle microseconds
    use_microseconds = (
        market_type == "spot"
        and file_date is not None
        and file_date >= MICROSECONDS_START_DATE
    )

    if use_microseconds:
        # Microseconds to datetime
        df = df.with_columns(
            (pl.col(time_col) / 1000).cast(pl.Int64).alias("_ts_ms"),
        ).with_columns(
            pl.from_epoch("_ts_ms", time_unit="ms").alias("ts"),
        ).drop("_ts_ms")
    else:
        # Milliseconds to datetime
        df = df.with_columns(
            pl.from_epoch(time_col, time_unit="ms").alias("ts"),
        )

    return df


def parse_trades(
    path: Path,
    market_type: str = "spot",
    file_date: date | None = None,
    symbol: str | None = None,
) -> pl.DataFrame:
    """Parse a trades ZIP file.

    Args:
        path: Path to the ZIP file
        market_type: Market type ("spot", "um", "cm")
        file_date: Date of the file (for timestamp conversion)
        symbol: Symbol to add as column

    Returns:
        Polars DataFrame with parsed trades
    """
    schema = get_schema(market_type, "trades")
    df = _read_zip_csv(path, schema.columns, schema.dtypes, schema.has_header)

    # Convert timestamp
    df = _convert_timestamp(df, "time", file_date, market_type)

    # Add symbol column if provided
    if symbol:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

    # Add derived columns
    df = df.with_columns(
        # Trade side: True if buyer is maker means SELL aggressor
        pl.when(pl.col("is_buyer_maker"))
        .then(pl.lit("SELL"))
        .otherwise(pl.lit("BUY"))
        .alias("side"),
    )

    return df.sort("ts")


def parse_aggTrades(
    path: Path,
    market_type: str = "spot",
    file_date: date | None = None,
    symbol: str | None = None,
) -> pl.DataFrame:
    """Parse an aggTrades ZIP file.

    Args:
        path: Path to the ZIP file
        market_type: Market type ("spot", "um", "cm")
        file_date: Date of the file (for timestamp conversion)
        symbol: Symbol to add as column

    Returns:
        Polars DataFrame with parsed aggregated trades
    """
    schema = get_schema(market_type, "aggTrades")
    df = _read_zip_csv(path, schema.columns, schema.dtypes, schema.has_header)

    # Convert timestamp
    df = _convert_timestamp(df, "time", file_date, market_type)

    # Add symbol column if provided
    if symbol:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

    # Add derived columns
    df = df.with_columns(
        pl.when(pl.col("is_buyer_maker"))
        .then(pl.lit("SELL"))
        .otherwise(pl.lit("BUY"))
        .alias("side"),
        # Number of trades in this aggregate
        (pl.col("last_trade_id") - pl.col("first_trade_id") + 1).alias("num_trades"),
    )

    return df.sort("ts")


def parse_klines(
    path: Path,
    market_type: str = "spot",
    file_date: date | None = None,
    symbol: str | None = None,
) -> pl.DataFrame:
    """Parse a klines ZIP file.

    Args:
        path: Path to the ZIP file
        market_type: Market type ("spot", "um", "cm")
        file_date: Date of the file (for timestamp conversion)
        symbol: Symbol to add as column

    Returns:
        Polars DataFrame with parsed klines (OHLCV)
    """
    schema = get_schema(market_type, "klines")
    df = _read_zip_csv(path, schema.columns, schema.dtypes, schema.has_header)

    # Convert timestamps (both open and close)
    df = df.with_columns(
        pl.from_epoch("open_time", time_unit="ms").alias("ts"),
        pl.from_epoch("close_time", time_unit="ms").alias("close_ts"),
    )

    # Add symbol column if provided
    if symbol:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

    # Add derived features
    df = df.with_columns(
        # Log return
        (pl.col("close") / pl.col("open")).log().alias("log_return"),
        # Price range
        (pl.col("high") - pl.col("low")).alias("range"),
        # Range as percentage of open
        ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("range_pct"),
        # VWAP approximation (typical price)
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price"),
        # Taker buy ratio
        (pl.col("taker_buy_base_volume") / (pl.col("volume") + 1e-10)).alias("taker_buy_ratio"),
    )

    # Drop the ignore column
    df = df.drop("ignore")

    return df.sort("ts")


def load_multiple_files(
    paths: list[Path],
    data_type: Literal["trades", "aggTrades", "klines"],
    market_type: str = "spot",
    symbol: str | None = None,
) -> pl.DataFrame:
    """Load and concatenate multiple data files.

    Args:
        paths: List of ZIP file paths
        data_type: Type of data files
        market_type: Market type
        symbol: Symbol to add as column

    Returns:
        Concatenated Polars DataFrame
    """
    parser_map = {
        "trades": parse_trades,
        "aggTrades": parse_aggTrades,
        "klines": parse_klines,
    }

    parser = parser_map.get(data_type)
    if not parser:
        raise ValueError(f"Unknown data_type: {data_type}")

    dfs: list[pl.DataFrame] = []
    for path in paths:
        try:
            # Extract date from filename (e.g., BTCUSDT-trades-2024-01-15.zip)
            stem = path.stem
            date_part = stem.split("-")[-3:]  # Get last 3 parts for YYYY-MM-DD
            if len(date_part) == 3:
                file_date = date(
                    int(date_part[0]),
                    int(date_part[1]),
                    int(date_part[2]),
                )
            else:
                file_date = None

            df = parser(path, market_type=market_type, file_date=file_date, symbol=symbol)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to parse {path}: {e}")
            continue

    if not dfs:
        raise ValueError("No files could be parsed successfully")

    return pl.concat(dfs, how="diagonal").sort("ts")


def scan_downloaded_files(
    base_dir: Path,
    symbol: str,
    data_type: str,
    market_type: str = "spot",
    interval: str | None = None,
) -> list[Path]:
    """Scan for downloaded files matching the given parameters.

    Args:
        base_dir: Base directory for downloaded data
        symbol: Symbol to search for
        data_type: Data type ("trades", "aggTrades", "klines")
        market_type: Market type
        interval: Kline interval (required for klines)

    Returns:
        Sorted list of matching ZIP file paths
    """
    if market_type == "spot":
        market_path = "spot"
    elif market_type == "um":
        market_path = "futures/um"
    else:
        market_path = "futures/cm"

    if data_type == "klines":
        if not interval:
            raise ValueError("interval required for klines")
        search_dir = base_dir / market_path / data_type / symbol / interval
    else:
        search_dir = base_dir / market_path / data_type / symbol

    if not search_dir.exists():
        return []

    files = sorted(search_dir.glob("*.zip"))
    return files
