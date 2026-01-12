"""Schema definitions for Binance public data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


# =============================================================================
# SPOT Market Schemas
# =============================================================================

SPOT_TRADES_COLUMNS = [
    "trade_id",
    "price",
    "qty",
    "quote_qty",
    "time",
    "is_buyer_maker",
    "is_best_match",
]

SPOT_TRADES_DTYPES: dict[str, Any] = {
    "trade_id": pl.Int64,
    "price": pl.Float64,
    "qty": pl.Float64,
    "quote_qty": pl.Float64,
    "time": pl.Int64,  # Epoch ms (or microseconds from 2025-01-01)
    "is_buyer_maker": pl.Boolean,
    "is_best_match": pl.Boolean,
}

SPOT_AGGTRADES_COLUMNS = [
    "agg_trade_id",
    "price",
    "qty",
    "first_trade_id",
    "last_trade_id",
    "time",
    "is_buyer_maker",
    "is_best_match",
]

SPOT_AGGTRADES_DTYPES: dict[str, Any] = {
    "agg_trade_id": pl.Int64,
    "price": pl.Float64,
    "qty": pl.Float64,
    "first_trade_id": pl.Int64,
    "last_trade_id": pl.Int64,
    "time": pl.Int64,
    "is_buyer_maker": pl.Boolean,
    "is_best_match": pl.Boolean,
}

KLINES_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

KLINES_DTYPES: dict[str, Any] = {
    "open_time": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time": pl.Int64,
    "quote_volume": pl.Float64,
    "num_trades": pl.Int64,
    "taker_buy_base_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64,
    "ignore": pl.Utf8,
}


# =============================================================================
# Futures (USD-M) Schemas
# =============================================================================

UM_TRADES_COLUMNS = [
    "trade_id",
    "price",
    "qty",
    "quote_qty",
    "time",
    "is_buyer_maker",
]

UM_TRADES_DTYPES: dict[str, Any] = {
    "trade_id": pl.Int64,
    "price": pl.Float64,
    "qty": pl.Float64,
    "quote_qty": pl.Float64,
    "time": pl.Int64,
    "is_buyer_maker": pl.Boolean,
}

# USD-M aggTrades same as spot
UM_AGGTRADES_COLUMNS = SPOT_AGGTRADES_COLUMNS
UM_AGGTRADES_DTYPES = SPOT_AGGTRADES_DTYPES

# USD-M klines same as spot
UM_KLINES_COLUMNS = KLINES_COLUMNS
UM_KLINES_DTYPES = KLINES_DTYPES


# =============================================================================
# Futures (COIN-M) Schemas
# =============================================================================

CM_TRADES_COLUMNS = [
    "trade_id",
    "price",
    "qty",
    "base_qty",  # Different from USD-M
    "time",
    "is_buyer_maker",
]

CM_TRADES_DTYPES: dict[str, Any] = {
    "trade_id": pl.Int64,
    "price": pl.Float64,
    "qty": pl.Float64,
    "base_qty": pl.Float64,
    "time": pl.Int64,
    "is_buyer_maker": pl.Boolean,
}

# COIN-M aggTrades same as spot
CM_AGGTRADES_COLUMNS = SPOT_AGGTRADES_COLUMNS
CM_AGGTRADES_DTYPES = SPOT_AGGTRADES_DTYPES

# COIN-M klines same as spot
CM_KLINES_COLUMNS = KLINES_COLUMNS
CM_KLINES_DTYPES = KLINES_DTYPES


# =============================================================================
# Schema Registry
# =============================================================================

@dataclass
class DataSchema:
    """Schema definition for a data type."""

    columns: list[str]
    dtypes: dict[str, Any]
    has_header: bool = False  # Binance CSVs have no header


def get_schema(market_type: str, data_type: str) -> DataSchema:
    """Get the appropriate schema for market and data type."""
    if data_type == "klines":
        return DataSchema(columns=KLINES_COLUMNS, dtypes=KLINES_DTYPES)

    if market_type == "spot":
        if data_type == "trades":
            return DataSchema(columns=SPOT_TRADES_COLUMNS, dtypes=SPOT_TRADES_DTYPES)
        elif data_type == "aggTrades":
            return DataSchema(columns=SPOT_AGGTRADES_COLUMNS, dtypes=SPOT_AGGTRADES_DTYPES)

    elif market_type == "um":
        if data_type == "trades":
            return DataSchema(columns=UM_TRADES_COLUMNS, dtypes=UM_TRADES_DTYPES)
        elif data_type == "aggTrades":
            return DataSchema(columns=UM_AGGTRADES_COLUMNS, dtypes=UM_AGGTRADES_DTYPES)

    elif market_type == "cm":
        if data_type == "trades":
            return DataSchema(columns=CM_TRADES_COLUMNS, dtypes=CM_TRADES_DTYPES)
        elif data_type == "aggTrades":
            return DataSchema(columns=CM_AGGTRADES_COLUMNS, dtypes=CM_AGGTRADES_DTYPES)

    raise ValueError(f"Unknown schema for market_type={market_type}, data_type={data_type}")


# =============================================================================
# Processed Data Schemas (after feature engineering)
# =============================================================================

# Output schema after trade aggregation
TRADE_FEATURES_COLUMNS = [
    "ts",
    "symbol",
    # OHLCV
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    # Order flow features
    "buy_volume",
    "sell_volume",
    "net_volume",  # buy - sell
    "volume_imbalance",  # (buy - sell) / (buy + sell)
    "buy_trades",
    "sell_trades",
    "trade_imbalance",
    # Execution features
    "vwap",  # Volume-weighted average price
    "twap",  # Time-weighted average price
    "price_range",  # high - low
    "price_return",  # log(close/open)
]
