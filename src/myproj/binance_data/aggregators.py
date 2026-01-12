"""Aggregation functions to derive features from raw trade data."""

from __future__ import annotations

import polars as pl


def aggregate_trades_to_ohlcv(
    df: pl.DataFrame,
    interval: str = "1s",
    price_col: str = "price",
    qty_col: str = "qty",
) -> pl.DataFrame:
    """Aggregate raw trades into OHLCV bars.

    Args:
        df: DataFrame with trades (must have 'ts', price_col, qty_col, and optionally 'symbol')
        interval: Time interval for aggregation (e.g., "1s", "1m", "5m", "1h")
        price_col: Name of the price column
        qty_col: Name of the quantity column

    Returns:
        DataFrame with OHLCV bars
    """
    has_symbol = "symbol" in df.columns
    group_cols = ["symbol", "ts_bucket"] if has_symbol else ["ts_bucket"]

    # Bucket by interval
    df = df.with_columns(
        pl.col("ts").dt.truncate(interval).alias("ts_bucket")
    )

    # Aggregate
    agg = df.group_by(group_cols).agg(
        # OHLCV
        pl.col(price_col).first().alias("open"),
        pl.col(price_col).max().alias("high"),
        pl.col(price_col).min().alias("low"),
        pl.col(price_col).last().alias("close"),
        pl.col(qty_col).sum().alias("volume"),
        (pl.col(price_col) * pl.col(qty_col)).sum().alias("quote_volume"),
        pl.len().alias("num_trades"),
        # VWAP
        ((pl.col(price_col) * pl.col(qty_col)).sum() / pl.col(qty_col).sum()).alias("vwap"),
    )

    # Rename bucket back to ts
    agg = agg.rename({"ts_bucket": "ts"}).sort("ts")

    return agg


def compute_trade_flow_features(
    df: pl.DataFrame,
    interval: str = "1s",
    price_col: str = "price",
    qty_col: str = "qty",
) -> pl.DataFrame:
    """Compute order flow features from trade data.

    This derives order book-like imbalance features from trade executions.
    The `is_buyer_maker` flag tells us who was the aggressor:
    - is_buyer_maker=True: Seller was aggressor (hit the bid)
    - is_buyer_maker=False: Buyer was aggressor (lifted the ask)

    Args:
        df: DataFrame with trades (must have 'ts', 'is_buyer_maker', price_col, qty_col)
        interval: Time interval for aggregation
        price_col: Name of the price column
        qty_col: Name of the quantity column

    Returns:
        DataFrame with order flow features per interval
    """
    has_symbol = "symbol" in df.columns
    group_cols = ["symbol", "ts_bucket"] if has_symbol else ["ts_bucket"]

    # Bucket by interval
    df = df.with_columns(
        pl.col("ts").dt.truncate(interval).alias("ts_bucket")
    )

    # Aggregate with buy/sell split
    agg = df.group_by(group_cols).agg(
        # OHLCV
        pl.col(price_col).first().alias("open"),
        pl.col(price_col).max().alias("high"),
        pl.col(price_col).min().alias("low"),
        pl.col(price_col).last().alias("close"),
        pl.col(qty_col).sum().alias("volume"),
        (pl.col(price_col) * pl.col(qty_col)).sum().alias("quote_volume"),
        pl.len().alias("num_trades"),

        # Buy side (aggressor buying = lifting ask, is_buyer_maker=False)
        pl.when(~pl.col("is_buyer_maker"))
        .then(pl.col(qty_col))
        .otherwise(0.0)
        .sum()
        .alias("buy_volume"),

        pl.when(~pl.col("is_buyer_maker"))
        .then(1)
        .otherwise(0)
        .sum()
        .alias("buy_trades"),

        # Sell side (aggressor selling = hitting bid, is_buyer_maker=True)
        pl.when(pl.col("is_buyer_maker"))
        .then(pl.col(qty_col))
        .otherwise(0.0)
        .sum()
        .alias("sell_volume"),

        pl.when(pl.col("is_buyer_maker"))
        .then(1)
        .otherwise(0)
        .sum()
        .alias("sell_trades"),

        # VWAP
        ((pl.col(price_col) * pl.col(qty_col)).sum() / pl.col(qty_col).sum()).alias("vwap"),

        # First and last prices for return calculation
        pl.col(price_col).first().alias("first_price"),
        pl.col(price_col).last().alias("last_price"),
    )

    # Derived features
    agg = agg.with_columns(
        # Net volume (positive = net buying)
        (pl.col("buy_volume") - pl.col("sell_volume")).alias("net_volume"),

        # Volume imbalance [-1, 1]
        (
            (pl.col("buy_volume") - pl.col("sell_volume"))
            / (pl.col("buy_volume") + pl.col("sell_volume") + 1e-10)
        ).alias("volume_imbalance"),

        # Trade count imbalance [-1, 1]
        (
            (pl.col("buy_trades") - pl.col("sell_trades"))
            / (pl.col("buy_trades") + pl.col("sell_trades") + 1e-10)
        ).alias("trade_imbalance"),

        # Price range
        (pl.col("high") - pl.col("low")).alias("price_range"),

        # Log return
        (pl.col("close") / pl.col("open")).log().alias("log_return"),
    )

    # Rename bucket back to ts
    agg = agg.rename({"ts_bucket": "ts"}).sort("ts")

    return agg


def add_rolling_flow_features(
    df: pl.DataFrame,
    windows: list[int] = [5, 15, 60, 300],
) -> pl.DataFrame:
    """Add rolling window features for order flow analysis.

    Args:
        df: DataFrame with trade flow features (from compute_trade_flow_features)
        windows: List of window sizes (in number of bars)

    Returns:
        DataFrame with additional rolling features
    """
    has_symbol = "symbol" in df.columns
    df = df.sort("ts")

    for w in windows:
        suffix = f"_{w}"

        # Rolling aggregations (use over("symbol") if multi-symbol)
        if has_symbol:
            df = df.with_columns(
                # Rolling volume
                pl.col("volume").rolling_sum(window_size=w).over("symbol").alias(f"vol_sum{suffix}"),

                # Rolling volume imbalance (mean)
                pl.col("volume_imbalance").rolling_mean(window_size=w).over("symbol").alias(f"vol_imb_mean{suffix}"),

                # Rolling trade imbalance (mean)
                pl.col("trade_imbalance").rolling_mean(window_size=w).over("symbol").alias(f"trade_imb_mean{suffix}"),

                # Rolling return volatility
                pl.col("log_return").rolling_std(window_size=w).over("symbol").alias(f"ret_std{suffix}"),

                # Cumulative net volume
                pl.col("net_volume").rolling_sum(window_size=w).over("symbol").alias(f"net_vol_sum{suffix}"),

                # VWAP deviation
                (
                    (pl.col("close") - pl.col("vwap").rolling_mean(window_size=w).over("symbol"))
                    / pl.col("close")
                ).alias(f"vwap_dev{suffix}"),
            )
        else:
            df = df.with_columns(
                pl.col("volume").rolling_sum(window_size=w).alias(f"vol_sum{suffix}"),
                pl.col("volume_imbalance").rolling_mean(window_size=w).alias(f"vol_imb_mean{suffix}"),
                pl.col("trade_imbalance").rolling_mean(window_size=w).alias(f"trade_imb_mean{suffix}"),
                pl.col("log_return").rolling_std(window_size=w).alias(f"ret_std{suffix}"),
                pl.col("net_volume").rolling_sum(window_size=w).alias(f"net_vol_sum{suffix}"),
                (
                    (pl.col("close") - pl.col("vwap").rolling_mean(window_size=w))
                    / pl.col("close")
                ).alias(f"vwap_dev{suffix}"),
            )

    return df


def add_return_labels(
    df: pl.DataFrame,
    horizons: list[int] = [5, 30, 60],
    price_col: str = "close",
) -> pl.DataFrame:
    """Add future return labels for supervised learning.

    Args:
        df: DataFrame with price data
        horizons: List of future horizons (in number of bars)
        price_col: Price column to use for returns

    Returns:
        DataFrame with return labels
    """
    has_symbol = "symbol" in df.columns
    df = df.sort("ts")

    for h in horizons:
        if has_symbol:
            future_price = pl.col(price_col).shift(-h).over("symbol")
        else:
            future_price = pl.col(price_col).shift(-h)

        # Log return
        df = df.with_columns(
            (future_price / pl.col(price_col)).log().alias(f"y_ret_{h}")
        )

    return df


def add_direction_labels(
    df: pl.DataFrame,
    horizons: list[int] = [5, 30, 60],
    deadzone_bps: float = 5.0,
) -> pl.DataFrame:
    """Add direction classification labels.

    Args:
        df: DataFrame with return labels (y_ret_*)
        horizons: List of horizons matching y_ret_* columns
        deadzone_bps: Deadzone threshold in basis points

    Returns:
        DataFrame with direction labels (-1, 0, +1)
    """
    threshold = deadzone_bps * 1e-4  # Convert bps to decimal

    for h in horizons:
        ret_col = f"y_ret_{h}"
        if ret_col not in df.columns:
            continue

        df = df.with_columns(
            pl.when(pl.col(ret_col) > threshold)
            .then(1)
            .when(pl.col(ret_col) < -threshold)
            .then(-1)
            .otherwise(0)
            .alias(f"y_dir_{h}")
        )

    return df


def build_ml_dataset(
    df: pl.DataFrame,
    interval: str = "1s",
    rolling_windows: list[int] = [5, 15, 60, 300],
    horizons: list[int] = [5, 30, 60],
    deadzone_bps: float = 5.0,
) -> pl.DataFrame:
    """Build a complete ML-ready dataset from raw trades.

    This is the main entry point for preparing data for model training.

    Args:
        df: Raw trades DataFrame
        interval: Aggregation interval
        rolling_windows: Window sizes for rolling features
        horizons: Prediction horizons for labels
        deadzone_bps: Deadzone for direction classification

    Returns:
        ML-ready DataFrame with features and labels
    """
    # Step 1: Aggregate to OHLCV + order flow
    agg = compute_trade_flow_features(df, interval=interval)

    # Step 2: Add rolling features
    agg = add_rolling_flow_features(agg, windows=rolling_windows)

    # Step 3: Add return labels
    agg = add_return_labels(agg, horizons=horizons)

    # Step 4: Add direction labels
    agg = add_direction_labels(agg, horizons=horizons, deadzone_bps=deadzone_bps)

    return agg


def add_advanced_flow_features(
    df: pl.DataFrame,
    windows: list[int] = [5, 15, 60, 300],
) -> pl.DataFrame:
    """Add advanced order flow and microstructure features.

    These features capture more nuanced market dynamics:
    - Order flow toxicity (VPIN-inspired)
    - Trade arrival intensity
    - Large trade detection
    - Price momentum and mean reversion
    - Volume-price correlation
    - Microstructure signals

    Args:
        df: DataFrame with basic trade flow features
        windows: Window sizes for rolling calculations

    Returns:
        DataFrame with advanced features
    """
    has_symbol = "symbol" in df.columns
    df = df.sort("ts")

    # =========================================================================
    # Base derived features
    # =========================================================================

    df = df.with_columns(
        # Relative spread proxy (high-low range as % of close)
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("rel_range"),

        # Trade size (avg volume per trade)
        (pl.col("volume") / (pl.col("num_trades") + 1e-10)).alias("avg_trade_size"),

        # Buy/Sell trade size ratio
        (
            (pl.col("buy_volume") / (pl.col("buy_trades") + 1e-10))
            / (pl.col("sell_volume") / (pl.col("sell_trades") + 1e-10) + 1e-10)
        ).alias("buy_sell_size_ratio"),

        # Signed volume (for momentum)
        (pl.col("net_volume") * pl.col("log_return").sign()).alias("signed_volume"),

        # Price efficiency (close vs VWAP deviation)
        ((pl.col("close") - pl.col("vwap")) / pl.col("close")).alias("price_efficiency"),

        # Trade intensity (trades per unit volume)
        (pl.col("num_trades") / (pl.col("volume") + 1e-10)).alias("trade_intensity"),
    )

    # =========================================================================
    # Rolling features - process per symbol if multi-symbol
    # =========================================================================

    if has_symbol:
        # Process each symbol separately to avoid .over() issues
        symbols = df["symbol"].unique().to_list()
        result_dfs = []

        for sym in symbols:
            sym_df = df.filter(pl.col("symbol") == sym).sort("ts")
            sym_df = _add_advanced_rolling_single(sym_df, windows)
            result_dfs.append(sym_df)

        df = pl.concat(result_dfs).sort(["symbol", "ts"])
    else:
        df = _add_advanced_rolling_single(df, windows)

    return df


def _add_advanced_rolling_single(df: pl.DataFrame, windows: list[int]) -> pl.DataFrame:
    """Add advanced rolling features for a single symbol (no grouping needed)."""

    # Pre-compute shifted columns needed for autocorrelation
    df = df.with_columns(
        pl.col("log_return").shift(1).alias("_log_return_lag1")
    )
    df = df.with_columns(
        (pl.col("log_return") * pl.col("_log_return_lag1")).alias("_ret_cross")
    )

    for w in windows:
        suffix = f"_{w}"

        df = df.with_columns(
            # Order Flow Toxicity
            pl.col("volume_imbalance").abs().rolling_mean(window_size=w).alias(f"toxicity{suffix}"),
            pl.col("volume_imbalance").rolling_std(window_size=w).alias(f"imb_volatility{suffix}"),

            # Trade Arrival
            pl.col("num_trades").rolling_mean(window_size=w).alias(f"trade_rate{suffix}"),
            (
                pl.col("volume").rolling_mean(window_size=w)
                - pl.col("volume").rolling_mean(window_size=w).shift(w)
            ).alias(f"vol_accel{suffix}"),

            # Large Trade Detection
            (
                pl.col("avg_trade_size")
                / (pl.col("avg_trade_size").rolling_mean(window_size=w) + 1e-10)
            ).alias(f"trade_size_ratio{suffix}"),

            # Momentum
            pl.col("log_return").rolling_sum(window_size=w).alias(f"momentum{suffix}"),
            pl.col("_ret_cross").rolling_mean(window_size=w).alias(f"ret_autocorr{suffix}"),

            # Volume-Price
            pl.col("signed_volume").rolling_sum(window_size=w).alias(f"kyle_lambda{suffix}"),
            (
                pl.col("log_return").rolling_sum(window_size=w)
                / (pl.col("volume").rolling_sum(window_size=w) + 1e-10)
            ).alias(f"price_impact{suffix}"),

            # Microstructure
            pl.col("price_efficiency").abs().rolling_mean(window_size=w).alias(f"vwap_tracking{suffix}"),
            (
                pl.col("rel_range")
                / (pl.col("rel_range").rolling_mean(window_size=w) + 1e-10)
            ).alias(f"range_ratio{suffix}"),
            (
                pl.col("trade_intensity").rolling_mean(window_size=w)
                / (pl.col("trade_intensity").rolling_mean(window_size=w*2) + 1e-10)
            ).alias(f"intensity_momentum{suffix}"),
        )

    # Clean up temp columns
    df = df.drop(["_log_return_lag1", "_ret_cross"])

    return df


def add_volatility_labels(
    df: pl.DataFrame,
    horizons: list[int] = [5, 30, 60],
) -> pl.DataFrame:
    """Add future volatility labels for volatility prediction.

    Args:
        df: DataFrame with log_return column
        horizons: List of future horizons (in number of bars)

    Returns:
        DataFrame with volatility labels
    """
    has_symbol = "symbol" in df.columns
    df = df.sort("ts")

    # First compute squared returns
    df = df.with_columns(
        (pl.col("log_return") ** 2).alias("_ret_sq")
    )

    for h in horizons:
        if has_symbol:
            # Sum of squared returns over next h bars
            future_vol = (
                pl.col("_ret_sq")
                .rolling_sum(window_size=h)
                .shift(-h)
                .over("symbol")
                .sqrt()
            )
        else:
            future_vol = (
                pl.col("_ret_sq")
                .rolling_sum(window_size=h)
                .shift(-h)
                .sqrt()
            )

        df = df.with_columns(future_vol.alias(f"y_vol_{h}"))

    df = df.drop("_ret_sq")
    return df


def merge_multi_symbol(
    dfs: dict[str, pl.DataFrame],
    target_symbol: str,
    feature_symbols: list[str] | None = None,
    suffix_format: str = "_{symbol}",
) -> pl.DataFrame:
    """Merge features from multiple symbols for cross-asset modeling.

    Args:
        dfs: Dictionary mapping symbol -> DataFrame
        target_symbol: Primary symbol (labels come from this)
        feature_symbols: Symbols to include as features (default: all except target)
        suffix_format: Format string for column suffixes

    Returns:
        Merged DataFrame with cross-asset features
    """
    if target_symbol not in dfs:
        raise ValueError(f"Target symbol {target_symbol} not found in data")

    # Start with target symbol
    result = dfs[target_symbol].clone()

    # Determine feature symbols
    if feature_symbols is None:
        feature_symbols = [s for s in dfs.keys() if s != target_symbol]

    # Feature columns to merge (exclude labels)
    feature_cols = [
        c for c in result.columns
        if not c.startswith("y_") and c not in ["ts", "symbol"]
    ]

    # Merge each feature symbol
    for sym in feature_symbols:
        if sym not in dfs:
            continue

        sym_df = dfs[sym].select(
            ["ts"] + feature_cols
        )

        # Rename columns with symbol suffix
        suffix = suffix_format.format(symbol=sym.lower())
        rename_map = {c: f"{c}{suffix}" for c in feature_cols}
        sym_df = sym_df.rename(rename_map)

        # Join on timestamp
        result = result.join(sym_df, on="ts", how="left")

    return result
