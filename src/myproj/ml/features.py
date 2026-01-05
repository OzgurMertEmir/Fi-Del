"""Feature builders for strategy updates and trade fills."""

from __future__ import annotations


import polars as pl


def _bucket(df: pl.DataFrame, every: str) -> pl.DataFrame:
    """Bucket events into fixed intervals."""
    return df.with_columns(pl.col("ts").dt.truncate(every).alias("ts_bucket")).drop_nulls(subset=["ts_bucket"])


def bucket_strategy_updates(df: pl.DataFrame, every: str = "1s") -> pl.DataFrame:
    """Aggregate strategy updates per symbol per time bucket."""
    df = _bucket(df, every)
    agg = (
        df.group_by(["symbol", "ts_bucket"])
        .agg(
            pl.len().alias("n_strat_events"),
            pl.col("filled_qty").sum().alias("sum_filled_qty"),
            pl.col("remaining_qty").sum().alias("sum_remaining_qty"),
            pl.col("open_qty").sum().alias("sum_open_qty"),
            pl.col("new_open_qty").sum().alias("sum_new_open_qty"),
            pl.col("strategy_id").n_unique().alias("n_strategies"),
        )
        .with_columns(
            (
                pl.col("sum_filled_qty")
                / (pl.col("sum_filled_qty") + pl.col("sum_remaining_qty") + 1e-9)
            ).alias("fill_ratio"),
            (pl.col("sum_new_open_qty") - pl.col("sum_open_qty")).alias("delta_open"),
        )
    )
    return agg


def bucket_trade_fills(df: pl.DataFrame, every: str = "1s") -> pl.DataFrame:
    """Aggregate trade fills per symbol per time bucket."""
    df = _bucket(df, every)
    agg = (
        df.group_by(["symbol", "ts_bucket"])
        .agg(
            pl.len().alias("n_fills"),
            pl.col("fill_qty").sum().alias("sum_fill_qty"),
            pl.col("fill_price").mean().alias("mean_fill_price"),
            pl.col("fill_price").last().alias("last_fill_price"),
            (pl.when(pl.col("side") == "BUY").then(pl.col("fill_qty")).otherwise(0.0)).sum().alias("buy_qty"),
            (pl.when(pl.col("side") == "SELL").then(pl.col("fill_qty")).otherwise(0.0)).sum().alias("sell_qty"),
        )
        .with_columns(
            (
                pl.col("buy_qty") - pl.col("sell_qty")
            ).alias("net_fill_qty"),
            (
                (pl.col("buy_qty") - pl.col("sell_qty"))
                / (pl.col("buy_qty") + pl.col("sell_qty") + 1e-9)
            ).alias("fill_imbalance"),
        )
    )
    return agg


def build_feature_matrix(
    strat_bucket: pl.DataFrame,
    fills_bucket: pl.DataFrame,
    every: str = "1s",
    forward_fill: bool = True,
) -> pl.DataFrame:
    """Join aggregated strategy and fill features into a single matrix."""
    # Ensure both inputs have the same bucket column name
    strat = strat_bucket.rename({"ts_bucket": "ts"}) if "ts_bucket" in strat_bucket.columns else strat_bucket
    fills = fills_bucket.rename({"ts_bucket": "ts"}) if "ts_bucket" in fills_bucket.columns else fills_bucket

    # Build a full time grid per symbol spanning min..max timestamps at the chosen bucket
    symbols = (
        pl.concat([strat.select("symbol", "ts"), fills.select("symbol", "ts")], how="diagonal")
        .group_by("symbol")
        .agg(pl.col("ts").min().alias("start"), pl.col("ts").max().alias("end"))
    )
    grids = []
    for row in symbols.iter_rows(named=True):
        grid_ts = pl.datetime_range(start=row["start"], end=row["end"], interval=every, eager=True)
        grids.append(pl.DataFrame({"symbol": row["symbol"], "ts": grid_ts}))
    base = pl.concat(grids) if grids else pl.DataFrame({"symbol": [], "ts": []})

    feat = base.join(strat, on=["symbol", "ts"], how="left").join(fills, on=["symbol", "ts"], how="left")

    feat = feat.sort(["symbol", "ts"])
    if forward_fill:
        feat = feat.with_columns(
            pl.all().exclude(["symbol", "ts"]).forward_fill().over("symbol")
        )

    return feat
