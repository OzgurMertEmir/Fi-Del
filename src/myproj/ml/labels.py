"""Label builders for price move and execution tasks."""

from __future__ import annotations


import polars as pl


def make_price_move_labels(
    df: pl.DataFrame, horizon_steps: int = 5, deadband: float = 5e-4
) -> pl.DataFrame:
    """Add future price move labels using last fill price as proxy for mid."""
    df = df.sort(["symbol", "ts"])
    df = df.with_columns(
        pl.col("last_fill_price").shift(-horizon_steps).over("symbol").alias("future_price")
    )
    df = df.with_columns(
        ((pl.col("future_price") - pl.col("last_fill_price")) / pl.col("last_fill_price")).alias("future_return")
    )
    label = (
        pl.when(pl.col("future_return") > deadband)
        .then(pl.lit(1))
        .when(pl.col("future_return") < -deadband)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
    )
    df = df.with_columns(label.alias("price_move_cls"))
    return df


def make_execution_labels(df: pl.DataFrame, horizon_steps: int = 5, source_col: str = "sum_filled_qty") -> pl.DataFrame:
    """Add execution labels using a specified fill quantity source over the next horizon."""
    future_sum = sum(
        pl.col(source_col).shift(-i).over("symbol") for i in range(1, horizon_steps + 1)
    )
    df = df.with_columns(
        future_sum.alias("future_fill_sum"),
    ).with_columns(
        (pl.col("future_fill_sum") > 0).cast(pl.Int8).alias("will_fill"),
    )
    return df
