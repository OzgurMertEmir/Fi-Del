"""Plotly visualizations for log-based signal and fill data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .plots import save_fig


def plot_signal_timeline(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    max_rows: int = 5000,
) -> Path | None:
    """Plot signal values over time by symbol."""
    if "signal_ts" not in df.columns or "signal_value" not in df.columns:
        return None

    if len(df) > max_rows:
        df = df.iloc[:: len(df) // max_rows]

    # Convert to pandas if polars
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    fig = px.scatter(
        df,
        x="signal_ts",
        y="signal_value",
        color="symbol",
        title=f"{prefix} Signal Values Over Time",
        opacity=0.6,
        height=500,
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Signal Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    out_path = out_dir / f"{prefix}_signal_timeline.html"
    save_fig(fig, out_path)
    return out_path


def plot_signal_distribution(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """Plot signal value distributions."""
    paths = []

    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    # Signal value histogram
    if "signal_value" in df.columns:
        fig = px.histogram(
            df,
            x="signal_value",
            color="symbol",
            barmode="group",
            title=f"{prefix} Signal Value Distribution by Symbol",
            height=400,
        )
        out_path = out_dir / f"{prefix}_signal_value_dist.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # Signal by strategy
    if "strategy_id" in df.columns:
        strategy_counts = df.groupby(["strategy_id", "signal_value"]).size().reset_index(name="count")
        fig = px.bar(
            strategy_counts,
            x="strategy_id",
            y="count",
            color="signal_value",
            title=f"{prefix} Signals by Strategy ID",
            height=400,
        )
        out_path = out_dir / f"{prefix}_signals_by_strategy.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_fill_analysis(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """Plot fill-related analysis."""
    paths = []

    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    # Fill rate by symbol
    if "had_fill" in df.columns and "symbol" in df.columns:
        fill_rates = df.groupby("symbol").agg(
            total=("had_fill", "count"),
            fills=("had_fill", "sum"),
        ).reset_index()
        fill_rates["fill_rate"] = fill_rates["fills"] / fill_rates["total"]

        fig = go.Figure(data=[
            go.Bar(
                x=fill_rates["symbol"],
                y=fill_rates["fill_rate"],
                text=[f"{r:.1%}" for r in fill_rates["fill_rate"]],
                textposition="auto",
                marker_color="steelblue",
            )
        ])
        fig.update_layout(
            title=f"{prefix} Fill Rate by Symbol",
            xaxis_title="Symbol",
            yaxis_title="Fill Rate",
            yaxis_tickformat=".0%",
            height=400,
        )
        out_path = out_dir / f"{prefix}_fill_rate_by_symbol.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # Fill rate by signal value
    if "had_fill" in df.columns and "signal_value" in df.columns:
        fill_by_signal = df.groupby("signal_value").agg(
            total=("had_fill", "count"),
            fills=("had_fill", "sum"),
        ).reset_index()
        fill_by_signal["fill_rate"] = fill_by_signal["fills"] / fill_by_signal["total"]

        colors = ["red" if v < 0 else "green" if v > 0 else "gray" for v in fill_by_signal["signal_value"]]
        fig = go.Figure(data=[
            go.Bar(
                x=fill_by_signal["signal_value"].astype(str),
                y=fill_by_signal["fill_rate"],
                text=[f"{r:.1%}" for r in fill_by_signal["fill_rate"]],
                textposition="auto",
                marker_color=colors,
            )
        ])
        fig.update_layout(
            title=f"{prefix} Fill Rate by Signal Value",
            xaxis_title="Signal Value",
            yaxis_title="Fill Rate",
            yaxis_tickformat=".0%",
            height=400,
        )
        out_path = out_dir / f"{prefix}_fill_rate_by_signal.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_price_analysis(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    max_rows: int = 2000,
) -> list[Path]:
    """Plot price-related analysis."""
    paths = []

    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    if len(df) > max_rows:
        df = df.iloc[:: len(df) // max_rows]

    # Mid price over time by symbol
    if "signal_ts" in df.columns and "mid_price" in df.columns:
        fig = px.line(
            df,
            x="signal_ts",
            y="mid_price",
            color="symbol",
            title=f"{prefix} Mid Price Over Time",
            height=500,
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Mid Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        out_path = out_dir / f"{prefix}_mid_price_timeline.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # Price change distribution
    if "price_change_pct" in df.columns:
        # Filter out extreme outliers for visualization
        filtered = df[df["price_change_pct"].abs() < 1]
        fig = px.histogram(
            filtered,
            x="price_change_pct",
            nbins=100,
            title=f"{prefix} Price Change % Distribution",
            height=400,
        )
        fig.update_layout(xaxis_title="Price Change %", yaxis_title="Count")
        out_path = out_dir / f"{prefix}_price_change_dist.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_fill_volume_analysis(
    fills_df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """Plot fill volume analysis from raw fills data."""
    paths = []

    if hasattr(fills_df, "to_pandas"):
        fills_df = fills_df.to_pandas()

    # Convert qty to numeric
    fills_df["qty"] = pd.to_numeric(fills_df["qty"], errors="coerce")

    # Volume by symbol pie chart
    vol_by_symbol = fills_df.groupby("symbol")["qty"].sum().reset_index()
    fig = px.pie(
        vol_by_symbol,
        values="qty",
        names="symbol",
        title=f"{prefix} Fill Volume by Symbol",
        height=400,
    )
    out_path = out_dir / f"{prefix}_fill_volume_by_symbol.html"
    save_fig(fig, out_path)
    paths.append(out_path)

    # Volume over time
    if "ts" in fills_df.columns:
        fills_df["ts"] = pd.to_datetime(fills_df["ts"], errors="coerce")
        fills_df = fills_df.dropna(subset=["ts"])
        fills_df = fills_df.set_index("ts")
        vol_over_time = fills_df.groupby([pd.Grouper(freq="1min"), "symbol"])["qty"].sum().reset_index()

        fig = px.bar(
            vol_over_time,
            x="ts",
            y="qty",
            color="symbol",
            title=f"{prefix} Fill Volume Over Time (1-min buckets)",
            height=400,
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Volume")
        out_path = out_dir / f"{prefix}_fill_volume_timeline.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # Buy vs Sell volume
    if "side" in fills_df.columns:
        vol_by_side = fills_df.groupby("side")["qty"].sum().reset_index()
        colors = ["green" if s == "BUY" else "red" for s in vol_by_side["side"]]
        fig = go.Figure(data=[
            go.Bar(
                x=vol_by_side["side"],
                y=vol_by_side["qty"],
                marker_color=colors,
                text=[f"{v:,.2f}" for v in vol_by_side["qty"]],
                textposition="auto",
            )
        ])
        fig.update_layout(
            title=f"{prefix} Total Volume by Side",
            xaxis_title="Side",
            yaxis_title="Volume",
            height=400,
        )
        out_path = out_dir / f"{prefix}_volume_by_side.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_model_metrics(
    metrics: dict,
    feature_importance: list[tuple[str, float]],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """Plot model evaluation metrics and feature importance."""
    paths = []

    # Feature importance
    if feature_importance:
        names = [x[0] for x in feature_importance[:15]]
        values = [x[1] for x in feature_importance[:15]]

        fig = go.Figure(go.Bar(
            x=values[::-1],
            y=names[::-1],
            orientation="h",
            marker_color="steelblue",
        ))
        fig.update_layout(
            title=f"{prefix} Feature Importance",
            xaxis_title="Importance (abs coefficient)",
            yaxis_title="Feature",
            height=max(400, len(names) * 25),
        )
        out_path = out_dir / f"{prefix}_feature_importance.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def generate_log_plots(
    signals_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    out_dir: Path,
    prefix: str = "logs",
) -> list[Path]:
    """Generate all log-related plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    # Convert to pandas if needed
    for name, df in [("signals", signals_df), ("fills", fills_df), ("ml", ml_df)]:
        if hasattr(df, "to_pandas"):
            if name == "signals":
                signals_df = df.to_pandas()
            elif name == "fills":
                fills_df = df.to_pandas()
            else:
                ml_df = df.to_pandas()

    # Signal timeline
    timeline_path = plot_signal_timeline(ml_df, out_dir, prefix)
    if timeline_path:
        paths.append(timeline_path)

    # Signal distributions
    paths.extend(plot_signal_distribution(signals_df, out_dir, prefix))

    # Fill analysis
    paths.extend(plot_fill_analysis(ml_df, out_dir, prefix))

    # Price analysis
    paths.extend(plot_price_analysis(ml_df, out_dir, prefix))

    # Fill volume analysis
    paths.extend(plot_fill_volume_analysis(fills_df, out_dir, prefix))

    return paths
