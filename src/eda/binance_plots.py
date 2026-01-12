"""Plotly visualizations for Binance order flow data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import save_fig


def plot_ohlcv_candlestick(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    max_rows: int = 500,
    resample_interval: str = "1min",
) -> Path | None:
    """Plot OHLCV candlestick chart with volume bars.

    For large datasets, resamples to a higher timeframe for cleaner visualization.
    """
    required = ["ts", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        return None

    # Resample to higher timeframe if too many rows
    if len(df) > max_rows:
        df = df.set_index("ts")
        df_resampled = df.resample(resample_interval).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna().reset_index()
        df = df_resampled
        title_suffix = f" ({resample_interval} bars)"
    else:
        title_suffix = ""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["ts"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Volume bars with color
    colors = ["#ef5350" if c < o else "#26a69a" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df["ts"], y=df["volume"], marker_color=colors, name="Volume", opacity=0.7),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{prefix} OHLCV{title_suffix}",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=True,
        xaxis2_rangeslider_thickness=0.05,
        height=700,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # Add range selector buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=4, label="4h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            bgcolor="#e0e0e0",
        ),
        row=1, col=1,
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    out_path = out_dir / f"{prefix}_ohlcv_candlestick.html"
    save_fig(fig, out_path)
    return out_path


def plot_order_flow_imbalance(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    max_rows: int = 2000,
) -> list[Path]:
    """Plot order flow imbalance metrics."""
    paths = []

    if len(df) > max_rows:
        df = df.iloc[::len(df)//max_rows]

    # Volume imbalance over time
    if "volume_imbalance" in df.columns and "ts" in df.columns:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
        )

        # Price with imbalance color
        if "close" in df.columns:
            fig.add_trace(
                go.Scatter(x=df["ts"], y=df["close"], mode="lines", name="Price"),
                row=1, col=1,
            )

        # Imbalance as bar chart
        colors = ["green" if v > 0 else "red" for v in df["volume_imbalance"]]
        fig.add_trace(
            go.Bar(x=df["ts"], y=df["volume_imbalance"], marker_color=colors, name="Vol Imbalance"),
            row=2, col=1,
        )

        fig.update_layout(title=f"{prefix} Volume Imbalance", height=500)
        out_path = out_dir / f"{prefix}_volume_imbalance.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # Trade imbalance histogram
    if "trade_imbalance" in df.columns:
        fig = px.histogram(
            df, x="trade_imbalance", nbins=50,
            title=f"{prefix} Trade Imbalance Distribution",
            color_discrete_sequence=["steelblue"],
        )
        out_path = out_dir / f"{prefix}_trade_imbalance_hist.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_toxicity_features(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    windows: list[int] = [5, 15, 60],
    max_rows: int = 2000,
) -> list[Path]:
    """Plot toxicity (VPIN-inspired) features across windows."""
    paths = []

    if len(df) > max_rows:
        df = df.iloc[::len(df)//max_rows]

    toxicity_cols = [f"toxicity_{w}" for w in windows if f"toxicity_{w}" in df.columns]

    if toxicity_cols and "ts" in df.columns:
        fig = go.Figure()
        for col in toxicity_cols:
            fig.add_trace(go.Scatter(x=df["ts"], y=df[col], mode="lines", name=col))

        fig.update_layout(
            title=f"{prefix} Order Flow Toxicity (VPIN)",
            xaxis_title="Time",
            yaxis_title="Toxicity",
            height=400,
        )
        out_path = out_dir / f"{prefix}_toxicity_timeseries.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_kyle_lambda(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    windows: list[int] = [5, 15, 60],
    max_rows: int = 2000,
) -> list[Path]:
    """Plot Kyle's lambda (price impact) features."""
    paths = []

    if len(df) > max_rows:
        df = df.iloc[::len(df)//max_rows]

    kyle_cols = [f"kyle_lambda_{w}" for w in windows if f"kyle_lambda_{w}" in df.columns]

    if kyle_cols and "ts" in df.columns:
        fig = go.Figure()
        for col in kyle_cols:
            fig.add_trace(go.Scatter(x=df["ts"], y=df[col], mode="lines", name=col))

        fig.update_layout(
            title=f"{prefix} Kyle's Lambda (Price Impact)",
            xaxis_title="Time",
            yaxis_title="Kyle Lambda",
            height=400,
        )
        out_path = out_dir / f"{prefix}_kyle_lambda_timeseries.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_momentum_features(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    windows: list[int] = [5, 15, 60],
    max_rows: int = 2000,
) -> list[Path]:
    """Plot momentum and autocorrelation features."""
    paths = []

    if len(df) > max_rows:
        df = df.iloc[::len(df)//max_rows]

    # Momentum
    mom_cols = [f"momentum_{w}" for w in windows if f"momentum_{w}" in df.columns]
    if mom_cols and "ts" in df.columns:
        fig = go.Figure()
        for col in mom_cols:
            fig.add_trace(go.Scatter(x=df["ts"], y=df[col], mode="lines", name=col))

        fig.update_layout(
            title=f"{prefix} Return Momentum",
            xaxis_title="Time",
            yaxis_title="Cumulative Return",
            height=400,
        )
        out_path = out_dir / f"{prefix}_momentum_timeseries.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # Autocorrelation
    autocorr_cols = [f"ret_autocorr_{w}" for w in windows if f"ret_autocorr_{w}" in df.columns]
    if autocorr_cols and "ts" in df.columns:
        fig = go.Figure()
        for col in autocorr_cols:
            fig.add_trace(go.Scatter(x=df["ts"], y=df[col], mode="lines", name=col))

        fig.update_layout(
            title=f"{prefix} Return Autocorrelation",
            xaxis_title="Time",
            yaxis_title="Autocorrelation",
            height=400,
        )
        out_path = out_dir / f"{prefix}_autocorr_timeseries.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_feature_importance(
    importance: list[tuple[str, float]],
    out_dir: Path,
    prefix: str,
    top_k: int = 20,
) -> Path:
    """Plot feature importance as horizontal bar chart."""
    names = [x[0] for x in importance[:top_k]]
    values = [x[1] for x in importance[:top_k]]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=names[::-1],
        orientation="h",
        marker_color="steelblue",
    ))

    fig.update_layout(
        title=f"{prefix} Top {top_k} Feature Importance",
        xaxis_title="Importance (abs coefficient)",
        yaxis_title="Feature",
        height=max(400, top_k * 25),
    )

    out_path = out_dir / f"{prefix}_feature_importance.html"
    save_fig(fig, out_path)
    return out_path


def plot_label_distribution(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    label_cols: list[str] | None = None,
) -> list[Path]:
    """Plot label distributions for direction prediction."""
    paths = []

    if label_cols is None:
        label_cols = [c for c in df.columns if c.startswith("y_dir_")]

    for col in label_cols:
        if col not in df.columns:
            continue

        vc = df[col].value_counts().sort_index()
        labels = {-1: "Down", 0: "Neutral", 1: "Up"}
        x_labels = [labels.get(i, str(i)) for i in vc.index]
        colors = ["red", "gray", "green"][:len(vc)]

        fig = go.Figure(go.Bar(
            x=x_labels,
            y=vc.values,
            marker_color=colors,
            text=[f"{v/len(df)*100:.1f}%" for v in vc.values],
            textposition="auto",
        ))

        fig.update_layout(
            title=f"{prefix} {col} Distribution",
            xaxis_title="Direction",
            yaxis_title="Count",
            height=350,
        )

        out_path = out_dir / f"{prefix}_{col}_distribution.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_feature_correlations(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    feature_groups: dict[str, list[str]] | None = None,
) -> list[Path]:
    """Plot correlation heatmaps for feature groups."""
    paths = []

    if feature_groups is None:
        # Default groups for order flow features
        feature_groups = {
            "order_flow": ["volume_imbalance", "trade_imbalance", "net_volume", "buy_volume", "sell_volume"],
            "microstructure": ["spread", "microprice", "vwap", "close"],
            "labels": [c for c in df.columns if c.startswith("y_")],
        }

    for group_name, cols in feature_groups.items():
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) < 2:
            continue

        corr = df[valid_cols].corr(method="spearman")

        fig = px.imshow(
            corr,
            text_auto=".2f",
            title=f"{prefix} {group_name} Correlation (Spearman)",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )

        out_path = out_dir / f"{prefix}_{group_name}_correlation.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def generate_binance_plots(
    df: pd.DataFrame,
    split_name: str,
    out_dir: Path,
    windows: list[int] = [5, 15, 60],
) -> list[Path]:
    """Generate all Binance-specific plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    # Convert to pandas if polars
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    # OHLCV candlestick
    ohlcv_path = plot_ohlcv_candlestick(df, out_dir, split_name)
    if ohlcv_path:
        paths.append(ohlcv_path)

    # Order flow imbalance
    paths.extend(plot_order_flow_imbalance(df, out_dir, split_name))

    # Toxicity features
    paths.extend(plot_toxicity_features(df, out_dir, split_name, windows))

    # Kyle's lambda
    paths.extend(plot_kyle_lambda(df, out_dir, split_name, windows))

    # Momentum features
    paths.extend(plot_momentum_features(df, out_dir, split_name, windows))

    # Label distribution
    paths.extend(plot_label_distribution(df, out_dir, split_name))

    # Feature correlations
    paths.extend(plot_feature_correlations(df, out_dir, split_name))

    return paths
