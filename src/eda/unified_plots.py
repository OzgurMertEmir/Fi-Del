"""Plotly visualizations for unified multi-source data."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import save_fig


def plot_price_timeline(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    max_rows: int = 5000,
) -> Optional[Path]:
    """Plot price data over time with multiple price sources."""
    # Identify price columns
    price_cols = [c for c in df.columns if any(x in c.lower() for x in ["price", "close", "mid"]) and "lag" not in c and "ma" not in c]

    if not price_cols:
        return None

    if len(df) > max_rows:
        df = df.iloc[:: len(df) // max_rows]

    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    for i, col in enumerate(price_cols[:5]):  # Limit to 5 price series
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index if "timestamp" not in df.columns else df["timestamp"],
                y=df[col],
                name=col,
                line=dict(color=colors[i % len(colors)]),
                opacity=0.8,
            ))

    fig.update_layout(
        title=f"{prefix} Price Data Over Time",
        xaxis_title="Time",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )

    out_path = out_dir / f"{prefix}_price_timeline.html"
    save_fig(fig, out_path)
    return out_path


def plot_signal_vs_fills(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
) -> List[Path]:
    """Plot signal and fill data relationship."""
    paths = []

    # Signal distribution by label
    if "last_signal" in df.columns and "label" in df.columns:
        fig = px.histogram(
            df,
            x="last_signal",
            color="label",
            barmode="group",
            title=f"{prefix} Signal Distribution by Label",
            height=400,
        )
        fig.update_layout(xaxis_title="Signal Value", yaxis_title="Count")
        out_path = out_dir / f"{prefix}_signal_by_label.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # Fill volume by signal
    if "last_signal" in df.columns and "fill_volume" in df.columns:
        signal_fill = df.groupby("last_signal")["fill_volume"].sum().reset_index()
        fig = px.bar(
            signal_fill,
            x="last_signal",
            y="fill_volume",
            title=f"{prefix} Fill Volume by Signal",
            height=400,
        )
        fig.update_layout(xaxis_title="Signal Value", yaxis_title="Total Fill Volume")
        out_path = out_dir / f"{prefix}_fill_volume_by_signal.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_order_book_features(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    max_rows: int = 2000,
) -> List[Path]:
    """Plot order book features (from .store data)."""
    paths = []

    if len(df) > max_rows:
        df = df.iloc[:: len(df) // max_rows]

    # Spread over time
    if "avg_spread_bps" in df.columns:
        fig = px.line(
            df,
            x="timestamp" if "timestamp" in df.columns else df.index,
            y="avg_spread_bps",
            title=f"{prefix} Spread (bps) Over Time",
            height=400,
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Spread (bps)")
        out_path = out_dir / f"{prefix}_spread_timeline.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # Book imbalance over time
    if "avg_book_imbalance" in df.columns:
        fig = px.line(
            df,
            x="timestamp" if "timestamp" in df.columns else df.index,
            y="avg_book_imbalance",
            title=f"{prefix} Book Imbalance Over Time",
            height=400,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_title="Time", yaxis_title="Book Imbalance")
        out_path = out_dir / f"{prefix}_book_imbalance.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_binance_features(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    max_rows: int = 2000,
) -> List[Path]:
    """Plot Binance order flow features."""
    paths = []

    if len(df) > max_rows:
        df = df.iloc[:: len(df) // max_rows]

    # Volume imbalance
    if "binance_vol_imbalance" in df.columns:
        fig = px.line(
            df,
            x="timestamp" if "timestamp" in df.columns else df.index,
            y="binance_vol_imbalance",
            title=f"{prefix} Binance Volume Imbalance",
            height=400,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        out_path = out_dir / f"{prefix}_binance_vol_imbalance.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # Toxicity if available
    if "binance_toxicity" in df.columns:
        fig = px.line(
            df,
            x="timestamp" if "timestamp" in df.columns else df.index,
            y="binance_toxicity",
            title=f"{prefix} Binance Order Flow Toxicity",
            height=400,
        )
        out_path = out_dir / f"{prefix}_binance_toxicity.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    # OHLCV candlestick if available
    binance_cols = [c for c in df.columns if c.startswith("binance_")]
    if all(f"binance_{x}" in binance_cols for x in ["open", "high", "low", "close"]):
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"] if "timestamp" in df.columns else df.index,
            open=df["binance_open"],
            high=df["binance_high"],
            low=df["binance_low"],
            close=df["binance_close"],
        )])
        fig.update_layout(
            title=f"{prefix} Binance OHLC",
            xaxis_title="Time",
            yaxis_title="Price",
            height=500,
        )
        out_path = out_dir / f"{prefix}_binance_ohlc.html"
        save_fig(fig, out_path)
        paths.append(out_path)

    return paths


def plot_feature_importance(
    importance: List[tuple],
    out_dir: Path,
    prefix: str,
    top_k: int = 20,
) -> Optional[Path]:
    """Plot feature importance from model."""
    if not importance:
        return None

    names = [x[0] for x in importance[:top_k]]
    values = [x[1] for x in importance[:top_k]]

    # Color by source
    colors = []
    for name in names:
        if name.startswith("binance_"):
            colors.append("steelblue")
        elif any(x in name for x in ["signal", "strategy", "entry"]):
            colors.append("green")
        elif any(x in name for x in ["fill", "volume", "buy", "sell"]):
            colors.append("orange")
        elif any(x in name for x in ["bid", "ask", "spread", "book"]):
            colors.append("purple")
        else:
            colors.append("gray")

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=names[::-1],
        orientation="h",
        marker_color=colors[::-1],
    ))

    fig.update_layout(
        title=f"{prefix} Feature Importance by Source",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, len(names) * 25),
    )

    out_path = out_dir / f"{prefix}_feature_importance.html"
    save_fig(fig, out_path)
    return out_path


def plot_label_distribution(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
) -> Optional[Path]:
    """Plot label distribution."""
    if "label" not in df.columns:
        return None

    label_counts = df["label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]

    colors = ["red" if lbl == 0 else "green" for lbl in label_counts["label"]]

    fig = go.Figure(go.Bar(
        x=label_counts["label"].astype(str),
        y=label_counts["count"],
        marker_color=colors,
        text=label_counts["count"],
        textposition="auto",
    ))

    fig.update_layout(
        title=f"{prefix} Label Distribution",
        xaxis_title="Label",
        yaxis_title="Count",
        height=400,
    )

    out_path = out_dir / f"{prefix}_label_distribution.html"
    save_fig(fig, out_path)
    return out_path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    max_features: int = 30,
) -> Optional[Path]:
    """Plot correlation heatmap for numeric features."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Filter out lag/ma/std columns to keep it readable
    base_cols = [c for c in numeric_cols if "_lag" not in c and "_ma" not in c and "_std" not in c]

    if len(base_cols) < 2:
        return None

    # Limit columns
    cols = base_cols[:max_features]
    corr = df[cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=cols,
        y=cols,
        colorscale="RdBu_r",
        zmid=0,
    ))

    fig.update_layout(
        title=f"{prefix} Feature Correlation",
        height=max(500, len(cols) * 20),
        width=max(600, len(cols) * 20),
    )

    out_path = out_dir / f"{prefix}_correlation.html"
    save_fig(fig, out_path)
    return out_path


def plot_symbol_timeseries(
    df: pd.DataFrame,
    out_dir: Path,
    symbol: str,
    max_rows: int = 3000,
) -> Optional[Path]:
    """Generate comprehensive timeseries dashboard for a single symbol.

    Creates a multi-panel plot showing all available timeseries data.
    """
    if len(df) == 0:
        return None

    if len(df) > max_rows:
        df = df.iloc[:: len(df) // max_rows]

    # Determine x-axis
    x_col = df["timestamp"] if "timestamp" in df.columns else df.index

    # Count available panels
    panels = []

    # Panel 1: Price data
    price_cols = [c for c in df.columns if any(x in c.lower() for x in ["price", "close", "mid"])
                  and "lag" not in c and "ma" not in c and "std" not in c][:4]
    if price_cols:
        panels.append(("Price", price_cols))

    # Panel 2: Volume/Fill data
    vol_cols = [c for c in df.columns if any(x in c.lower() for x in ["volume", "n_fills"])
                and "lag" not in c and "ma" not in c][:3]
    if vol_cols:
        panels.append(("Volume", vol_cols))

    # Panel 3: Signal data
    signal_cols = [c for c in df.columns if "signal" in c.lower()
                   and "lag" not in c and "ma" not in c and "std" not in c][:3]
    if signal_cols:
        panels.append(("Signal", signal_cols))

    # Panel 4: Spread/Book data
    book_cols = [c for c in df.columns if any(x in c.lower() for x in ["spread", "imbalance"])
                 and "lag" not in c and "ma" not in c][:3]
    if book_cols:
        panels.append(("Order Book", book_cols))

    # Panel 5: Label
    if "label" in df.columns:
        panels.append(("Label", ["label"]))

    if not panels:
        return None

    # Create subplots
    n_panels = len(panels)
    heights = [0.3 if i == 0 else 0.2 for i in range(n_panels)]  # Price panel larger

    fig = make_subplots(
        rows=n_panels,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
        subplot_titles=[p[0] for p in panels],
    )

    colors = px.colors.qualitative.Set2

    for row_idx, (panel_name, cols) in enumerate(panels, 1):
        for col_idx, col in enumerate(cols):
            if col not in df.columns:
                continue

            fig.add_trace(
                go.Scatter(
                    x=x_col,
                    y=df[col],
                    name=col,
                    line=dict(color=colors[col_idx % len(colors)], width=1),
                    showlegend=True,
                    legendgroup=panel_name,
                ),
                row=row_idx,
                col=1,
            )

    fig.update_layout(
        title=f"{symbol} - Multi-Source Timeseries",
        height=200 + n_panels * 180,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
    )

    # Update x-axis only on bottom panel
    fig.update_xaxes(title_text="Time", row=n_panels, col=1)

    out_path = out_dir / f"{symbol}_timeseries.html"
    save_fig(fig, out_path)
    return out_path


def plot_symbol_features_grid(
    df: pd.DataFrame,
    out_dir: Path,
    symbol: str,
) -> Optional[Path]:
    """Generate a 2x2 grid of key feature plots for a symbol."""
    if len(df) == 0:
        return None

    x_col = df["timestamp"] if "timestamp" in df.columns else df.index

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Price vs Signal", "Volume Analysis", "Fill Imbalance", "Label Over Time"],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Plot 1: Price with signal overlay
    price_col = None
    for c in ["avg_fill_price", "last_fill_price", "signal_mid_price", "mid_price"]:
        if c in df.columns:
            price_col = c
            break

    if price_col:
        fig.add_trace(
            go.Scatter(x=x_col, y=df[price_col], name=price_col, line=dict(color="steelblue")),
            row=1, col=1
        )

    if "last_signal" in df.columns:
        fig.add_trace(
            go.Scatter(x=x_col, y=df["last_signal"], name="Signal",
                      line=dict(color="orange"), yaxis="y2"),
            row=1, col=1
        )

    # Plot 2: Volume
    if "fill_volume" in df.columns:
        fig.add_trace(
            go.Bar(x=x_col, y=df["fill_volume"], name="Fill Volume", marker_color="green", opacity=0.6),
            row=1, col=2
        )
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        fig.add_trace(
            go.Scatter(x=x_col, y=df["buy_volume"], name="Buy Vol", line=dict(color="green")),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=x_col, y=df["sell_volume"], name="Sell Vol", line=dict(color="red")),
            row=1, col=2
        )

    # Plot 3: Fill imbalance
    if "fill_imbalance" in df.columns:
        colors = ["green" if v > 0 else "red" for v in df["fill_imbalance"].fillna(0)]
        fig.add_trace(
            go.Bar(x=x_col, y=df["fill_imbalance"], name="Fill Imbalance", marker_color=colors),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Plot 4: Label
    if "label" in df.columns:
        colors = ["green" if v == 1 else "red" for v in df["label"].fillna(0)]
        fig.add_trace(
            go.Scatter(x=x_col, y=df["label"], name="Label", mode="markers",
                      marker=dict(color=colors, size=4)),
            row=2, col=2
        )

    fig.update_layout(
        title=f"{symbol} - Feature Analysis",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    out_path = out_dir / f"{symbol}_features.html"
    save_fig(fig, out_path)
    return out_path


def generate_unified_plots(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str = "unified",
    feature_importance: Optional[List[tuple]] = None,
) -> List[Path]:
    """Generate all unified data plots.

    Args:
        df: Unified DataFrame (pandas)
        out_dir: Output directory
        prefix: Filename prefix
        feature_importance: Optional feature importance list

    Returns:
        List of generated plot paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []

    # Convert polars to pandas if needed
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    # Price timeline
    price_path = plot_price_timeline(df, out_dir, prefix)
    if price_path:
        paths.append(price_path)

    # Signal vs fills
    paths.extend(plot_signal_vs_fills(df, out_dir, prefix))

    # Order book features
    paths.extend(plot_order_book_features(df, out_dir, prefix))

    # Binance features
    paths.extend(plot_binance_features(df, out_dir, prefix))

    # Label distribution
    label_path = plot_label_distribution(df, out_dir, prefix)
    if label_path:
        paths.append(label_path)

    # Correlation heatmap
    corr_path = plot_correlation_heatmap(df, out_dir, prefix)
    if corr_path:
        paths.append(corr_path)

    # Feature importance
    if feature_importance:
        imp_path = plot_feature_importance(feature_importance, out_dir, prefix)
        if imp_path:
            paths.append(imp_path)

    return paths
