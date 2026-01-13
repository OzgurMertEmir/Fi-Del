"""EDA dashboard for Binance public data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import save_fig


def plot_binance_ohlcv(
    df: pd.DataFrame,
    out_dir: Path,
    symbol: str,
) -> Optional[Path]:
    """Plot OHLCV candlestick chart with volume."""
    if "close" not in df.columns:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=["Price", "Volume"],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["ts"] if "ts" in df.columns else df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ["green" if c >= o else "red" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(
            x=df["ts"] if "ts" in df.columns else df.index,
            y=df["volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.7,
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f"{symbol} - OHLCV",
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )

    out_path = out_dir / f"{symbol}_ohlcv.html"
    save_fig(fig, out_path)
    return out_path


def plot_binance_order_flow(
    df: pd.DataFrame,
    out_dir: Path,
    symbol: str,
) -> Optional[Path]:
    """Plot order flow features."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Buy vs Sell Volume", "Volume Imbalance", "Net Volume"],
        row_heights=[0.4, 0.3, 0.3],
    )

    x_col = df["ts"] if "ts" in df.columns else df.index

    # Buy/Sell volume
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        fig.add_trace(
            go.Scatter(x=x_col, y=df["buy_volume"], name="Buy Volume",
                      line=dict(color="green"), fill="tozeroy"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_col, y=-df["sell_volume"], name="Sell Volume",
                      line=dict(color="red"), fill="tozeroy"),
            row=1, col=1
        )

    # Volume imbalance
    if "vol_imbalance" in df.columns:
        colors = ["green" if v > 0 else "red" for v in df["vol_imbalance"]]
        fig.add_trace(
            go.Bar(x=x_col, y=df["vol_imbalance"], name="Imbalance",
                  marker_color=colors),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Net volume
    if "net_volume" in df.columns:
        colors = ["green" if v > 0 else "red" for v in df["net_volume"]]
        fig.add_trace(
            go.Bar(x=x_col, y=df["net_volume"], name="Net Volume",
                  marker_color=colors),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    fig.update_layout(
        title=f"{symbol} - Order Flow Analysis",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    out_path = out_dir / f"{symbol}_order_flow.html"
    save_fig(fig, out_path)
    return out_path


def plot_binance_advanced_features(
    df: pd.DataFrame,
    out_dir: Path,
    symbol: str,
) -> Optional[Path]:
    """Plot advanced order flow features like toxicity."""
    panels = []
    x_col = df["ts"] if "ts" in df.columns else df.index

    # Check what features are available
    if "toxicity" in df.columns:
        panels.append(("toxicity", "Order Flow Toxicity"))
    if "kyle_lambda" in df.columns:
        panels.append(("kyle_lambda", "Kyle's Lambda (Price Impact)"))
    if "vwap" in df.columns and "close" in df.columns:
        panels.append(("vwap_diff", "VWAP vs Close"))

    if not panels:
        return None

    fig = make_subplots(
        rows=len(panels), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[p[1] for p in panels],
    )

    for i, (col, title) in enumerate(panels, 1):
        if col == "vwap_diff":
            diff = df["close"] - df["vwap"]
            fig.add_trace(
                go.Scatter(x=x_col, y=diff, name="Close - VWAP",
                          line=dict(color="purple")),
                row=i, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=i, col=1)
        elif col in df.columns:
            fig.add_trace(
                go.Scatter(x=x_col, y=df[col], name=col,
                          line=dict(color="steelblue")),
                row=i, col=1
            )

    fig.update_layout(
        title=f"{symbol} - Advanced Features",
        height=200 + len(panels) * 200,
        showlegend=False,
    )

    out_path = out_dir / f"{symbol}_advanced.html"
    save_fig(fig, out_path)
    return out_path


def plot_binance_returns(
    df: pd.DataFrame,
    out_dir: Path,
    symbol: str,
) -> Optional[Path]:
    """Plot return distribution and label analysis."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Return Distribution", "Cumulative Returns",
                       "Direction Labels", "Return by Hour"],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Return distribution
    if "ret_1" in df.columns:
        fig.add_trace(
            go.Histogram(x=df["ret_1"].dropna() * 100, nbinsx=100,
                        name="1-bar Return (%)", marker_color="steelblue"),
            row=1, col=1
        )

    # Cumulative returns
    if "close" in df.columns:
        cum_ret = (df["close"] / df["close"].iloc[0] - 1) * 100
        fig.add_trace(
            go.Scatter(x=df["ts"] if "ts" in df.columns else df.index,
                      y=cum_ret, name="Cumulative Return (%)",
                      line=dict(color="green")),
            row=1, col=2
        )

    # Direction labels distribution
    label_cols = [c for c in df.columns if c.startswith("y_dir_")]
    if label_cols:
        label_col = label_cols[0]
        label_counts = df[label_col].value_counts().sort_index()
        colors = ["red", "gray", "green"][:len(label_counts)]
        fig.add_trace(
            go.Bar(x=label_counts.index.astype(str), y=label_counts.values,
                  marker_color=colors, name="Labels"),
            row=2, col=1
        )

    # Return by hour (if timestamp available)
    if "ts" in df.columns and "ret_1" in df.columns:
        df_temp = df.copy()
        df_temp["hour"] = pd.to_datetime(df_temp["ts"]).dt.hour
        hourly_ret = df_temp.groupby("hour")["ret_1"].mean() * 100
        fig.add_trace(
            go.Bar(x=hourly_ret.index, y=hourly_ret.values,
                  marker_color="orange", name="Avg Return by Hour"),
            row=2, col=2
        )

    fig.update_layout(
        title=f"{symbol} - Return Analysis",
        height=600,
        showlegend=False,
    )

    out_path = out_dir / f"{symbol}_returns.html"
    save_fig(fig, out_path)
    return out_path


def build_binance_dashboard(
    out_dir: Path,
    summary: Dict,
    symbol_plots: Dict[str, List[Dict[str, str]]],
) -> Path:
    """Build HTML dashboard for Binance data."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "<html><head><title>Binance Data Dashboard</title><style>",
        "body { font-family: Arial, sans-serif; margin: 20px; max-width: 1600px; background: #f5f5f5; }",
        ".header { background: linear-gradient(135deg, #f7931a 0%, #ff9900 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }",
        ".header h1 { color: white; margin: 0 0 10px 0; }",
        ".header p { margin: 0; opacity: 0.9; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 20px; }",
        ".card { border: 1px solid #ddd; padding: 20px; border-radius: 12px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".card h3 { margin: 0 0 8px 0; font-size: 0.85em; color: #666; text-transform: uppercase; }",
        ".card .value { font-size: 1.6em; font-weight: bold; color: #333; }",
        ".section { background: white; padding: 24px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }",
        ".plot-link { padding: 12px 16px; background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border-radius: 8px; display: block; text-decoration: none; color: #333; transition: all 0.2s; border: 1px solid #fed7aa; }",
        ".plot-link:hover { background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%); transform: translateY(-2px); }",
        ".muted { color: #666; font-size: 0.9em; }",
        ".symbol-section { margin-bottom: 24px; padding: 20px; background: #fafafa; border-radius: 8px; }",
        ".symbol-section h3 { margin: 0 0 12px 0; color: #f7931a; }",
        "</style></head><body>",
    ]

    # Header
    lines.append("<div class='header'>")
    lines.append("<h1>Binance Public Data Dashboard</h1>")
    lines.append("<p>Order flow features from Binance historical trade data</p>")
    lines.append("</div>")

    # Summary cards
    lines.append("<div class='grid'>")
    for label, value in [
        ("Symbols", summary.get("n_symbols", 0)),
        ("Total Rows", f"{summary.get('total_rows', 0):,}"),
        ("Features", summary.get("n_features", 0)),
        ("Date Range", summary.get("date_range", "N/A")),
        ("Interval", summary.get("interval", "1s")),
    ]:
        lines.append(f"<div class='card'><h3>{label}</h3><div class='value'>{value}</div></div>")
    lines.append("</div>")

    # Per-symbol sections
    if symbol_plots:
        lines.append("<div class='section'><h2>Symbol Analysis</h2>")

        for symbol in sorted(symbol_plots.keys()):
            sym_info = summary.get("symbols", {}).get(symbol, {})
            plots = symbol_plots[symbol]

            lines.append("<div class='symbol-section'>")
            lines.append(f"<h3>{symbol}</h3>")
            lines.append(f"<p class='muted'>Rows: {sym_info.get('rows', 'N/A'):,} | ")
            lines.append(f"Features: {sym_info.get('features', 'N/A')} | ")
            lines.append(f"Price Range: ${sym_info.get('min_price', 0):,.2f} - ${sym_info.get('max_price', 0):,.2f}</p>")

            lines.append("<div class='plot-grid'>")
            for plot in plots:
                lines.append(f'<a href="{plot["href"]}" class="plot-link">{plot["label"]}</a>')
            lines.append("</div></div>")

        lines.append("</div>")

    lines.append("</body></html>")

    dashboard_path = out_dir / "index.html"
    dashboard_path.write_text("\n".join(lines), encoding="utf-8")
    return dashboard_path


def run_binance_eda(
    data_dir: str = "data/binance",
    out_dir: str = "reports/binance_eda",
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1s",
) -> Path:
    """Run EDA on Binance public data.

    Args:
        data_dir: Directory with downloaded Binance data
        out_dir: Output directory for reports
        symbols: List of symbols to analyze (None = auto-detect)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        interval: Aggregation interval

    Returns:
        Path to the generated dashboard
    """
    from datetime import datetime

    from myproj.binance_data import (
        BinancePipelineConfig,
        compute_trade_flow_features,
        add_rolling_flow_features,
        add_advanced_flow_features,
        add_return_labels,
        add_direction_labels,
    )
    from myproj.binance_data.pipeline import _load_symbol_data

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect symbols if not specified
    if symbols is None:
        trade_dirs = list((data_dir / "spot" / "trades").glob("*"))
        symbols = [d.name for d in trade_dirs if d.is_dir()]

    if not symbols:
        raise ValueError(f"No symbols found in {data_dir}")

    print(f"Analyzing symbols: {symbols}")

    # Determine date range
    if start_date is None:
        start_date = "2025-01-01"
    if end_date is None:
        end_date = "2025-01-03"

    summary = {
        "n_symbols": len(symbols),
        "total_rows": 0,
        "n_features": 0,
        "date_range": f"{start_date} to {end_date}",
        "interval": interval,
        "symbols": {},
    }

    symbol_plots: Dict[str, List[Dict[str, str]]] = {}

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")

        config = BinancePipelineConfig(
            symbols=[symbol],
            start_date=datetime.strptime(start_date, "%Y-%m-%d").date(),
            end_date=datetime.strptime(end_date, "%Y-%m-%d").date(),
            interval=interval,
            raw_data_dir=data_dir,
            download_if_missing=False,
        )

        # Load data
        df = _load_symbol_data(config, symbol)
        if df is None or len(df) == 0:
            print(f"  No data found for {symbol}")
            continue

        print(f"  Loaded {len(df):,} raw trades")

        # Compute features
        agg = compute_trade_flow_features(df, interval=interval)
        agg = add_rolling_flow_features(agg, windows=[5, 15, 60])
        agg = add_advanced_flow_features(agg, windows=[5, 15, 60])
        agg = add_return_labels(agg, horizons=[1, 5, 30])
        agg = add_direction_labels(agg, horizons=[30], deadzone_bps=5.0)

        print(f"  Aggregated to {len(agg):,} bars with {len(agg.columns)} features")

        # Convert to pandas for plotting
        df_pd = agg.to_pandas()

        # Store symbol info
        summary["symbols"][symbol] = {
            "rows": len(df_pd),
            "features": len(df_pd.columns),
            "min_price": float(df_pd["low"].min()) if "low" in df_pd.columns else 0,
            "max_price": float(df_pd["high"].max()) if "high" in df_pd.columns else 0,
        }
        summary["total_rows"] += len(df_pd)
        summary["n_features"] = max(summary["n_features"], len(df_pd.columns))

        # Generate plots
        plots = []

        ohlcv_path = plot_binance_ohlcv(df_pd, plots_dir, symbol)
        if ohlcv_path:
            plots.append({"href": ohlcv_path.relative_to(out_dir).as_posix(), "label": "OHLCV Chart"})

        flow_path = plot_binance_order_flow(df_pd, plots_dir, symbol)
        if flow_path:
            plots.append({"href": flow_path.relative_to(out_dir).as_posix(), "label": "Order Flow"})

        adv_path = plot_binance_advanced_features(df_pd, plots_dir, symbol)
        if adv_path:
            plots.append({"href": adv_path.relative_to(out_dir).as_posix(), "label": "Advanced Features"})

        ret_path = plot_binance_returns(df_pd, plots_dir, symbol)
        if ret_path:
            plots.append({"href": ret_path.relative_to(out_dir).as_posix(), "label": "Return Analysis"})

        symbol_plots[symbol] = plots
        print(f"  Generated {len(plots)} plots")

    # Build dashboard
    print("\nBuilding dashboard...")
    dashboard_path = build_binance_dashboard(out_dir, summary, symbol_plots)
    print(f"Dashboard saved to: {dashboard_path}")

    return dashboard_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on Binance public data")
    parser.add_argument("--data-dir", default="data/binance", help="Binance data directory")
    parser.add_argument("--out", default="reports/binance_eda", help="Output directory")
    parser.add_argument("--symbols", nargs="*", help="Symbols to analyze")
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1s", help="Aggregation interval")

    args = parser.parse_args()

    run_binance_eda(
        data_dir=args.data_dir,
        out_dir=args.out,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
