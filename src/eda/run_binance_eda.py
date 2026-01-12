"""CLI to run EDA on Binance order flow data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


from .binance_plots import (
    generate_binance_plots,
    plot_feature_importance,
)


def build_binance_dashboard(
    out_dir: Path,
    summary: Dict,
    plot_entries: List[Dict[str, str]],
) -> None:
    """Build HTML dashboard for Binance data analysis."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "<html><head><title>Binance Data Dashboard</title><style>",
        "body { font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; background: #f5f5f5; }",
        "h1 { margin-top: 0; color: #333; }",
        "h2 { margin: 12px 0; color: #444; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; }",
        ".card { border: 1px solid #ddd; padding: 16px; border-radius: 8px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".card h3 { margin: 0 0 8px 0; font-size: 0.9em; color: #666; }",
        ".card .value { font-size: 1.5em; font-weight: bold; color: #333; }",
        ".card .value.positive { color: #22c55e; }",
        ".card .value.negative { color: #ef4444; }",
        ".muted { color: #666; font-size: 0.95em; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eee; }",
        "th { background: #f8f8f8; }",
        "a { text-decoration: none; color: #2563eb; }",
        "a:hover { text-decoration: underline; }",
        ".section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 8px; }",
        ".plot-link { padding: 8px 12px; background: #f0f0f0; border-radius: 4px; display: block; }",
        "</style></head><body>",
        "<h1>Binance Order Flow Dashboard</h1>",
        "<p class='muted'>Analysis of trade data with advanced microstructure features.</p>",
    ]

    # Dataset metrics
    if summary:
        lines.append("<div class='grid'>")
        metrics = [
            ("Symbol", summary.get("symbol", "—")),
            ("Total Rows", f"{summary.get('total_rows', 0):,}"),
            ("Features", summary.get("n_features", "—")),
            ("Train Size", f"{summary.get('train_size', 0):,}"),
            ("Val Size", f"{summary.get('val_size', 0):,}"),
            ("Interval", summary.get("interval", "—")),
        ]
        for label, val in metrics:
            lines.append(f"<div class='card'><h3>{label}</h3><div class='value'>{val}</div></div>")
        lines.append("</div>")

        # Model metrics
        model_metrics = summary.get("model_metrics", {})
        if model_metrics:
            lines.append("<div class='section'><h2>Model Performance</h2>")
            lines.append("<div class='grid'>")
            for name, val in model_metrics.items():
                css_class = "positive" if name == "accuracy" and val > 0.4 else ""
                if isinstance(val, float):
                    lines.append(f"<div class='card'><h3>{name}</h3><div class='value {css_class}'>{val:.4f}</div></div>")
                else:
                    lines.append(f"<div class='card'><h3>{name}</h3><div class='value'>{val}</div></div>")
            lines.append("</div></div>")

        # Label distribution
        label_dist = summary.get("label_distribution", {})
        if label_dist:
            lines.append("<div class='section'><h2>Label Distribution</h2>")
            lines.append("<table><tr><th>Class</th><th>Count</th><th>Percentage</th></tr>")
            total = sum(label_dist.values())
            label_names = {-1: "Down", 0: "Neutral", 1: "Up"}
            for cls, count in sorted(label_dist.items()):
                pct = count / total * 100 if total > 0 else 0
                name = label_names.get(int(cls), str(cls))
                lines.append(f"<tr><td>{name}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>")
            lines.append("</table></div>")

        # Top features
        top_features = summary.get("top_features", [])
        if top_features:
            lines.append("<div class='section'><h2>Top 10 Features (by importance)</h2>")
            lines.append("<table><tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>")
            for i, (name, imp) in enumerate(top_features[:10], 1):
                lines.append(f"<tr><td>{i}</td><td>{name}</td><td>{imp:.4f}</td></tr>")
            lines.append("</table></div>")

    # Plots
    if plot_entries:
        # Group plots by type
        plot_groups = {
            "Price & Volume": [],
            "Order Flow": [],
            "Microstructure": [],
            "Labels & Correlations": [],
        }

        for entry in plot_entries:
            label = entry["label"].lower()
            if any(x in label for x in ["ohlcv", "candlestick"]):
                plot_groups["Price & Volume"].append(entry)
            elif any(x in label for x in ["imbalance", "toxicity", "kyle", "momentum", "autocorr"]):
                plot_groups["Microstructure"].append(entry)
            elif any(x in label for x in ["distribution", "correlation", "importance"]):
                plot_groups["Labels & Correlations"].append(entry)
            else:
                plot_groups["Order Flow"].append(entry)

        for group_name, entries in plot_groups.items():
            if entries:
                lines.append(f"<div class='section'><h2>{group_name}</h2>")
                lines.append("<div class='plot-grid'>")
                for item in entries:
                    lines.append(f'<a href="{item["href"]}" class="plot-link">{item["label"]}</a>')
                lines.append("</div></div>")

    lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def run_binance_eda(
    symbol: str = "BTCUSDT",
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-07",
    interval: str = "1s",
    data_dir: str = "data/binance",
    out_dir: str = "reports/binance_eda",
    label_horizon: int = 30,
    deadzone_bps: float = 2.0,
    include_advanced_features: bool = True,
    run_model: bool = True,
) -> Path:
    """Run EDA on Binance data and generate dashboard.

    Args:
        symbol: Trading pair symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Aggregation interval
        data_dir: Directory with downloaded Binance data
        out_dir: Output directory for reports
        label_horizon: Prediction horizon for labels
        deadzone_bps: Deadzone for direction classification
        include_advanced_features: Whether to compute advanced features
        run_model: Whether to train and evaluate baseline model

    Returns:
        Path to the generated dashboard
    """
    from myproj.binance_data import (
        build_binance_ml_dataset,
        get_feature_importance,
    )
    from myproj.ml.models import train_price_move_model, evaluate_classifier

    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building dataset for {symbol} ({start_date} to {end_date})...")

    # Build dataset
    dataset = build_binance_ml_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        data_dir=data_dir,
        label_horizon=label_horizon,
        deadzone_bps=deadzone_bps,
        include_advanced_features=include_advanced_features,
    )

    print(f"Dataset: {len(dataset.df):,} rows, {len(dataset.feature_cols)} features")

    # Summary dict
    summary = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "total_rows": len(dataset.df),
        "n_features": len(dataset.feature_cols),
        "train_size": len(dataset.X_train),
        "val_size": len(dataset.X_val),
        "label_col": dataset.label_col,
    }

    # Label distribution
    import numpy as np
    unique, counts = np.unique(dataset.y_train, return_counts=True)
    summary["label_distribution"] = {int(k): int(v) for k, v in zip(unique, counts)}

    # Train model and get metrics
    if run_model:
        print("Training baseline model...")
        model = train_price_move_model(dataset.X_train, dataset.y_train)
        metrics = evaluate_classifier(model, dataset.X_val, dataset.y_val)
        summary["model_metrics"] = metrics

        # Feature importance
        importance = get_feature_importance(model, dataset.feature_cols, top_k=20)
        summary["top_features"] = importance

        # Plot feature importance
        imp_path = plot_feature_importance(importance, plots_dir, "binance")
        print(f"  Feature importance plot: {imp_path}")

    # Generate plots
    print("Generating plots...")
    windows = [5, 15, 60] if include_advanced_features else [5, 15]
    plot_paths = generate_binance_plots(dataset.df, "binance", plots_dir, windows=windows)

    # Create plot entries for dashboard
    plot_entries = []
    for p in plot_paths:
        plot_entries.append({
            "href": p.relative_to(out_dir).as_posix(),
            "label": p.stem.replace("binance_", "").replace("_", " ").title(),
        })

    # Add feature importance plot
    if run_model:
        plot_entries.append({
            "href": "plots/binance_feature_importance.html",
            "label": "Feature Importance",
        })

    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Build dashboard
    print("Building dashboard...")
    build_binance_dashboard(out_dir, summary, plot_entries)

    dashboard_path = out_dir / "index.html"
    print(f"Dashboard saved to: {dashboard_path}")

    return dashboard_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on Binance order flow data.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-01-07", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1s", help="Aggregation interval")
    parser.add_argument("--data-dir", default="data/binance", help="Binance data directory")
    parser.add_argument("--out", default="reports/binance_eda", help="Output directory")
    parser.add_argument("--horizon", type=int, default=30, help="Label horizon in bars")
    parser.add_argument("--deadzone", type=float, default=2.0, help="Deadzone in bps")
    parser.add_argument("--no-advanced", action="store_true", help="Skip advanced features")
    parser.add_argument("--no-model", action="store_true", help="Skip model training")

    args = parser.parse_args()

    run_binance_eda(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        data_dir=args.data_dir,
        out_dir=args.out,
        label_horizon=args.horizon,
        deadzone_bps=args.deadzone,
        include_advanced_features=not args.no_advanced,
        run_model=not args.no_model,
    )


if __name__ == "__main__":
    main()
