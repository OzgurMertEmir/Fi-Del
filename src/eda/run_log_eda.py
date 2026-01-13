"""CLI to run EDA on log-based signal and fill data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .log_plots import generate_log_plots, plot_model_metrics


def build_log_dashboard(
    out_dir: Path,
    summary: dict,
    plot_entries: list[dict[str, str]],
) -> None:
    """Build HTML dashboard for log data analysis."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "<html><head><title>Signal & Fill Dashboard</title><style>",
        "body { font-family: Arial, sans-serif; margin: 20px; max-width: 1400px; background: #f5f5f5; }",
        "h1 { margin-top: 0; color: #333; }",
        "h2 { margin: 12px 0; color: #444; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }",
        ".card { border: 1px solid #ddd; padding: 16px; border-radius: 8px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".card h3 { margin: 0 0 8px 0; font-size: 0.9em; color: #666; }",
        ".card .value { font-size: 1.4em; font-weight: bold; color: #333; }",
        ".card .value.positive { color: #22c55e; }",
        ".card .value.negative { color: #ef4444; }",
        ".muted { color: #666; font-size: 0.95em; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eee; }",
        "th { background: #f8f8f8; }",
        "a { text-decoration: none; color: #2563eb; }",
        "a:hover { text-decoration: underline; }",
        ".section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px; }",
        ".plot-link { padding: 8px 12px; background: #f0f0f0; border-radius: 4px; display: block; }",
        "</style></head><body>",
        "<h1>Signal & Fill Analysis Dashboard</h1>",
        "<p class='muted'>Analysis of trading signals and execution fills from log data.</p>",
    ]

    # Dataset metrics
    if summary:
        lines.append("<div class='grid'>")
        metrics = [
            ("Total Signals", f"{summary.get('total_signals', 0):,}"),
            ("Total Fills", f"{summary.get('total_fills', 0):,}"),
            ("Unique Orders", summary.get("unique_orders", "—")),
            ("Symbols", summary.get("n_symbols", "—")),
            ("Fill Rate", f"{summary.get('fill_rate', 0):.1%}"),
            ("Time Range", summary.get("time_range", "—")),
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
                css_class = "positive" if name == "accuracy" and val > 0.55 else ""
                if isinstance(val, float):
                    lines.append(f"<div class='card'><h3>{name}</h3><div class='value {css_class}'>{val:.4f}</div></div>")
                else:
                    lines.append(f"<div class='card'><h3>{name}</h3><div class='value'>{val}</div></div>")
            lines.append("</div></div>")

        # Signal distribution
        signal_dist = summary.get("signal_distribution", {})
        if signal_dist:
            lines.append("<div class='section'><h2>Signal Distribution</h2>")
            lines.append("<table><tr><th>Signal Value</th><th>Count</th><th>Percentage</th></tr>")
            total = sum(signal_dist.values())
            for sig_val in sorted(signal_dist.keys(), key=int):
                count = signal_dist[sig_val]
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"<tr><td>{sig_val}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>")
            lines.append("</table></div>")

        # Symbol breakdown
        symbol_stats = summary.get("symbol_stats", {})
        if symbol_stats:
            lines.append("<div class='section'><h2>Symbol Breakdown</h2>")
            lines.append("<table><tr><th>Symbol</th><th>Signals</th><th>Fills</th><th>Fill Rate</th></tr>")
            for symbol, stats in sorted(symbol_stats.items(), key=lambda x: -x[1].get("signals", 0)):
                signals = stats.get("signals", 0)
                fills = stats.get("fills", 0)
                rate = fills / signals if signals > 0 else 0
                lines.append(f"<tr><td>{symbol}</td><td>{signals:,}</td><td>{fills:,}</td><td>{rate:.1%}</td></tr>")
            lines.append("</table></div>")

        # Top features
        top_features = summary.get("top_features", [])
        if top_features:
            lines.append("<div class='section'><h2>Top Features (by importance)</h2>")
            lines.append("<table><tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>")
            for i, (name, imp) in enumerate(top_features[:10], 1):
                lines.append(f"<tr><td>{i}</td><td>{name}</td><td>{imp:.4f}</td></tr>")
            lines.append("</table></div>")

    # Plots
    if plot_entries:
        plot_groups = {
            "Signal Analysis": [],
            "Fill Analysis": [],
            "Price Analysis": [],
            "Model Performance": [],
        }

        for entry in plot_entries:
            label = entry["label"].lower()
            if any(x in label for x in ["signal", "strategy"]):
                plot_groups["Signal Analysis"].append(entry)
            elif any(x in label for x in ["fill", "volume", "side"]):
                plot_groups["Fill Analysis"].append(entry)
            elif any(x in label for x in ["price", "mid"]):
                plot_groups["Price Analysis"].append(entry)
            elif any(x in label for x in ["feature", "importance"]):
                plot_groups["Model Performance"].append(entry)
            else:
                plot_groups["Signal Analysis"].append(entry)

        for group_name, entries in plot_groups.items():
            if entries:
                lines.append(f"<div class='section'><h2>{group_name}</h2>")
                lines.append("<div class='plot-grid'>")
                for item in entries:
                    lines.append(f'<a href="{item["href"]}" class="plot-link">{item["label"]}</a>')
                lines.append("</div></div>")

    lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def run_log_eda(
    data_dir: str = "data/processed",
    out_dir: str = "reports/log_eda",
    fill_window_sec: float = 30.0,
    run_model: bool = True,
) -> Path:
    """Run EDA on log data and generate dashboard.

    Args:
        data_dir: Directory with processed signals and fills
        out_dir: Output directory for reports
        fill_window_sec: Time window for signal-fill matching
        run_model: Whether to train and evaluate baseline model

    Returns:
        Path to the generated dashboard
    """
    from myproj.data import (
        load_signals,
        load_fills,
        build_log_ml_dataset,
        get_dataset_summary,
        DatasetConfig,
    )
    from myproj.ml.models import train_price_move_model, evaluate_classifier

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}...")

    # Load raw data
    signals_df = load_signals(data_dir)
    fills_df = load_fills(data_dir)

    print(f"  Signals: {len(signals_df):,} rows")
    print(f"  Fills: {len(fills_df):,} rows")

    # Build ML dataset
    config = DatasetConfig(fill_window_sec=fill_window_sec)
    dataset = build_log_ml_dataset(data_dir=data_dir, config=config)
    ml_summary = get_dataset_summary(dataset)

    print(f"  ML dataset: {ml_summary['total_rows']:,} rows, {ml_summary['n_features']} features")

    # Compute summary statistics
    signals_pd = signals_df.to_pandas()
    _ = fills_df.to_pandas()

    # Time range
    time_min = signals_pd["timestamp"].min()
    time_max = signals_pd["timestamp"].max()

    summary = {
        "total_signals": len(signals_df),
        "total_fills": len(fills_df),
        "unique_orders": fills_df["order_id"].n_unique(),
        "n_symbols": signals_df["symbol"].n_unique(),
        "fill_rate": ml_summary["label_distribution"].get(1, 0) / ml_summary["total_rows"],
        "time_range": f"{time_min} to {time_max}",
        "signal_distribution": {
            str(k): int(v) for k, v in signals_df.group_by("signal_value").agg(
                __import__("polars").len()
            ).sort("signal_value").iter_rows()
        },
    }

    # Symbol stats
    symbol_signals = dict(signals_df.group_by("symbol").agg(__import__("polars").len()).iter_rows())
    symbol_fills = dict(fills_df.group_by("symbol").agg(__import__("polars").len()).iter_rows())
    summary["symbol_stats"] = {
        sym: {"signals": symbol_signals.get(sym, 0), "fills": symbol_fills.get(sym, 0)}
        for sym in set(symbol_signals.keys()) | set(symbol_fills.keys())
    }

    # Train model
    feature_importance = []
    if run_model:
        print("Training baseline model...")
        model = train_price_move_model(dataset.X_train, dataset.y_train)
        metrics = evaluate_classifier(model, dataset.X_val, dataset.y_val)
        summary["model_metrics"] = metrics

        # Extract feature importance
        if hasattr(model, "named_steps"):
            lr = model.named_steps.get("logisticregression")
            if lr is not None and hasattr(lr, "coef_"):
                coef = lr.coef_.ravel()
                feature_importance = sorted(
                    zip(dataset.feature_cols, np.abs(coef).tolist()),
                    key=lambda x: -x[1]
                )
                summary["top_features"] = feature_importance

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")

    # Generate plots
    print("Generating plots...")
    plot_paths = generate_log_plots(
        signals_df,
        fills_df,
        dataset.df,
        plots_dir,
        prefix="logs",
    )

    # Add feature importance plot
    if feature_importance:
        imp_paths = plot_model_metrics(
            summary.get("model_metrics", {}),
            feature_importance,
            plots_dir,
            "logs",
        )
        plot_paths.extend(imp_paths)

    # Create plot entries
    plot_entries = []
    for p in plot_paths:
        plot_entries.append({
            "href": p.relative_to(out_dir).as_posix(),
            "label": p.stem.replace("logs_", "").replace("_", " ").title(),
        })

    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Build dashboard
    print("Building dashboard...")
    build_log_dashboard(out_dir, summary, plot_entries)

    dashboard_path = out_dir / "index.html"
    print(f"Dashboard saved to: {dashboard_path}")

    return dashboard_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on log-based signal and fill data.")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--out", default="reports/log_eda", help="Output directory")
    parser.add_argument("--window", type=float, default=30.0, help="Fill window in seconds")
    parser.add_argument("--no-model", action="store_true", help="Skip model training")

    args = parser.parse_args()

    run_log_eda(
        data_dir=args.data_dir,
        out_dir=args.out,
        fill_window_sec=args.window,
        run_model=not args.no_model,
    )


if __name__ == "__main__":
    main()
