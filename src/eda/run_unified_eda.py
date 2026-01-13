"""CLI to run EDA on unified multi-source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .unified_plots import (
    generate_unified_plots,
    plot_symbol_timeseries,
    plot_symbol_features_grid,
)


def build_unified_dashboard(
    out_dir: Path,
    summary: Dict,
    plot_entries: List[Dict[str, str]],
    symbols: List[str],
    symbol_plots: Optional[Dict[str, List[Dict[str, str]]]] = None,
) -> None:
    """Build HTML dashboard for unified multi-source data.

    Args:
        out_dir: Output directory
        summary: Summary statistics
        plot_entries: List of {href, label} for plots
        symbols: List of symbols in dataset
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "<html><head><title>Unified Data Dashboard</title><style>",
        "body { font-family: Arial, sans-serif; margin: 20px; max-width: 1600px; background: #f5f5f5; }",
        "h1 { margin-top: 0; color: #333; }",
        "h2 { margin: 12px 0; color: #444; }",
        ".header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }",
        ".header h1 { color: white; margin: 0 0 10px 0; }",
        ".header p { margin: 0; opacity: 0.9; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }",
        ".card { border: 1px solid #ddd; padding: 20px; border-radius: 12px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".card h3 { margin: 0 0 8px 0; font-size: 0.85em; color: #666; text-transform: uppercase; }",
        ".card .value { font-size: 1.6em; font-weight: bold; color: #333; }",
        ".card .value.positive { color: #22c55e; }",
        ".card .value.negative { color: #ef4444; }",
        ".card .sub { font-size: 0.85em; color: #888; margin-top: 4px; }",
        ".muted { color: #666; font-size: 0.95em; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { text-align: left; padding: 12px; border-bottom: 1px solid #eee; }",
        "th { background: #f8f8f8; font-weight: 600; }",
        "tr:hover { background: #fafafa; }",
        ".section { background: white; padding: 24px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }",
        ".plot-link { padding: 12px 16px; background: linear-gradient(135deg, #f8f8f8 0%, #f0f0f0 100%); border-radius: 8px; display: block; text-decoration: none; color: #333; transition: all 0.2s; }",
        ".plot-link:hover { background: linear-gradient(135deg, #e8e8e8 0%, #e0e0e0 100%); transform: translateY(-2px); }",
        ".source-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin-left: 8px; }",
        ".source-binance { background: #f0f9ff; color: #0369a1; }",
        ".source-signals { background: #f0fdf4; color: #15803d; }",
        ".source-fills { background: #fff7ed; color: #c2410c; }",
        ".source-store { background: #faf5ff; color: #7e22ce; }",
        ".tabs { display: flex; gap: 8px; margin-bottom: 16px; }",
        ".tab { padding: 8px 16px; background: #e5e7eb; border-radius: 6px; cursor: pointer; }",
        ".tab.active { background: #667eea; color: white; }",
        "</style></head><body>",
    ]

    # Header
    lines.append("<div class='header'>")
    lines.append("<h1>Unified Multi-Source Data Dashboard</h1>")
    lines.append("<p>Combined analysis of order book, trading signals, fills, and Binance public data</p>")
    lines.append("</div>")

    # Data source summary
    if summary:
        lines.append("<div class='grid'>")

        # Overall metrics
        total_rows = summary.get("total_rows", 0)
        n_features = summary.get("n_features", 0)
        n_symbols = len(symbols)

        metrics = [
            ("Symbols", str(n_symbols), ""),
            ("Total Rows", f"{total_rows:,}", ""),
            ("Features", str(n_features), ""),
            ("Train Size", f"{summary.get('train_size', 0):,}", ""),
            ("Val Size", f"{summary.get('val_size', 0):,}", ""),
        ]

        # Add label distribution
        label_dist = summary.get("label_distribution", {})
        if label_dist:
            total = sum(label_dist.values())
            pos_rate = label_dist.get(1, 0) / total if total > 0 else 0
            metrics.append(("Positive Rate", f"{pos_rate:.1%}", ""))

        for label, val, sub in metrics:
            lines.append(f"<div class='card'><h3>{label}</h3><div class='value'>{val}</div>")
            if sub:
                lines.append(f"<div class='sub'>{sub}</div>")
            lines.append("</div>")

        lines.append("</div>")

        # Data sources breakdown
        data_sources = summary.get("data_sources", {})
        if data_sources:
            lines.append("<div class='section'><h2>Data Sources</h2>")
            lines.append("<table><tr><th>Source</th><th>Status</th><th>Rows</th><th>Features</th></tr>")
            for source, info in data_sources.items():
                status = "Available" if info.get("available") else "Not loaded"
                rows = info.get("rows", "—")
                features = info.get("features", "—")
                lines.append(f"<tr><td>{source}</td><td>{status}</td><td>{rows}</td><td>{features}</td></tr>")
            lines.append("</table></div>")

        # Symbol breakdown
        symbol_stats = summary.get("symbol_stats", {})
        if symbol_stats:
            lines.append("<div class='section'><h2>Symbol Performance</h2>")
            lines.append("<table><tr><th>Symbol</th><th>Rows</th><th>Features</th><th>Accuracy</th><th>AUC</th></tr>")
            for sym, stats in sorted(symbol_stats.items()):
                rows = stats.get("rows", "—")
                features = stats.get("features", "—")
                acc = f"{stats.get('accuracy', 0):.1%}" if stats.get("accuracy") else "—"
                auc = f"{stats.get('auc', 0):.1%}" if stats.get("auc") else "—"
                lines.append(f"<tr><td>{sym}</td><td>{rows}</td><td>{features}</td><td>{acc}</td><td>{auc}</td></tr>")
            lines.append("</table></div>")

        # Top features
        top_features = summary.get("top_features", [])
        if top_features:
            lines.append("<div class='section'><h2>Top Features (by importance)</h2>")
            lines.append("<table><tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Source</th></tr>")
            for i, (name, imp) in enumerate(top_features[:15], 1):
                # Determine source
                if name.startswith("binance_"):
                    source = "<span class='source-badge source-binance'>Binance</span>"
                elif any(x in name for x in ["signal", "strategy", "entry"]):
                    source = "<span class='source-badge source-signals'>Signals</span>"
                elif any(x in name for x in ["fill", "buy", "sell", "vwap"]):
                    source = "<span class='source-badge source-fills'>Fills</span>"
                elif any(x in name for x in ["bid", "ask", "spread", "book"]):
                    source = "<span class='source-badge source-store'>Order Book</span>"
                else:
                    source = ""
                lines.append(f"<tr><td>{i}</td><td>{name}</td><td>{imp:.4f}</td><td>{source}</td></tr>")
            lines.append("</table></div>")

    # Plots section
    if plot_entries:
        # Group by category
        plot_groups = {
            "Price & Timeline": [],
            "Signal Analysis": [],
            "Order Book": [],
            "Binance Features": [],
            "Model Performance": [],
        }

        for entry in plot_entries:
            label = entry["label"].lower()
            if any(x in label for x in ["price", "timeline", "ohlc"]):
                plot_groups["Price & Timeline"].append(entry)
            elif any(x in label for x in ["signal", "strategy"]):
                plot_groups["Signal Analysis"].append(entry)
            elif any(x in label for x in ["spread", "book", "bid", "ask"]):
                plot_groups["Order Book"].append(entry)
            elif any(x in label for x in ["binance", "toxicity", "imbalance"]):
                plot_groups["Binance Features"].append(entry)
            elif any(x in label for x in ["importance", "label", "correlation"]):
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

    # Per-symbol sections
    if symbol_plots:
        lines.append("<div class='section'><h2>Per-Symbol Timeseries</h2>")
        lines.append("<p class='muted'>Click on a symbol to view its detailed timeseries data</p>")

        # Symbol navigation tabs
        lines.append("<div class='tabs' style='flex-wrap: wrap; margin-bottom: 20px;'>")
        for sym in sorted(symbol_plots.keys()):
            lines.append(f"<a href='#{sym}' class='tab'>{sym}</a>")
        lines.append("</div>")

        # Individual symbol sections
        for sym in sorted(symbol_plots.keys()):
            sym_entries = symbol_plots.get(sym, [])
            if sym_entries:
                stats = summary.get("symbol_stats", {}).get(sym, {})
                rows = stats.get("rows", "—")
                features = stats.get("features", "—")
                acc = f"{stats.get('accuracy', 0):.1%}" if stats.get("accuracy") else "—"
                auc = f"{stats.get('auc', 0):.1%}" if stats.get("auc") else "—"

                lines.append(f"<div id='{sym}' style='margin-bottom: 24px; padding: 16px; background: #fafafa; border-radius: 8px;'>")
                lines.append(f"<h3 style='margin: 0 0 12px 0;'>{sym}</h3>")
                lines.append(f"<p class='muted' style='margin: 0 0 12px 0;'>Rows: {rows} | Features: {features} | Accuracy: {acc} | AUC: {auc}</p>")
                lines.append("<div class='plot-grid'>")
                for item in sym_entries:
                    lines.append(f'<a href="{item["href"]}" class="plot-link">{item["label"]}</a>')
                lines.append("</div></div>")

        lines.append("</div>")

    lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def run_unified_eda(
    unified_dir: str = "data/unified",
    out_dir: str = "reports/unified_eda",
    symbols: Optional[List[str]] = None,
    run_model: bool = True,
) -> Path:
    """Run EDA on unified multi-source data.

    Args:
        unified_dir: Directory with unified datasets
        out_dir: Output directory for reports
        symbols: List of symbols to analyze (None = all)
        run_model: Whether to train and evaluate a model

    Returns:
        Path to the generated dashboard
    """
    import polars as pl
    from myproj.ml.models import train_price_move_model, evaluate_classifier

    unified_dir = Path(unified_dir)
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading unified datasets from {unified_dir}...")

    # Find available datasets
    parquet_files = list(unified_dir.glob("unified_*_full.parquet"))
    available_symbols = [f.stem.replace("unified_", "").replace("_full", "") for f in parquet_files]

    if symbols:
        available_symbols = [s for s in available_symbols if s in symbols]

    if not available_symbols:
        raise FileNotFoundError(f"No datasets found in {unified_dir}")

    print(f"Found symbols: {available_symbols}")

    # Load and combine datasets
    all_dfs = []
    symbol_stats = {}

    for symbol in available_symbols:
        parquet_path = unified_dir / f"unified_{symbol}_full.parquet"
        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
            all_dfs.append(df)
            symbol_stats[symbol] = {
                "rows": len(df),
                "features": len([c for c in df.columns if c not in ["timestamp", "symbol", "label"]]),
            }
            print(f"  {symbol}: {len(df):,} rows")

    if not all_dfs:
        raise ValueError("No data loaded")

    # Combine all dataframes (use diagonal concat to handle different columns)
    combined_df = pl.concat(all_dfs, how="diagonal")
    print(f"\nCombined dataset: {len(combined_df):,} rows")

    # Summary
    summary = {
        "total_rows": len(combined_df),
        "n_features": len([c for c in combined_df.columns if c not in ["timestamp", "symbol", "label"]]),
        "train_size": 0,
        "val_size": 0,
        "symbol_stats": symbol_stats,
        "data_sources": {
            "Order Book (.store)": {
                "available": any(c.startswith("bid_") or c.startswith("ask_") for c in combined_df.columns),
                "features": len([c for c in combined_df.columns if c.startswith("bid_") or c.startswith("ask_") or "spread" in c]),
            },
            "Signals (logs)": {
                "available": "last_signal" in combined_df.columns,
                "features": len([c for c in combined_df.columns if "signal" in c.lower()]),
            },
            "Fills (logs)": {
                "available": "fill_volume" in combined_df.columns or "n_fills" in combined_df.columns,
                "features": len([c for c in combined_df.columns if "fill" in c.lower() or "buy_volume" in c or "sell_volume" in c]),
            },
            "Binance Public": {
                "available": any(c.startswith("binance_") for c in combined_df.columns),
                "features": len([c for c in combined_df.columns if c.startswith("binance_")]),
            },
        },
    }

    # Train model if requested
    feature_importance = []
    if run_model:
        print("\nTraining baseline model on combined data...")

        # Load numpy arrays from first symbol with good data
        for symbol in available_symbols:
            arrays_path = unified_dir / f"unified_{symbol}_arrays.npz"
            if arrays_path.exists():
                data = np.load(arrays_path, allow_pickle=True)
                X_train = data["X_train"]
                X_val = data["X_val"]
                y_train = data["y_train"]
                y_val = data["y_val"]
                feature_cols = data["feature_cols"].tolist()

                # Check for valid data
                if len(np.unique(y_train)) < 2:
                    continue

                summary["train_size"] += len(X_train)
                summary["val_size"] += len(X_val)

                try:
                    model = train_price_move_model(X_train, y_train)
                    metrics = evaluate_classifier(model, X_val, y_val)
                    symbol_stats[symbol]["accuracy"] = metrics["accuracy"]
                    symbol_stats[symbol]["auc"] = metrics.get("auc", 0)

                    # Extract feature importance
                    if hasattr(model, "named_steps"):
                        lr = model.named_steps.get("logisticregression")
                        if lr is not None and hasattr(lr, "coef_"):
                            coef = np.abs(lr.coef_.ravel())
                            if len(coef) == len(feature_cols):
                                feature_importance = sorted(
                                    zip(feature_cols, coef.tolist()),
                                    key=lambda x: -x[1]
                                )

                    print(f"  {symbol}: Accuracy={metrics['accuracy']:.2%}, AUC={metrics.get('auc', 0):.2%}")
                except Exception as e:
                    print(f"  {symbol}: Training failed - {e}")

    summary["top_features"] = feature_importance[:20]
    summary["label_distribution"] = {0: 0, 1: 0}
    if "label" in combined_df.columns:
        label_counts = combined_df.group_by("label").len()
        for row in label_counts.iter_rows():
            summary["label_distribution"][int(row[0])] = row[1]

    # Generate plots
    print("\nGenerating plots...")
    combined_pd = combined_df.to_pandas()

    plot_paths = generate_unified_plots(
        combined_pd,
        plots_dir,
        prefix="unified",
        feature_importance=feature_importance,
    )

    # Create plot entries
    plot_entries = []
    for p in plot_paths:
        plot_entries.append({
            "href": p.relative_to(out_dir).as_posix(),
            "label": p.stem.replace("unified_", "").replace("_", " ").title(),
        })

    # Generate per-symbol plots
    print("\nGenerating per-symbol timeseries plots...")
    symbol_plots: Dict[str, List[Dict[str, str]]] = {}

    for symbol in available_symbols:
        parquet_path = unified_dir / f"unified_{symbol}_full.parquet"
        if not parquet_path.exists():
            continue

        df_sym = pl.read_parquet(parquet_path).to_pandas()
        sym_paths = []

        # Timeseries plot
        ts_path = plot_symbol_timeseries(df_sym, plots_dir, symbol)
        if ts_path:
            sym_paths.append({
                "href": ts_path.relative_to(out_dir).as_posix(),
                "label": "Timeseries Dashboard",
            })

        # Feature grid plot
        feat_path = plot_symbol_features_grid(df_sym, plots_dir, symbol)
        if feat_path:
            sym_paths.append({
                "href": feat_path.relative_to(out_dir).as_posix(),
                "label": "Feature Analysis",
            })

        if sym_paths:
            symbol_plots[symbol] = sym_paths
            print(f"  {symbol}: {len(sym_paths)} plots generated")

    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Build dashboard
    print("Building dashboard...")
    build_unified_dashboard(out_dir, summary, plot_entries, available_symbols, symbol_plots)

    dashboard_path = out_dir / "index.html"
    print(f"\nDashboard saved to: {dashboard_path}")

    return dashboard_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on unified multi-source data.")
    parser.add_argument("--unified-dir", default="data/unified", help="Directory with unified datasets")
    parser.add_argument("--out", default="reports/unified_eda", help="Output directory")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to analyze")
    parser.add_argument("--no-model", action="store_true", help="Skip model training")

    args = parser.parse_args()

    run_unified_eda(
        unified_dir=args.unified_dir,
        out_dir=args.out,
        symbols=args.symbols,
        run_model=not args.no_model,
    )


if __name__ == "__main__":
    main()
