"""Generate profiling and a richer dashboard index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ydata_profiling import ProfileReport

from .config import ensure_dir


def profiling_report(df, out_path: Path, title: str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = ProfileReport(df, title=title, minimal=True)
    profile.to_file(out_path)
    return out_path


def write_summary(summary: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def write_index(out_dir: Path, entries: List[Dict[str, str]]) -> None:
    out_dir = ensure_dir(out_dir)
    lines = ["<html><body><h2>EDA Outputs</h2><ul>"]
    for item in entries:
        lines.append(f"<li><a href=\"{item['href']}\">{item['label']}</a></li>")
    lines.append("</ul></body></html>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def build_dashboard(
    out_dir: Path,
    summary_path: Path,
    plot_entries: List[Dict[str, str]],
    profile_entries: List[Dict[str, str]],
    raw_plot_entries: List[Dict[str, str]] | None = None,
) -> None:
    """Render a single HTML dashboard linking all artifacts."""
    out_dir = ensure_dir(out_dir)
    summary: Optional[Dict] = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def safe_get(dct, *keys, default=None):
        cur = dct
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    lines: List[str] = [
        "<html><head><title>EDA Dashboard</title><style>",
        "body { font-family: Arial, sans-serif; margin: 20px; max-width: 1100px; }",
        "h1 { margin-top: 0; }",
        "h2 { margin: 12px 0; }",
        "ul { line-height: 1.6; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }",
        ".card { border: 1px solid #ddd; padding: 12px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }",
        ".muted { color: #666; font-size: 0.95em; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { text-align: left; padding: 6px 4px; border-bottom: 1px solid #eee; }",
        "a { text-decoration: none; color: #005ea5; }",
        "</style></head><body>",
        "<h1>EDA Dashboard</h1>",
        "<p class='muted'>Central view of data quality, targets, and links to plots/profiles.</p>",
    ]

    if summary:
        # Top-level metrics
        rows_train = safe_get(summary, "quality", "train", "rows", default="–")
        rows_val = safe_get(summary, "quality", "val", "rows", default="–")
        rows_test = safe_get(summary, "quality", "test", "rows", default="–")
        pct_dt = safe_get(summary, "quality", "train", "pct_dt_expected", default="–")
        lines.append("<div class='grid'>")
        for label, val in [
            ("Train rows", rows_train),
            ("Val rows", rows_val),
            ("Test rows", rows_test),
            ("% dt==expected (train)", f"{pct_dt:.2f}%" if isinstance(pct_dt, (int, float)) else pct_dt),
        ]:
            lines.append(f"<div class='card'><h3>{label}</h3><div>{val}</div></div>")
        lines.append("</div>")

        # Target class balance and baselines (direction)
        dir_stats = safe_get(summary, "targets", "train", "dir", default={})
        if dir_stats:
            lines.append("<div class='card'><h2>Direction Targets (train)</h2><table>")
            lines.append("<tr><th>Target</th><th>Unique</th><th>Majority %</th></tr>")
            for tgt, info in dir_stats.items():
                lines.append(
                    f"<tr><td>{tgt}</td><td>{info.get('unique','–')}</td>"
                    f"<td>{info.get('pct_majority','–'):.2f}%</td></tr>"
                )
            lines.append("</table></div>")

        # Baselines
        baselines = safe_get(summary, "targets", "train", "baselines", default={})
        if baselines:
            lines.append("<div class='card'><h2>Baseline Metrics (train)</h2><table>")
            lines.append("<tr><th>Metric</th><th>Value</th></tr>")
            for k, v in baselines.items():
                if isinstance(v, (int, float)):
                    lines.append(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>")
                else:
                    lines.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
            lines.append("</table></div>")

    def render_section(title: str, entries: List[Dict[str, str]]) -> None:
        if not entries:
            return
        lines.append('<div class="card">')
        lines.append(f"<h2>{title}</h2><ul>")
        for item in entries:
            lines.append(f'<li><a href="{item["href"]}">{item["label"]}</a></li>')
        lines.append("</ul></div>")

    render_section("EDA Plots", plot_entries)
    render_section("Profiling Reports", profile_entries)
    render_section("Raw Data Plots", raw_plot_entries or [])

    lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")
