"""Generate profiling and index reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

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
