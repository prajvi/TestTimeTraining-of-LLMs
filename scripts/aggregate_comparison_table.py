#!/usr/bin/env python3
"""
Aggregate SimpleToM evaluation summaries into a comparison table (Milestone 4).

Usage:
  python scripts/aggregate_comparison_table.py outputs/simpletom_*_summary_*.json
  python scripts/aggregate_comparison_table.py outputs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics_row(data: Dict[str, Any]) -> Dict[str, Any]:
    meta = data.get("meta", {})
    metrics = data.get("metrics", {})
    method = meta.get("method", "unknown")
    model = meta.get("model", "unknown")

    row: Dict[str, Any] = {
        "method": method,
        "model": model,
        "runtime_seconds": meta.get("runtime_seconds"),
    }
    row.update(metrics)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SimpleToM summaries into comparison table")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Summary JSON files or directory containing *_summary_*.json",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--md", type=str, default=None, help="Output Markdown path")
    args = parser.parse_args()

    files: List[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*_summary_*.json")))
        elif path.exists():
            files.append(path)

    if not files:
        print("No summary files found.", file=sys.stderr)
        return

    rows: List[Dict[str, Any]] = []
    for f in files:
        try:
            data = load_summary(f)
            rows.append(extract_metrics_row(data))
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}", file=sys.stderr)

    if not rows:
        print("No valid summaries loaded.", file=sys.stderr)
        return

    out_dir = Path(args.output).parent if args.output else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(rows)

    # Reorder columns: method, model, accuracies, then rest
    acc_cols = [c for c in df.columns if "accuracy" in c and "ci95" not in c and "lower" not in c and "upper" not in c]
    other_cols = [c for c in df.columns if c not in acc_cols]
    col_order = ["method", "model"] + sorted(acc_cols) + [c for c in other_cols if c not in ("method", "model")]
    df = df[[c for c in col_order if c in df.columns]]

    csv_path = args.output or str(out_dir / "simpletom_comparison.csv")
    md_path = args.md or str(Path(csv_path).with_suffix(".md"))

    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Markdown table
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v) if v is not None else ""

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    md_content = "\n".join([header, sep] + body_lines) + "\n"

    Path(md_path).write_text(md_content, encoding="utf-8")
    print(f"Wrote {md_path}")

    print("\nComparison table:")
    print(md_content)


if __name__ == "__main__":
    main()
