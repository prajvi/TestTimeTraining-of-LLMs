#!/usr/bin/env python3
"""
Summarize SimpleToM multiseed summaries into CSV + Markdown tables.

Example:
  python3 scripts/summarize_simpletom_multiseed.py \
    --glob "outputs/simpletom/multiseed/simpletom_*_summary_seed*.json" \
    --out-csv "outputs/simpletom/multiseed/simpletom_multiseed_summary.csv" \
    --out-md "outputs/simpletom/multiseed/simpletom_multiseed_summary.md"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List


FILE_RE = re.compile(r"simpletom_(?P<method>.+?)_summary_seed(?P<seed>\d+)\.json$")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="outputs/simpletom/multiseed/simpletom_*_summary_seed*.json",
        help="Glob for per-seed summary json files.",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/simpletom/multiseed/simpletom_multiseed_summary.csv",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/simpletom/multiseed/simpletom_multiseed_summary.md",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        name = os.path.basename(fp)
        m = FILE_RE.search(name)
        if not m:
            continue
        method = m.group("method")
        seed = int(m.group("seed"))
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        metrics = obj.get("metrics", {})
        meta = obj.get("meta", {})
        rows.append(
            {
                "method": method,
                "seed": seed,
                "avg_accuracy": _safe_float(metrics.get("average_accuracy")),
                "mental_state_accuracy": _safe_float(metrics.get("mental_state_accuracy")),
                "behavior_accuracy": _safe_float(metrics.get("behavior_accuracy")),
                "judgment_accuracy": _safe_float(metrics.get("judgment_accuracy")),
                "ms_minus_behavior": _safe_float(metrics.get("ms_minus_behavior")),
                "ms_minus_judgment": _safe_float(metrics.get("ms_minus_judgment")),
                "runtime_seconds": _safe_float(meta.get("runtime_seconds")),
            }
        )

    if not rows:
        raise SystemExit("Matched files, but none had expected filename format.")

    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    fields = [
        "method",
        "n_seeds",
        "avg_acc_mean",
        "avg_acc_std",
        "mental_state_mean",
        "behavior_mean",
        "judgment_mean",
        "ms_minus_behavior_mean",
        "ms_minus_judgment_mean",
        "runtime_sec_mean",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for method in sorted(by_method.keys()):
        rs = by_method[method]
        summary_rows.append(
            {
                "method": method,
                "n_seeds": len(rs),
                "avg_acc_mean": mean(r["avg_accuracy"] for r in rs),
                "avg_acc_std": pstdev(r["avg_accuracy"] for r in rs) if len(rs) > 1 else 0.0,
                "mental_state_mean": mean(r["mental_state_accuracy"] for r in rs),
                "behavior_mean": mean(r["behavior_accuracy"] for r in rs),
                "judgment_mean": mean(r["judgment_accuracy"] for r in rs),
                "ms_minus_behavior_mean": mean(r["ms_minus_behavior"] for r in rs),
                "ms_minus_judgment_mean": mean(r["ms_minus_judgment"] for r in rs),
                "runtime_sec_mean": mean(r["runtime_seconds"] for r in rs),
            }
        )

    out_csv_dir = os.path.dirname(args.out_csv)
    if out_csv_dir:
        os.makedirs(out_csv_dir, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    out_md_dir = os.path.dirname(args.out_md)
    if out_md_dir:
        os.makedirs(out_md_dir, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# SimpleToM Multiseed Summary\n\n")
        f.write("| Method | n | AvgAcc (mean+-std) | MS | Behavior | Judgment | MS-Behavior | MS-Judgment |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['method']} | {r['n_seeds']} | {_fmt(r['avg_acc_mean'])}+/-{_fmt(r['avg_acc_std'])} "
                f"| {_fmt(r['mental_state_mean'])} | {_fmt(r['behavior_mean'])} | {_fmt(r['judgment_mean'])} "
                f"| {_fmt(r['ms_minus_behavior_mean'])} | {_fmt(r['ms_minus_judgment_mean'])} |\n"
            )

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote MD:  {args.out_md}")


if __name__ == "__main__":
    main()
