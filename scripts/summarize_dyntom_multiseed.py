#!/usr/bin/env python3
"""
Summarize DynToM multiseed summaries into CSV + Markdown tables.

Example:
  python scripts/summarize_dyntom_multiseed.py \
    --glob "outputs/dyntom/multiseed/dyntom_*_summary_seed*.json" \
    --out-csv "outputs/dyntom/multiseed/dyntom_multiseed_summary.csv" \
    --out-md "outputs/dyntom/multiseed/dyntom_multiseed_summary.md"
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


FILE_RE = re.compile(r"dyntom_(?P<method>.+?)_summary_seed(?P<seed>\d+)\.json$")


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
        default="outputs/dyntom/multiseed/dyntom_*_summary_seed*.json",
        help="Glob for per-seed summary json files.",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/dyntom/multiseed/dyntom_multiseed_summary.csv",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/dyntom/multiseed/dyntom_multiseed_summary.md",
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
        rows.append(
            {
                "method": method,
                "seed": seed,
                "accuracy": _safe_float(metrics.get("accuracy")),
                "type_a_accuracy": _safe_float(metrics.get("type_a_accuracy")),
                "type_c_accuracy": _safe_float(metrics.get("type_c_accuracy")),
                "type_d_accuracy": _safe_float(metrics.get("type_d_accuracy")),
                "transformation_accuracy": _safe_float(metrics.get("transformation_accuracy")),
                "table_coverage": _safe_float(metrics.get("table_coverage")),
                "runtime": _safe_float(obj.get("runtime")),
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
        "accuracy_mean",
        "accuracy_std",
        "type_a_mean",
        "type_c_mean",
        "type_d_mean",
        "transform_mean",
        "table_cov_mean",
        "runtime_mean",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for method in sorted(by_method.keys()):
        rs = by_method[method]
        summary_rows.append(
            {
                "method": method,
                "n_seeds": len(rs),
                "accuracy_mean": mean(r["accuracy"] for r in rs),
                "accuracy_std": pstdev(r["accuracy"] for r in rs) if len(rs) > 1 else 0.0,
                "type_a_mean": mean(r["type_a_accuracy"] for r in rs),
                "type_c_mean": mean(r["type_c_accuracy"] for r in rs),
                "type_d_mean": mean(r["type_d_accuracy"] for r in rs),
                "transform_mean": mean(r["transformation_accuracy"] for r in rs),
                "table_cov_mean": mean(r["table_coverage"] for r in rs),
                "runtime_mean": mean(r["runtime"] for r in rs),
            }
        )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# DynToM Multiseed Summary\n\n")
        f.write("| Method | n | Acc (mean±std) | type_a | type_c | type_d | Transform | TableCov |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['method']} | {r['n_seeds']} | {_fmt(r['accuracy_mean'])}±{_fmt(r['accuracy_std'])} "
                f"| {_fmt(r['type_a_mean'])} | {_fmt(r['type_c_mean'])} | {_fmt(r['type_d_mean'])} "
                f"| {_fmt(r['transform_mean'])} | {_fmt(r['table_cov_mean'])} |\n"
            )

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote MD:  {args.out_md}")


if __name__ == "__main__":
    main()

