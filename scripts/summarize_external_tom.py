#!/usr/bin/env python3
"""
Summarize external ToM runs into CSV + Markdown.
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
from typing import Any, Dict, List, Set


FILE_RE = re.compile(r"(?P<dataset>hitom|opentom)_(?P<method>.+?)_summary_seed(?P<seed>\d+)\.json$")


def _fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.4f}"


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="outputs/external_tom/*_summary_seed*.json")
    p.add_argument("--out-csv", default="outputs/external_tom/external_tom_summary.csv")
    p.add_argument("--out-md", default="outputs/external_tom/external_tom_summary.md")
    args = p.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob}")

    rows = []
    metric_keys_seen: Set[str] = set()
    for fp in files:
        m = FILE_RE.search(os.path.basename(fp))
        if not m:
            continue
        obj = json.load(open(fp, "r", encoding="utf-8"))
        metric = obj.get("metrics", {})
        for k, v in metric.items():
            if isinstance(v, (int, float)):
                metric_keys_seen.add(k)
        rows.append(
            {
                "dataset": m.group("dataset"),
                "method": m.group("method"),
                "seed": int(m.group("seed")),
                "metrics": metric,
                "runtime": _safe_float(obj.get("runtime")),
            }
        )

    by: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["method"])].append(r)

    metric_keys = sorted(metric_keys_seen)
    summary_rows = []
    for (dataset, method), rs in sorted(by.items()):
        out: Dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "n_seeds": len(rs),
            "runtime_mean": mean(r["runtime"] for r in rs),
            "runtime_std": pstdev([r["runtime"] for r in rs]) if len(rs) > 1 else 0.0,
        }
        for mk in metric_keys:
            vals = [_safe_float(r["metrics"].get(mk)) for r in rs]
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                continue
            out[f"{mk}_mean"] = mean(vals)
            out[f"{mk}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append(out)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    ordered_cols = ["dataset", "method", "n_seeds"]
    for mk in metric_keys:
        ordered_cols.append(f"{mk}_mean")
        ordered_cols.append(f"{mk}_std")
    ordered_cols += ["runtime_mean", "runtime_std"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered_cols)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    extra_acc_keys = [k for k in metric_keys if k.endswith("_accuracy") and k != "accuracy"]
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# External ToM Summary\n\n")
        header = ["Dataset", "Method", "n", "Accuracy (mean+-std)"]
        for k in extra_acc_keys:
            header.append(f"{k} (mean+-std)")
        header.extend(["NumExamples", "Runtime(s)"])
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in summary_rows:
            cells = [
                str(r["dataset"]),
                str(r["method"]),
                str(r["n_seeds"]),
                f"{_fmt(_safe_float(r.get('accuracy_mean')))}+/-{_fmt(_safe_float(r.get('accuracy_std')))}",
            ]
            for k in extra_acc_keys:
                cells.append(
                    f"{_fmt(_safe_float(r.get(f'{k}_mean')))}+/-{_fmt(_safe_float(r.get(f'{k}_std')))}"
                )
            cells.append(_fmt(_safe_float(r.get("num_examples_mean"))))
            cells.append(_fmt(_safe_float(r.get("runtime_mean"))))
            f.write("| " + " | ".join(cells) + " |\n")

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
