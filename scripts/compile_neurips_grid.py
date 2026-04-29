#!/usr/bin/env python3
"""
Compile a NeurIPS-style benchmark grid table from:
  1) local experiment summaries
  2) optional paper-reported baselines jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x: float) -> str:
    if math.isnan(x):
        return "-"
    return f"{x:.4f}"


def _collect_external(outputs_dir: str) -> Dict[Tuple[str, str], List[float]]:
    out: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    patt = os.path.join(outputs_dir, "external_tom", "*_summary_seed*.json")
    for fp in sorted(glob.glob(patt)):
        name = os.path.basename(fp)
        if "_summary_seed" not in name:
            continue
        prefix = name.split("_summary_seed")[0]
        if "_" not in prefix:
            continue
        dataset = prefix.split("_", 1)[0]
        method = prefix.split("_", 1)[1]
        try:
            obj = json.load(open(fp, "r", encoding="utf-8"))
            acc = _safe_float(obj.get("metrics", {}).get("accuracy"))
            if not math.isnan(acc):
                out[(method, dataset)].append(acc)
        except Exception:
            continue
    return out


def _collect_reported(jsonl_path: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if not jsonl_path or not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            metric = str(obj.get("metric", "accuracy"))
            if metric != "accuracy":
                continue
            method = str(obj.get("method", "")).strip()
            dataset = str(obj.get("dataset", "")).strip().lower()
            value = _safe_float(obj.get("value"))
            if method and dataset and not math.isnan(value):
                out[(method, dataset)] = value
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--reported-jsonl", default="outputs/reported_baselines/reported_baselines.jsonl")
    p.add_argument("--out-csv", default="outputs/benchmark_grid.csv")
    p.add_argument("--out-md", default="outputs/benchmark_grid.md")
    args = p.parse_args()

    # Grid columns can be extended later.
    datasets = ["simpletom", "dyntom", "hitom", "fantom", "opentom"]
    methods_order = [
        "Frozen",
        "CoT",
        "Few-shot (4-shot)",
        "SimToM",
        "TimeToM",
        "SymbolicToM",
        "DEL-ToM",
        "BSTTT-NTL",
        "BSTTT-AR",
        "Scratchpad-Frozen",
        "Scratchpad-TTT",
        "Scratchpad-Oracle",
        "Hierarchical TTT",
    ]

    external = _collect_external(args.outputs_dir)
    reported = _collect_reported(args.reported_jsonl)

    # Alias local method names -> paper names.
    aliases = {
        "frozen": "Frozen",
        "cot": "CoT",
        "scratchpad_frozen": "Scratchpad-Frozen",
        "scratchpad_ttt": "Scratchpad-TTT",
        "scratchpad_oracle": "Scratchpad-Oracle",
        "hierarchical_ttt": "Hierarchical TTT",
        "simtom": "SimToM",
        "symbolictom": "SymbolicToM",
    }

    cell: Dict[Tuple[str, str], str] = {}

    # From local external_tom runs.
    for (method_raw, dataset), vals in external.items():
        paper_method = aliases.get(method_raw, method_raw)
        m = mean(vals)
        s = pstdev(vals) if len(vals) > 1 else 0.0
        cell[(paper_method, dataset)] = f"{m:.4f}±{s:.4f}"

    # From reported baselines.
    for (method, dataset), v in reported.items():
        if (method, dataset) not in cell:
            cell[(method, dataset)] = f"{v:.4f} (paper)"

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + datasets)
        for method in methods_order:
            row = [method]
            for ds in datasets:
                row.append(cell.get((method, ds), "-"))
            w.writerow(row)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Benchmark Grid\n\n")
        f.write("| Method | " + " | ".join(datasets) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(datasets)) + "|\n")
        for method in methods_order:
            vals = [cell.get((method, ds), "-") for ds in datasets]
            f.write("| " + method + " | " + " | ".join(vals) + " |\n")

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
