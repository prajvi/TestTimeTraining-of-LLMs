#!/usr/bin/env python3
"""
Machine-validation baselines for external ToM MCQA datasets.

Baselines:
  1) majority_index: predicts most common answer index from train split
  2) lexical_overlap: picks option with max token overlap with context+question
  3) tfidf_logreg: predicts answer index from context+question text
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from bsttt.data.loaders.tom_external import ExternalToMExample, load_external_tom_processed


def _tok(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _build_xy(rows: Sequence[ExternalToMExample]) -> Tuple[List[str], List[int]]:
    x = [f"{r.story}\n\nQuestion: {r.question}" for r in rows]
    y = [int(r.answer_index) for r in rows]
    return x, y


def _accuracy(pred: Sequence[int], gold: Sequence[int]) -> float:
    if not gold:
        return 0.0
    return float(np.mean([int(p == g) for p, g in zip(pred, gold)]))


def _majority_index(train_y: Sequence[int], test_n: int) -> List[int]:
    if not train_y:
        return [0] * test_n
    c = Counter(train_y)
    maj = c.most_common(1)[0][0]
    return [maj] * test_n


def _lexical_overlap_predict(rows: Sequence[ExternalToMExample]) -> List[int]:
    out: List[int] = []
    for r in rows:
        base = _tok(f"{r.story} {r.question}")
        best_i = 0
        best_score = -1
        for i, opt in enumerate(r.choices):
            score = len(base & _tok(opt))
            if score > best_score:
                best_score = score
                best_i = i
        out.append(best_i)
    return out


def _tfidf_logreg_predict(train_x: Sequence[str], train_y: Sequence[int], test_x: Sequence[str]) -> List[int]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    xtr = vec.fit_transform(train_x)
    xte = vec.transform(test_x)
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(xtr, train_y)
    return list(clf.predict(xte))


def _train_test_split(rows: Sequence[ExternalToMExample], *, train_ratio: float, seed: int) -> Tuple[List[ExternalToMExample], List[ExternalToMExample]]:
    idx = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = max(1, min(len(rows) - 1, int(math.floor(train_ratio * len(rows)))))
    tr = [rows[i] for i in idx[:n_train]]
    te = [rows[i] for i in idx[n_train:]]
    return tr, te


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hitom", "opentom"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/machine_validation")
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    rows = load_external_tom_processed(
        dataset_name=args.dataset,
        split=args.split,
        streaming=args.streaming,
        limit=args.limit,
        force_rebuild=args.force_rebuild,
    )
    train_rows, test_rows = _train_test_split(rows, train_ratio=args.train_ratio, seed=args.seed)
    _, train_y = _build_xy(train_rows)
    test_x, test_y = _build_xy(test_rows)

    pred_majority = _majority_index(train_y, len(test_rows))
    pred_overlap = _lexical_overlap_predict(test_rows)
    train_x, train_y = _build_xy(train_rows)
    pred_tfidf = _tfidf_logreg_predict(train_x, train_y, test_x)

    results: Dict[str, float] = {
        "majority_index_accuracy": _accuracy(pred_majority, test_y),
        "lexical_overlap_accuracy": _accuracy(pred_overlap, test_y),
        "tfidf_logreg_accuracy": _accuracy(pred_tfidf, test_y),
        "num_total": float(len(rows)),
        "num_train": float(len(train_rows)),
        "num_test": float(len(test_rows)),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{args.dataset}_machine_validation_seed{args.seed}.json"
    out_fp.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "limit": args.limit,
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    print(f"Saved: {out_fp}")


if __name__ == "__main__":
    main()
