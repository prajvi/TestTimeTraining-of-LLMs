#!/usr/bin/env python3
"""
Inspect raw HF schema for external ToM datasets.
Useful when a dataset changes format or requires auth.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from datasets import load_dataset


def _preview_record(d: Dict[str, Any], max_len: int = 180) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in d.items():
        s = repr(v)
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        out[k] = s
    return out


def _resolve_split(dataset_id: str, requested_split: str, streaming: bool):
    def _load(split: str | None, use_streaming: bool):
        try:
            if split is None:
                return load_dataset(dataset_id, streaming=use_streaming), use_streaming
            return load_dataset(dataset_id, split=split, streaming=use_streaming), use_streaming
        except Exception as e:
            if not use_streaming:
                print(
                    f"[inspect_external] load_dataset failed with streaming={use_streaming} "
                    f"({type(e).__name__}: {e}). Retrying with streaming=True."
                )
                if split is None:
                    return load_dataset(dataset_id, streaming=True), True
                return load_dataset(dataset_id, split=split, streaming=True), True
            raise

    req = (requested_split or "auto").strip()
    if req.lower() != "auto":
        try:
            ds, _used = _load(req, streaming)
            return ds, req, []
        except ValueError:
            pass
    ds_dict, _used = _load(None, streaming)
    if not hasattr(ds_dict, "keys"):
        return ds_dict, req if req.lower() != "auto" else "default", []
    names = list(ds_dict.keys())
    pref = ["Long", "ExtraLong", "test", "validation", "train"] if dataset_id == "SeacowX/OpenToM" else ["test", "validation", "train", "Long", "ExtraLong"]
    chosen = next((s for s in pref if s in names), names[0])
    return ds_dict[chosen], chosen, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hitom", "opentom"], required=True)
    parser.add_argument("--split", default="auto")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    dataset_id = "Hi-ToM/Hi-ToM_Dataset" if args.dataset == "hitom" else "SeacowX/OpenToM"
    ds, used_split, available = _resolve_split(dataset_id, args.split, args.streaming)

    print(f"Dataset: {dataset_id} | split={used_split} | streaming={args.streaming}")
    if available:
        print(f"Available splits: {available}")
    for i, rec in enumerate(ds):
        if i >= args.n:
            break
        rec_d = dict(rec)
        print(f"\n--- Example {i} ---")
        print("keys:", sorted(rec_d.keys()))
        print(json.dumps(_preview_record(rec_d), indent=2))


if __name__ == "__main__":
    main()
