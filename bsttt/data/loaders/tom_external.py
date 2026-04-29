"""
Loaders for external ToM MCQA datasets (Hi-ToM, OpenToM).

These loaders normalize heterogeneous HF schemas into one consistent format:

{
  "id": str,
  "dataset": "hitom" | "opentom",
  "story": str,
  "question": str,
  "choices": List[str],
  "answer": str,
  "answer_index": int,
  "question_type": str,
  "scenario_name": str
}
"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


@dataclass(frozen=True)
class ExternalToMExample:
    id: str
    dataset: str
    story: str
    question: str
    choices: List[str]
    answer: str
    answer_index: int
    question_type: str
    scenario_name: str


_STORY_KEYS = [
    "story",
    "context",
    "passage",
    "article",
    "narrative",
    "scenario",
    "situation",
    "text",
    "dialogue",
]

_QUESTION_KEYS = [
    "question",
    "query",
    "prompt",
    "q",
]

_CHOICES_KEYS = [
    "choices",
    "options",
    "option_list",
    "candidate_answers",
    "candidates",
    "answers",
]

_ANSWER_KEYS = [
    "answer",
    "answer_text",
    "gold",
    "gold_answer",
    "correct_answer",
    "label",
    "answerKey",
]

_ANSWER_INDEX_KEYS = [
    "answer_idx",
    "answer_index",
    "label_idx",
    "correct_index",
    "correct_option_idx",
]

_QTYPE_KEYS = [
    "question_type",
    "type",
    "category",
    "level",
    "task",
]

_SCENARIO_KEYS = [
    "scenario_name",
    "scenario_id",
    "story_id",
    "context_id",
    "dialogue_id",
    "episode_id",
]

_ID_KEYS = [
    "id",
    "example_id",
    "uid",
    "qid",
]


def _pick_first(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _story_group_id(dataset_name: str, story: str) -> str:
    canon = re.sub(r"\s+", " ", (story or "").strip())
    digest = hashlib.md5(canon.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_name}_story_{digest}"


def _normalize_choice_item(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    if isinstance(x, dict):
        for k in ["text", "option", "content", "answer"]:
            if k in x and x[k] is not None:
                s = str(x[k]).strip()
                if s:
                    return s
        # Fallback to first non-empty scalar field.
        for v in x.values():
            if isinstance(v, (str, int, float)):
                s = str(v).strip()
                if s:
                    return s
        return None
    if isinstance(x, (int, float)):
        return str(x)
    return None


def _parse_choices(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: List[str] = []
        for it in raw:
            s = _normalize_choice_item(it)
            if s:
                out.append(s)
        return out
    if isinstance(raw, dict):
        # Common HF style: {"text": [...], "label": [...]}
        if "text" in raw:
            return _parse_choices(raw["text"])
        # Otherwise pick first list-valued field.
        for v in raw.values():
            if isinstance(v, list):
                out = _parse_choices(v)
                if out:
                    return out
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        # Labeled options like "A. foo, B. bar, C. baz"
        labeled = list(re.finditer(r"([A-Za-z])[\)\.\:]\s*", s))
        if len(labeled) >= 2:
            out: List[str] = []
            for i, m in enumerate(labeled):
                start = m.end()
                end = labeled[i + 1].start() if i + 1 < len(labeled) else len(s)
                seg = s[start:end].strip(" ,;\t\n")
                if seg:
                    out.append(seg)
            if len(out) >= 2:
                return out
        # Try robust separators.
        for sep in ["||", ";;", "\n", ";", "|", ","]:
            parts = [p.strip() for p in s.split(sep)]
            if len([p for p in parts if p]) >= 2:
                return [p for p in parts if p]
        # Try A) / B) pattern.
        parts = re.split(r"\s+[A-Za-z][\)\.\:]\s+", " " + s)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return parts
        return [s]
    return []


def _resolve_answer(choices: Sequence[str], answer_raw: Any, answer_index_raw: Any) -> Tuple[Optional[str], Optional[int]]:
    if not choices:
        return None, None

    if isinstance(answer_index_raw, int) and 0 <= answer_index_raw < len(choices):
        return choices[answer_index_raw], answer_index_raw
    if isinstance(answer_index_raw, str) and answer_index_raw.strip().isdigit():
        idx = int(answer_index_raw.strip())
        if 0 <= idx < len(choices):
            return choices[idx], idx

    if answer_raw is None:
        return None, None

    ans = str(answer_raw).strip()
    if not ans:
        return None, None

    # A/B/C/D style.
    if len(ans) == 1 and ans.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        idx = ord(ans.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx], idx

    # Numeric index style.
    if ans.isdigit():
        idx = int(ans)
        if 0 <= idx < len(choices):
            return choices[idx], idx

    # Exact and case-insensitive text matches.
    if ans in choices:
        return ans, choices.index(ans)
    lower_to_idx = {c.lower(): i for i, c in enumerate(choices)}
    if ans.lower() in lower_to_idx:
        i = lower_to_idx[ans.lower()]
        return choices[i], i

    # Prefix "A. choice".
    m = re.match(r"^\s*([A-Za-z])[\)\.\:]\s*(.+)$", ans)
    if m:
        letter = m.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx], idx
        ans2 = m.group(2).strip()
        if ans2.lower() in lower_to_idx:
            i = lower_to_idx[ans2.lower()]
            return choices[i], i

    return None, None


def _extract_question_bundle(question_raw: Any) -> Tuple[str, str, Any, Any]:
    if not isinstance(question_raw, dict):
        return _to_text(question_raw), "", None, None

    q_text = _to_text(_pick_first(question_raw, ["question", "query", "prompt", "q", "text"]))
    q_type = _to_text(_pick_first(question_raw, ["type", "question_type", "category", "task"]))
    q_answer = _pick_first(question_raw, ["answer", "answer_text", "gold", "label", "answerKey"])
    q_choices = _pick_first(question_raw, ["choices", "options", "candidate_answers", "candidates", "answers"])
    return q_text, q_type, q_answer, q_choices


def _normalize_record(record: Dict[str, Any], *, dataset_name: str, row_idx: int) -> Optional[ExternalToMExample]:
    story = _to_text(_pick_first(record, _STORY_KEYS))
    question_raw = _pick_first(record, _QUESTION_KEYS)
    question, qtype_from_question, q_answer_raw, q_choices_raw = _extract_question_bundle(question_raw)
    choices_raw = q_choices_raw if q_choices_raw is not None else _pick_first(record, _CHOICES_KEYS)
    answer_raw = _pick_first(record, _ANSWER_KEYS)
    if answer_raw is None:
        answer_raw = q_answer_raw
    answer_index_raw = _pick_first(record, _ANSWER_INDEX_KEYS)

    choices = _parse_choices(choices_raw)
    answer, answer_index = _resolve_answer(choices, answer_raw, answer_index_raw)
    if answer is None:
        fallback = _to_text(answer_raw)
        if fallback:
            answer = fallback
            if choices:
                lower_to_idx = {c.lower(): i for i, c in enumerate(choices)}
                if fallback in choices:
                    answer_index = choices.index(fallback)
                elif fallback.lower() in lower_to_idx:
                    answer_index = lower_to_idx[fallback.lower()]

    # Some datasets (e.g., OpenToM) provide class labels without explicit options.
    # Keep a single-label placeholder; we'll expand label spaces after loading.
    if answer and len(choices) < 2:
        choices = [answer]
        answer_index = 0

    if not story or not question or answer is None or answer_index is None:
        return None

    qtype = _to_text(_pick_first(record, _QTYPE_KEYS)) or qtype_from_question or "unknown"
    scenario_name = _to_text(_pick_first(record, _SCENARIO_KEYS))
    if not scenario_name:
        # Crucial for TTT support construction: group rows by shared story if dataset
        # does not expose an explicit scenario/story id.
        scenario_name = _story_group_id(dataset_name, story)
    ex_id = _to_text(_pick_first(record, _ID_KEYS)) or f"{dataset_name}_{row_idx}"

    return ExternalToMExample(
        id=ex_id,
        dataset=dataset_name,
        story=story,
        question=question,
        choices=choices,
        answer=answer,
        answer_index=answer_index,
        question_type=qtype,
        scenario_name=scenario_name,
    )


def _expand_choice_sets(rows: Sequence[ExternalToMExample]) -> Tuple[List[ExternalToMExample], int, int]:
    if not rows:
        return [], 0, 0

    by_qtype: Dict[str, set[str]] = {}
    all_labels: set[str] = set()
    for r in rows:
        lbl = r.answer.strip()
        if not lbl:
            continue
        all_labels.add(lbl)
        by_qtype.setdefault(r.question_type, set()).add(lbl)

    global_labels = sorted(all_labels)
    out: List[ExternalToMExample] = []
    expanded = 0
    dropped = 0

    for r in rows:
        choices = list(r.choices)
        answer = r.answer.strip()
        if not answer:
            dropped += 1
            continue

        if len(choices) < 2:
            q_labels = sorted(by_qtype.get(r.question_type, set()))
            label_space = q_labels if len(q_labels) >= 2 else global_labels
            if len(label_space) < 2:
                dropped += 1
                continue
            choices = list(label_space)
            if answer not in choices:
                choices.append(answer)
            expanded += 1

        if answer in choices:
            answer_index = choices.index(answer)
        else:
            lower_to_idx = {c.lower(): i for i, c in enumerate(choices)}
            if answer.lower() in lower_to_idx:
                answer_index = lower_to_idx[answer.lower()]
                answer = choices[answer_index]
            else:
                choices = choices + [answer]
                answer_index = len(choices) - 1

        if len(choices) < 2:
            dropped += 1
            continue

        out.append(
            ExternalToMExample(
                id=r.id,
                dataset=r.dataset,
                story=r.story,
                question=r.question,
                choices=choices,
                answer=answer,
                answer_index=answer_index,
                question_type=r.question_type,
                scenario_name=r.scenario_name,
            )
        )

    return out, expanded, dropped


def _resolve_dataset_split(dataset_id: str, requested_split: str, *, streaming: bool) -> Tuple[Any, str]:
    def _load(split: Optional[str], use_streaming: bool) -> Tuple[Any, bool]:
        try:
            if split is None:
                return load_dataset(dataset_id, streaming=use_streaming), use_streaming
            return load_dataset(dataset_id, split=split, streaming=use_streaming), use_streaming
        except Exception as e:
            if not use_streaming:
                print(
                    f"[tom_external] load_dataset failed with streaming={use_streaming} "
                    f"({type(e).__name__}: {e}). Retrying with streaming=True."
                )
                if split is None:
                    return load_dataset(dataset_id, streaming=True), True
                return load_dataset(dataset_id, split=split, streaming=True), True
            raise

    req = (requested_split or "auto").strip()
    if req.lower() != "auto":
        try:
            ds, used_streaming = _load(req, streaming)
            if used_streaming != streaming:
                print(
                    f"[tom_external] Using streaming=True fallback for dataset={dataset_id}, split={req}."
                )
            return ds, req
        except ValueError:
            pass

    ds_dict, used_streaming = _load(None, streaming)
    if used_streaming != streaming:
        print(f"[tom_external] Using streaming=True fallback for dataset={dataset_id} while resolving split.")
    if not hasattr(ds_dict, "keys"):
        # Unusual single-split dataset object.
        return ds_dict, req if req.lower() != "auto" else "default"

    split_names = list(ds_dict.keys())
    if not split_names:
        raise ValueError(f"No splits found for dataset '{dataset_id}'.")

    # Dataset-aware preference order.
    if dataset_id == "SeacowX/OpenToM":
        pref = ["Long", "ExtraLong", "test", "validation", "train"]
    else:
        pref = ["test", "validation", "train", "Long", "ExtraLong"]

    chosen = None
    for s in pref:
        if s in split_names:
            chosen = s
            break
    if chosen is None:
        chosen = split_names[0]

    print(
        f"[tom_external] Requested split='{requested_split}' unavailable for {dataset_id}. "
        f"Using split='{chosen}' from available={split_names}"
    )
    return ds_dict[chosen], chosen


def _iterate_records(dataset_id: str, *, split: str, streaming: bool, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    ds, used_split = _resolve_dataset_split(dataset_id, split, streaming=streaming)
    emitted = 0
    try:
        for rec in ds:
            if limit is not None and emitted >= limit:
                break
            emitted += 1
            yield dict(rec)
        return
    except Exception as e:
        print(
            f"[tom_external] iterable dataset iteration failed for {dataset_id} "
            f"(split={used_split}): {type(e).__name__}: {e}"
        )
        print("[tom_external] Falling back to direct JSON file loading from HF Hub.")

    remaining = None if limit is None else max(0, limit - emitted)
    for rec in _iterate_records_hub_json(dataset_id, split=used_split, limit=remaining):
        yield rec


def _flatten_records_from_obj(obj: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(obj, dict):
        # Direct record dict.
        if any(k in obj for k in (_QUESTION_KEYS + _STORY_KEYS)):
            yield obj
            return
        # Wrapped lists in common keys.
        for key in ["data", "examples", "records", "items", "questions", "rows"]:
            v = obj.get(key)
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        yield it
                return
        # Fallback: recurse dict values.
        for v in obj.values():
            if isinstance(v, (list, dict)):
                yield from _flatten_records_from_obj(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _flatten_records_from_obj(it)


def _candidate_json_files(dataset_id: str, split: str) -> List[str]:
    files = list_repo_files(dataset_id, repo_type="dataset")
    json_files = [f for f in files if f.lower().endswith((".json", ".jsonl"))]
    if not json_files:
        return []

    s = (split or "auto").lower()
    # Prefer split-specific files first when possible.
    scored: List[Tuple[int, str]] = []
    for f in json_files:
        fl = f.lower()
        score = 0
        if "readme" in fl:
            score += 50
        if s in ("long", "extralong") and s in fl:
            score -= 20
        if dataset_id == "SeacowX/OpenToM":
            if s in ("auto", "long") and ("opentom.json" in fl or "/opentom.json" in fl):
                score -= 30
            if s == "extralong" and "opentom_long" in fl:
                score -= 30
        scored.append((score, f))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [f for _, f in scored]


def _iterate_records_hub_json(dataset_id: str, *, split: str, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    candidates = _candidate_json_files(dataset_id, split)
    if not candidates:
        return

    yielded = 0
    seen_ids: set[str] = set()
    for fp in candidates:
        try:
            local_fp = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=fp)
        except Exception as e:
            print(f"[tom_external] Could not download {fp}: {type(e).__name__}: {e}")
            continue

        try:
            if local_fp.lower().endswith(".jsonl"):
                with open(local_fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(rec, dict):
                            continue
                        rid = str(rec.get("id", ""))
                        if rid and rid in seen_ids:
                            continue
                        if rid:
                            seen_ids.add(rid)
                        yield rec
                        yielded += 1
                        if limit is not None and yielded >= limit:
                            return
            else:
                with open(local_fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                for rec in _flatten_records_from_obj(obj):
                    rid = str(rec.get("id", ""))
                    if rid and rid in seen_ids:
                        continue
                    if rid:
                        seen_ids.add(rid)
                    yield rec
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
        except Exception as e:
            print(f"[tom_external] Failed parsing {fp}: {type(e).__name__}: {e}")
            continue


def _cache_path(cache_dir: Path, stem: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{stem}.jsonl"


def _load_cached(fp: Path) -> List[ExternalToMExample]:
    out: List[ExternalToMExample] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out.append(ExternalToMExample(**obj))
    return out


def _write_cache(fp: Path, rows: Sequence[ExternalToMExample]) -> None:
    with fp.open("w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def load_external_tom_processed(
    *,
    dataset_name: str,
    split: str = "auto",
    streaming: bool = True,
    limit: Optional[int] = None,
    cache_dir: "Path | str" = "bsttt/data/cache",
    force_rebuild: bool = False,
) -> List[ExternalToMExample]:
    """
    Load and normalize external ToM dataset into a common MCQA schema.

    Supported `dataset_name`:
      - "hitom"   -> HF "Hi-ToM/Hi-ToM_Dataset"
      - "opentom" -> HF "SeacowX/OpenToM"
    """
    name = dataset_name.strip().lower()
    if name == "hitom":
        dataset_id = "Hi-ToM/Hi-ToM_Dataset"
    elif name == "opentom":
        dataset_id = "SeacowX/OpenToM"
    else:
        raise ValueError(f"Unknown dataset_name='{dataset_name}'. Expected one of ['hitom', 'opentom'].")

    cache_dir_p = Path(cache_dir)
    lim = "all" if limit is None else str(limit)
    cache_fp = _cache_path(cache_dir_p, f"{name}_{split}_{lim}_processed")
    if cache_fp.exists() and not force_rebuild:
        cached = _load_cached(cache_fp)
        if cached:
            return cached

    rows_raw: List[ExternalToMExample] = []
    skipped = 0
    for i, rec in enumerate(_iterate_records(dataset_id, split=split, streaming=streaming, limit=limit)):
        ex = _normalize_record(rec, dataset_name=name, row_idx=i)
        if ex is None:
            skipped += 1
            continue
        rows_raw.append(ex)

    rows, expanded, dropped_after_expand = _expand_choice_sets(rows_raw)
    skipped += dropped_after_expand

    if not rows:
        raise ValueError(
            f"No parseable examples for dataset='{dataset_id}' split='{split}'. "
            "Try another split or inspect schema via scripts/inspect_external_tom_schema.py."
        )

    _write_cache(cache_fp, rows)
    print(
        f"Loaded {len(rows)} normalized examples for {name} (split={split}, limit={limit}). "
        f"Expanded {expanded} rows with inferred label spaces. "
        f"Skipped {skipped} unparseable rows. Cache: {cache_fp}"
    )
    return rows
