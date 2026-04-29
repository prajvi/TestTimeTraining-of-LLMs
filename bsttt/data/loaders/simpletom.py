"""
SimpleToM loader.

This milestone normalizes the Hugging Face `allenai/SimpleToM` dataset into a common schema:

{
  "id": ...,
  "story": ...,
  "question_type": one of ["mental_state", "behavior", "judgment"],
  "question": ...,
  "choices": [...],
  "answer": ...,
  "scenario_name": ...
}
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

from datasets import load_dataset


QuestionType = Literal["mental_state", "behavior", "judgment"]

_HF_SUBSET_TO_QTYPE: Dict[str, QuestionType] = {
    "mental-state-qa": "mental_state",
    "behavior-qa": "behavior",
    "judgment-qa": "judgment",
}


@dataclass(frozen=True)
class SimpleToMExample:
    id: str
    story: str
    question_type: QuestionType
    question: str
    choices: List[str]
    answer: str
    scenario_name: str

    @staticmethod
    def from_hf_record(record: Dict[str, Any], question_type: QuestionType) -> "SimpleToMExample":
        missing = [k for k in ["id", "story", "question", "scenario_name", "choices", "answerKey"] if k not in record]
        if missing:
            raise ValueError(f"SimpleToM record missing keys {missing}. Available keys: {list(record.keys())}")

        answer_key = record["answerKey"]
        if not isinstance(answer_key, str):
            answer_key = str(answer_key)

        choices_raw = record["choices"]
        choices, answer = coerce_simpletom_choices_and_answer(choices_raw, answer_key)
        return SimpleToMExample(
            id=str(record["id"]),
            story=str(record["story"]),
            question=str(record["question"]),
            question_type=question_type,
            choices=choices,
            answer=answer,
            scenario_name=str(record["scenario_name"]),
        )


def coerce_simpletom_choices_and_answer(choices_raw: Any, answer_key: str) -> Tuple[List[str], str]:
    """
    Normalize SimpleToM `choices` into:
      - choices_text: List[str]
      - answer_text: str

    SimpleToM uses a dict form like:
      choices_raw = {"text": ["Yes", "No"], "label": ["A","B"]}
    """
    # Common case: choices already as list[str]
    if isinstance(choices_raw, list):
        choices_text = choices_raw
        if not choices_text or not all(isinstance(c, str) for c in choices_text):
            raise ValueError(f"SimpleToM `choices` has unexpected list contents: {choices_raw}")
        return choices_text, _answer_key_to_choice(answer_key, choices_text)

    # Common HF dataset case: choices as {'text': [...], 'label': [...]}
    if isinstance(choices_raw, dict):
        if "text" in choices_raw:
            choices_text = choices_raw["text"]
        else:
            # Heuristic: take the first list-valued field as texts.
            list_fields = [v for v in choices_raw.values() if isinstance(v, list)]
            if not list_fields:
                raise ValueError(f"SimpleToM `choices` dict has no list-valued fields: keys={list(choices_raw.keys())}")
            choices_text = list_fields[0]

        if not isinstance(choices_text, list) or not choices_text or not all(isinstance(c, str) for c in choices_text):
            raise ValueError(f"SimpleToM `choices.text` has unexpected format: {type(choices_text).__name__}")

        # If we have labels, map answerKey to the corresponding text.
        if "label" in choices_raw:
            labels = choices_raw["label"]
            if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
                raise ValueError("SimpleToM `choices.label` must be list[str]")
            if len(labels) != len(choices_text):
                raise ValueError("SimpleToM `choices.text` and `choices.label` must align in length")

            key_upper = answer_key.strip().upper()
            labels_upper = [l.strip().upper() for l in labels]
            if key_upper in labels_upper:
                idx = labels_upper.index(key_upper)
                return choices_text, choices_text[idx]

            # Fallback: answer_key might still be A/B/C/D.
            try:
                answer_text = _answer_key_to_choice(answer_key, choices_text)
                return choices_text, answer_text
            except Exception:
                pass

        # Fallback: try direct match in text.
        if answer_key in choices_text:
            return choices_text, answer_key
        if answer_key.upper() in [c.upper() for c in choices_text]:
            upper_to_text = {c.upper(): c for c in choices_text}
            return choices_text, upper_to_text[answer_key.upper()]

        raise ValueError(f"Could not map answerKey='{answer_key}' into choices_text={choices_text}")

    raise ValueError(f"SimpleToM `choices` has unexpected type: {type(choices_raw).__name__}")


def _answer_key_to_choice(answer_key: str, choices: Sequence[str]) -> str:
    """
    Convert SimpleToM `answerKey` to the canonical answer string.

    SimpleToM uses A/B/C/D answerKey.
    """
    key = answer_key.strip()
    key_upper = key.upper()
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    if key_upper in letter_to_idx:
        idx = letter_to_idx[key_upper]
        if idx < 0 or idx >= len(choices):
            raise ValueError(f"answerKey={answer_key} maps to idx={idx} but len(choices)={len(choices)}")
        return str(choices[idx])

    # Sometimes datasets store the choice text directly.
    if key in choices:
        return key
    if key_upper in choices:
        return key_upper

    # Fallback: attempt to parse an integer index.
    try:
        idx = int(key)
        if 0 <= idx < len(choices):
            return str(choices[idx])
    except Exception:
        pass

    raise ValueError(f"Unrecognized answerKey='{answer_key}' for choices={choices}")


def iter_simpletom_records(
    *,
    subset: str,
    split: str = "test",
    streaming: bool = True,
    limit: Optional[int] = None,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """
    Yield raw HF records for a given SimpleToM subset.

    Notes:
      - In this environment, `SimpleToM` subset configs use `split='test'`.
      - `streaming=True` avoids downloading full tables.
    """
    if subset not in _HF_SUBSET_TO_QTYPE:
        raise ValueError(f"Unknown SimpleToM subset '{subset}'. Expected one of {sorted(_HF_SUBSET_TO_QTYPE)}")

    ds = load_dataset("allenai/SimpleToM", subset, split=split, streaming=streaming)
    rng = random.Random(seed)

    if limit is None:
        yield from ds
        return

    # For streaming datasets, iteration order is deterministic but may be fixed by server.
    # We optionally shuffle by taking a reservoir sample.
    reservoir: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        if i < limit:
            reservoir.append(ex)
        else:
            j = rng.randint(0, i)
            if j < limit:
                reservoir[j] = ex
        if i > (limit * 50):  # safety: don't loop forever for sanity checks
            break

    # If we couldn't reach limit due to early stop, return what we have.
    yield from reservoir


def load_simpletom_subset(
    *,
    subset: str,
    split: str = "test",
    streaming: bool = True,
    limit: Optional[int] = None,
) -> List[SimpleToMExample]:
    question_type = _HF_SUBSET_TO_QTYPE[subset]
    examples: List[SimpleToMExample] = []
    for record in iter_simpletom_records(subset=subset, split=split, streaming=streaming, limit=limit):
        examples.append(SimpleToMExample.from_hf_record(record, question_type))
    return examples


def validate_simpletom_examples(examples: Sequence[SimpleToMExample]) -> None:
    if not examples:
        raise ValueError("No examples to validate.")

    required = ["id", "story", "question_type", "question", "choices", "answer", "scenario_name"]
    for ex in examples:
        as_dict = asdict(ex)
        missing = [k for k in required if k not in as_dict]
        if missing:
            raise ValueError(f"Example missing {missing}: {as_dict.keys()}")
        if ex.question_type not in ("mental_state", "behavior", "judgment"):
            raise ValueError(f"Invalid question_type: {ex.question_type}")
        if not ex.choices or not isinstance(ex.choices, list) or not all(isinstance(c, str) for c in ex.choices):
            raise ValueError("choices must be non-empty list[str]")
        if ex.answer not in ex.choices:
            raise ValueError(f"answer='{ex.answer}' must be one of choices={ex.choices}")
        if not ex.story or not ex.question:
            raise ValueError("story/question must be non-empty strings")


def _default_cache_path(cache_dir: Path, cache_stem: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_stem}.jsonl"


def load_simpletom_processed(
    *,
    cache_dir: "Path | str" = "bsttt/data/cache",
    cache_stem: str = "simpletom_processed_small",
    cache_max_items_per_subset: int = 64,
    split: str = "test",
    streaming: bool = True,
    force_rebuild: bool = False,
    seed: int = 42,
) -> List[SimpleToMExample]:
    """
    Load and cache a small processed SimpleToM subset.

    We cache a small number of examples per subset to support fast, deterministic sanity checks.
    """
    cache_path = _default_cache_path(Path(cache_dir), cache_stem)
    if cache_path.exists() and not force_rebuild:
        out: List[SimpleToMExample] = []
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                out.append(
                    SimpleToMExample(
                        id=row["id"],
                        story=row["story"],
                        question_type=row["question_type"],
                        question=row["question"],
                        choices=list(row["choices"]),
                        answer=row["answer"],
                        scenario_name=row["scenario_name"],
                    )
                )
        validate_simpletom_examples(out)
        return out

    processed: List[SimpleToMExample] = []
    for subset in ["mental-state-qa", "behavior-qa", "judgment-qa"]:
        subset_examples = load_simpletom_subset(
            subset=subset,
            split=split,
            streaming=streaming,
            limit=cache_max_items_per_subset,
        )
        validate_simpletom_examples(subset_examples)
        processed.extend(subset_examples)

    # Deterministic shuffle for stable cached output ordering.
    rng = random.Random(seed)
    rng.shuffle(processed)

    with cache_path.open("w", encoding="utf-8") as f:
        for ex in processed:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    return processed


def split_examples_by_type(examples: Sequence[SimpleToMExample]) -> Dict[QuestionType, List[SimpleToMExample]]:
    out: Dict[QuestionType, List[SimpleToMExample]] = {"mental_state": [], "behavior": [], "judgment": []}
    for ex in examples:
        out[ex.question_type].append(ex)
    return out


def example_print(examples: Sequence[SimpleToMExample], n: int = 1) -> None:
    for ex in examples[:n]:
        print(json.dumps(asdict(ex), indent=2, ensure_ascii=False)[:2500])


def sanity_check_simpletom_loader(
    *,
    cache_max_items_per_subset: int = 8,
    force_rebuild: bool = False,
) -> None:
    processed = load_simpletom_processed(
        cache_max_items_per_subset=cache_max_items_per_subset,
        force_rebuild=force_rebuild,
    )
    validate_simpletom_examples(processed)
    by_type = split_examples_by_type(processed)
    for qt, exs in by_type.items():
        print(f"[sanity_check] {qt}: {len(exs)} examples")


if __name__ == "__main__":
    sanity_check_simpletom_loader(cache_max_items_per_subset=4, force_rebuild=False)
    processed = load_simpletom_processed(cache_max_items_per_subset=4)
    by_type = split_examples_by_type(processed)
    print("\n--- Examples (one per subset) ---")
    for qt in ["mental_state", "behavior", "judgment"]:
        print(f"\n[{qt}]")
        example_print(by_type[qt], n=1)

