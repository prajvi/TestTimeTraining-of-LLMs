"""
Prompt-only baselines for SimpleToM evaluation.

Supported prompt strategies:
  1. vanilla: direct QA prompt, no extra reasoning (same scoring as frozen)
  2. cot: chain-of-thought ("Let's think step by step") with generation + parsing
  3. ms_reminder: inject predicted mental-state into behavior/judgment prompts
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from bsttt.data.loaders.simpletom import QuestionType, SimpleToMExample

SimpleToMTask = Literal["mental_state", "behavior", "judgment"]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_vanilla_prompt(story: str, question: str, choices: Sequence[str]) -> str:
    """Direct QA prompt (identical to frozen baseline)."""
    return _build_mc_prompt(story, question, choices, cot=False, ms_hint=None)


def build_cot_prompt(story: str, question: str, choices: Sequence[str]) -> str:
    """Chain-of-thought prompt with reasoning instruction."""
    return _build_mc_prompt(story, question, choices, cot=True, ms_hint=None)


def build_ms_reminder_prompt(
    story: str,
    question: str,
    choices: Sequence[str],
    predicted_ms: str,
) -> str:
    """Behavior/judgment prompt with injected mental-state reminder."""
    ms_hint = _format_ms_hint(predicted_ms)
    return _build_mc_prompt(story, question, choices, cot=False, ms_hint=ms_hint)


def _build_mc_prompt(
    story: str,
    question: str,
    choices: Sequence[str],
    cot: bool = False,
    ms_hint: Optional[str] = None,
) -> str:
    choice_lines = []
    for i, c in enumerate(choices):
        letter = LETTERS[i] if i < len(LETTERS) else f"C{i}"
        choice_lines.append(f"{letter}. {c}")

    parts = [f"Story:\n{story}\n\n", f"Question:\n{question}\n\n"]
    if ms_hint:
        parts.insert(1, f"Mental-state hint: {ms_hint}\n\n")
    if cot:
        parts.append(
            "Let's think step by step to reason about this question, then give your final answer.\n\n"
        )
    parts.append("Choices:\n" + "\n".join(choice_lines) + "\n\n")
    parts.append("Answer:" if not cot else "Answer (state the letter or choice):")

    return "".join(parts)


def _format_ms_hint(predicted_ms: str) -> str:
    """Convert mental-state answer to a hint string."""
    txt = str(predicted_ms).strip().lower()
    if txt in ("yes", "aware", "true", "1"):
        return "The character is predicted to be aware."
    if txt in ("no", "unaware", "false", "0"):
        return "The character is predicted to be unaware."
    return f"The character's predicted mental state: {predicted_ms}"


def parse_answer_from_generation(
    text: str,
    choices: Sequence[str],
) -> Tuple[int, str]:
    """
    Parse model output to extract predicted choice index and text.

    Tries (in order):
      1. Letter (A/B/C/D) at start of line or after "Answer"
      2. Exact match of choice text (case-insensitive)
      3. Substring match of choice text
      4. Fallback to first choice

    Returns:
      (pred_index, pred_choice_text)
    """
    if not choices:
        return 0, ""

    text_clean = text.strip()
    if not text_clean:
        return 0, choices[0]

    # 1. Look for letter (A/B/C) at beginning of last line or after "Answer"
    last_line = text_clean.split("\n")[-1].strip()
    letter_match = re.search(r"\b([A-D])\b", last_line, re.IGNORECASE)
    if letter_match:
        letter = letter_match.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return idx, choices[idx]

    # Also check anywhere in text for "Answer: A" or "answer is B"
    anywhere = re.search(r"(?:answer|choice)\s*[:\s]+\s*([A-D])\b", text_clean, re.IGNORECASE)
    if anywhere:
        letter = anywhere.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return idx, choices[idx]

    # 2. Exact match (case-insensitive)
    choices_lower = [c.lower() for c in choices]
    text_lower = text_clean.lower()
    for i, c in enumerate(choices_lower):
        if c in text_lower or text_lower.endswith(c):
            return i, choices[i]

    # 3. Substring match
    for i, c in enumerate(choices):
        if c.lower() in text_lower:
            return i, choices[i]

    # 4. Fallback
    return 0, choices[0]


def get_mental_state_example_for_scenario(
    by_scenario_and_type: Dict[str, Dict[QuestionType, List[SimpleToMExample]]],
    scenario_name: str,
) -> Optional[SimpleToMExample]:
    """Get any mental-state example for a scenario (for ms_reminder)."""
    bucket = by_scenario_and_type.get(scenario_name, {})
    ms_list = bucket.get("mental_state", [])
    return ms_list[0] if ms_list else None
