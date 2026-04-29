"""
Metrics for SimpleToM evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Sequence, Tuple


SimpleToMTask = Literal["mental_state", "behavior", "judgment"]


def accuracy_from_bools(correct: Sequence[bool]) -> float:
    if len(correct) == 0:
        raise ValueError("accuracy_from_bools: empty correct list")
    return sum(1 for x in correct if x) / len(correct)


def compute_simpletom_metrics(
    *,
    correct_by_task: Dict[SimpleToMTask, Sequence[bool]],
) -> Dict[str, float]:
    """
    Compute accuracy and gap metrics for SimpleToM.
    """
    for t in ["mental_state", "behavior", "judgment"]:
        if t not in correct_by_task:
            raise ValueError(f"Missing task '{t}' in correct_by_task")

    acc_ms = accuracy_from_bools(correct_by_task["mental_state"])
    acc_b = accuracy_from_bools(correct_by_task["behavior"])
    acc_j = accuracy_from_bools(correct_by_task["judgment"])

    acc_avg = (acc_ms + acc_b + acc_j) / 3.0

    return {
        "mental_state_accuracy": acc_ms,
        "behavior_accuracy": acc_b,
        "judgment_accuracy": acc_j,
        "average_accuracy": acc_avg,
        "ms_minus_behavior": acc_ms - acc_b,
        "ms_minus_judgment": acc_ms - acc_j,
    }

