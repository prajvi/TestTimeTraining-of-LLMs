"""
SimpleToM evaluation entry point.

Supported methods:
  - frozen / vanilla: multiple-choice scoring (log-likelihood)
  - cot: chain-of-thought prompt + answer-option scoring
  - ms_reminder: inject predicted mental-state into behavior/judgment prompts
  - bsttt: episodic LoRA adaptation (action reconstruction or next-token loss)
  - ms_reminder_bsttt: composed prompting + adaptation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from bsttt.data.episode_builders.simpletom_episode_builder import SimpleToMEpisodeBuilder
from bsttt.data.loaders.simpletom import QuestionType, load_simpletom_processed
from bsttt.eval.bootstrap import bootstrap_ci
from bsttt.eval.metrics import compute_simpletom_metrics
from bsttt.models.hf_lm_wrapper import HFCausalLMWrapper
from bsttt.trainers.prompt_baselines import (
    build_cot_prompt,
    build_ms_reminder_prompt,
    build_vanilla_prompt,
)


SimpleToMTask = Literal["mental_state", "behavior", "judgment"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_frozen(
    *,
    model: HFCausalLMWrapper,
    processed_examples: Sequence[Any],
    support_size: int,
    num_episodes_per_task: int,
    seed: int,
    batch_size: int,
    max_seq_len: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Tuple[float, float, float]]]:
    """
    Evaluate the frozen model on SimpleToM.

    Returns:
      (predictions, metrics, bootstrap_cis)
    """
    builder = SimpleToMEpisodeBuilder(processed_examples, seed=seed)

    query_tasks: List[SimpleToMTask] = ["mental_state", "behavior", "judgment"]

    predictions: List[Dict[str, Any]] = []
    correct_by_task: Dict[SimpleToMTask, List[bool]] = {t: [] for t in query_tasks}

    # Build all episodes up-front so scoring can batch across tasks.
    episodes = []
    for qt in query_tasks:
        eps = builder.build_episodes(
            query_task=qt,  # type: ignore[arg-type]
            support_size=support_size,
            num_episodes=num_episodes_per_task,
            support_task="behavior",
            raise_on_shortfall=False,
        )
        episodes.extend(eps)

    prompts: List[str] = []
    options: List[List[str]] = []
    for ep in episodes:
        q = ep.query
        prompts.append(build_vanilla_prompt(q.story, q.question, q.choices))
        options.append(list(q.choices))

    mc_outs = model.score_options(
        prompts=prompts,
        options=options,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    # Emit predictions and correctness.
    for ep, mc, prompt in zip(episodes, mc_outs, prompts):
        q = ep.query
        pred_choice = q.choices[mc.pred_index]
        is_correct = pred_choice == q.answer
        correct_by_task[q.question_type].append(is_correct)  # type: ignore[index]

        predictions.append(
            {
                "episode_id": ep.episode_id,
                "scenario_name": ep.scenario_name,
                "query_task": q.question_type,
                "query_id": q.id,
                "question": q.question,
                "choices": q.choices,
                "answer": q.answer,
                "prompt": prompt,
                "pred_choice": pred_choice,
                "pred_index": mc.pred_index,
                "option_scores": mc.option_scores,
                "correct": is_correct,
            }
        )

    metrics = compute_simpletom_metrics(
        correct_by_task=correct_by_task,  # type: ignore[arg-type]
    )
    metrics["mental_state_num_examples"] = len(correct_by_task["mental_state"])
    metrics["behavior_num_examples"] = len(correct_by_task["behavior"])
    metrics["judgment_num_examples"] = len(correct_by_task["judgment"])

    bootstrap_cis: Dict[str, Tuple[float, float, float]] = {}
    for t in query_tasks:
        values = [1.0 if x else 0.0 for x in correct_by_task[t]]
        mean, lower, upper = bootstrap_ci(values, n_resamples=2000, ci=0.95, seed=seed)
        bootstrap_cis[f"{t}_accuracy_ci95"] = (mean, lower, upper)

    return predictions, metrics, bootstrap_cis


def evaluate_vanilla(
    *,
    model: HFCausalLMWrapper,
    processed_examples: Sequence[Any],
    support_size: int,
    num_episodes_per_task: int,
    seed: int,
    batch_size: int,
    max_seq_len: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Tuple[float, float, float]]]:
    """Vanilla prompt: same as frozen (direct QA, option scoring)."""
    return evaluate_frozen(
        model=model,
        processed_examples=processed_examples,
        support_size=support_size,
        num_episodes_per_task=num_episodes_per_task,
        seed=seed,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )


def evaluate_cot(
    *,
    model: HFCausalLMWrapper,
    processed_examples: Sequence[Any],
    support_size: int,
    num_episodes_per_task: int,
    seed: int,
    batch_size: int,
    max_seq_len: Optional[int],
    max_new_tokens: int = 256,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Tuple[float, float, float]]]:
    """
    CoT baseline:
      - use a CoT-style prompt template
      - still score answer options with log-likelihood (robust multiple-choice evaluation)
    """
    builder = SimpleToMEpisodeBuilder(processed_examples, seed=seed)
    query_tasks: List[SimpleToMTask] = ["mental_state", "behavior", "judgment"]

    predictions: List[Dict[str, Any]] = []
    correct_by_task: Dict[SimpleToMTask, List[bool]] = {t: [] for t in query_tasks}

    episodes = []
    for qt in query_tasks:
        eps = builder.build_episodes(
            query_task=qt,  # type: ignore[arg-type]
            support_size=support_size,
            num_episodes=num_episodes_per_task,
            support_task="behavior",
            raise_on_shortfall=False,
        )
        episodes.extend(eps)

    prompts = [build_cot_prompt(ep.query.story, ep.query.question, ep.query.choices) for ep in episodes]
    options: List[List[str]] = [list(ep.query.choices) for ep in episodes]

    # `max_new_tokens` is unused now (we don't generate), but kept in the signature for CLI compatibility.
    mc_outs = model.score_options(
        prompts=prompts,
        options=options,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    for ep, mc, prompt in zip(episodes, mc_outs, prompts):
        q = ep.query
        pred_choice = q.choices[mc.pred_index]
        is_correct = pred_choice == q.answer
        correct_by_task[q.question_type].append(is_correct)  # type: ignore[index]

        predictions.append(
            {
                "episode_id": ep.episode_id,
                "scenario_name": ep.scenario_name,
                "query_task": q.question_type,
                "query_id": q.id,
                "question": q.question,
                "choices": q.choices,
                "answer": q.answer,
                "prompt": prompt,
                "pred_choice": pred_choice,
                "pred_index": mc.pred_index,
                "option_scores": mc.option_scores,
                "correct": is_correct,
            }
        )

    metrics = compute_simpletom_metrics(correct_by_task=correct_by_task)  # type: ignore[arg-type]
    metrics["mental_state_num_examples"] = len(correct_by_task["mental_state"])
    metrics["behavior_num_examples"] = len(correct_by_task["behavior"])
    metrics["judgment_num_examples"] = len(correct_by_task["judgment"])
    bootstrap_cis = {}
    for t in query_tasks:
        values = [1.0 if x else 0.0 for x in correct_by_task[t]]
        mean, lower, upper = bootstrap_ci(values, n_resamples=2000, ci=0.95, seed=seed)
        bootstrap_cis[f"{t}_accuracy_ci95"] = (mean, lower, upper)

    return predictions, metrics, bootstrap_cis


def evaluate_ms_reminder(
    *,
    model: HFCausalLMWrapper,
    processed_examples: Sequence[Any],
    support_size: int,
    num_episodes_per_task: int,
    seed: int,
    batch_size: int,
    max_seq_len: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Tuple[float, float, float]]]:
    """MS reminder: for behavior/judgment, inject model-predicted mental state into prompt."""
    builder = SimpleToMEpisodeBuilder(processed_examples, seed=seed)
    query_tasks: List[SimpleToMTask] = ["mental_state", "behavior", "judgment"]

    predictions: List[Dict[str, Any]] = []
    correct_by_task: Dict[SimpleToMTask, List[bool]] = {t: [] for t in query_tasks}

    episodes = []
    for qt in query_tasks:
        eps = builder.build_episodes(
            query_task=qt,  # type: ignore[arg-type]
            support_size=support_size,
            num_episodes=num_episodes_per_task,
            support_task="behavior",
            raise_on_shortfall=False,
        )
        episodes.extend(eps)

    # Step 1: For each episode, get predicted mental state if behavior/judgment.
    ms_predictions: Dict[str, str] = {}  # scenario_name -> predicted MS answer
    episodes_needing_ms: List[Tuple[int, str]] = []  # (ep_idx, scenario_name)
    for i, ep in enumerate(episodes):
        if ep.query.question_type in ("behavior", "judgment"):
            scenario = ep.scenario_name
            if scenario not in ms_predictions:
                ms_ex = builder.get_mental_state_example(scenario)
                if ms_ex is not None:
                    episodes_needing_ms.append((i, scenario))

    # Batch predict mental state for all unique scenarios.
    if episodes_needing_ms:
        scenarios_to_predict = list(dict.fromkeys(s for _, s in episodes_needing_ms))
        ms_examples = [builder.get_mental_state_example(s) for s in scenarios_to_predict]
        ms_examples = [ex for ex in ms_examples if ex is not None]
        if ms_examples:
            ms_prompts = [
                build_vanilla_prompt(ex.story, ex.question, ex.choices)
                for ex in ms_examples
            ]
            ms_options = [list(ex.choices) for ex in ms_examples]
            ms_outs = model.score_options(
                prompts=ms_prompts,
                options=ms_options,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )
            for (ex, mc) in zip(ms_examples, ms_outs):
                pred = ex.choices[mc.pred_index]
                ms_predictions[ex.scenario_name] = pred

    # Step 2: Build prompts and score.
    prompts: List[str] = []
    options: List[List[str]] = []
    for ep in episodes:
        q = ep.query
        if q.question_type == "mental_state":
            prompt = build_vanilla_prompt(q.story, q.question, q.choices)
        else:
            pred_ms = ms_predictions.get(ep.scenario_name, "Unknown")
            prompt = build_ms_reminder_prompt(q.story, q.question, q.choices, pred_ms)
        prompts.append(prompt)
        options.append(list(q.choices))

    mc_outs = model.score_options(
        prompts=prompts,
        options=options,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    for ep, mc, prompt in zip(episodes, mc_outs, prompts):
        q = ep.query
        pred_choice = q.choices[mc.pred_index]
        is_correct = pred_choice == q.answer
        correct_by_task[q.question_type].append(is_correct)  # type: ignore[index]

        predictions.append(
            {
                "episode_id": ep.episode_id,
                "scenario_name": ep.scenario_name,
                "query_task": q.question_type,
                "query_id": q.id,
                "question": q.question,
                "choices": q.choices,
                "answer": q.answer,
                "prompt": prompt,
                "pred_choice": pred_choice,
                "pred_index": mc.pred_index,
                "option_scores": mc.option_scores,
                "ms_hint": ms_predictions.get(ep.scenario_name) if q.question_type != "mental_state" else None,
                "correct": is_correct,
            }
        )

    metrics = compute_simpletom_metrics(correct_by_task=correct_by_task)  # type: ignore[arg-type]
    metrics["mental_state_num_examples"] = len(correct_by_task["mental_state"])
    metrics["behavior_num_examples"] = len(correct_by_task["behavior"])
    metrics["judgment_num_examples"] = len(correct_by_task["judgment"])
    bootstrap_cis = {}
    for t in query_tasks:
        values = [1.0 if x else 0.0 for x in correct_by_task[t]]
        mean, lower, upper = bootstrap_ci(values, n_resamples=2000, ci=0.95, seed=seed)
        bootstrap_cis[f"{t}_accuracy_ci95"] = (mean, lower, upper)

    return predictions, metrics, bootstrap_cis


def evaluate_bsttt(
    *,
    model: HFCausalLMWrapper,
    processed_examples: Sequence[Any],
    support_size: int,
    num_episodes_per_task: int,
    seed: int,
    batch_size: int,
    max_seq_len: Optional[int],
    adapt_steps: int,
    lr: float,
    weight_decay: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    bsttt_loss: Literal["action_reconstruction", "action_reconstruction_plus_consistency"] = "action_reconstruction",
    consistency_margin: float = 0.1,
    consistency_weight: float = 1.0,
    query_prompt_style: Literal["vanilla", "ms_reminder"] = "vanilla",
    support_prompt_style: Literal["vanilla", "ms_reminder"] = "vanilla",
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Tuple[float, float, float]]]:
    """
    BSTTT V1 evaluation on SimpleToM.

    For each episode:
      - reset LoRA fast weights
      - predict query before adaptation
      - adapt on behavior support with action reconstruction loss for K steps
      - predict query after adaptation
    """
    from bsttt.trainers.bsttt_simpletom import BSTTTLoRAConfig, BSTTTLoRATrainer

    builder = SimpleToMEpisodeBuilder(processed_examples, seed=seed)
    query_tasks: List[SimpleToMTask] = ["mental_state", "behavior", "judgment"]

    trainer = BSTTTLoRATrainer(
        wrapper=model,
        cfg=BSTTTLoRAConfig(
            adapt_steps=adapt_steps,
            lr=lr,
            weight_decay=weight_decay,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            max_seq_len=max_seq_len,
            enable_belief_action_consistency=bsttt_loss == "action_reconstruction_plus_consistency",
            consistency_margin=consistency_margin,
            consistency_weight=consistency_weight,
        ),
        seed=seed,
    )

    predictions: List[Dict[str, Any]] = []
    correct_before_by_task: Dict[SimpleToMTask, List[bool]] = {t: [] for t in query_tasks}
    correct_after_by_task: Dict[SimpleToMTask, List[bool]] = {t: [] for t in query_tasks}

    # Run episodes sequentially (LoRA reset + small K). This is the simplest correct baseline.
    episodes: List[Tuple[str, str, SimpleToMTask, Any]] = []
    for qt in query_tasks:
        # Some support sizes can be infeasible for a task on a given sampled subset.
        # We try progressively smaller support sizes until we can build at least
        # one episode for this query task (robustness for quick prototypes).
        base_support_size = support_size if qt != "behavior" else max(1, support_size - 1)
        eps: List[Any] = []
        for s in range(base_support_size, 0, -1):
            eps = builder.build_episodes(
                query_task=qt,  # type: ignore[arg-type]
                support_size=s,
                num_episodes=num_episodes_per_task,
                support_task="behavior",
                raise_on_shortfall=False,
            )
            if len(eps) > 0:
                break
        for ep in eps:
            episodes.append((ep.episode_id, ep.scenario_name, qt, ep))

    for _eid, _scenario, _qt, ep in episodes:
        q = ep.query
        ms_ex = builder.get_mental_state_example(ep.scenario_name)
        row = trainer.run_episode(
            episode_id=ep.episode_id,
            episode_scenario_name=ep.scenario_name,
            query_example=q,
            support_examples=ep.support,
            query_task=q.question_type,
            mental_state_example=ms_ex,
            bsttt_loss=bsttt_loss,
            query_prompt_style=query_prompt_style,
            support_prompt_style=support_prompt_style,
        )

        correct_before_by_task[q.question_type].append(bool(row["correct_before"]))
        correct_after_by_task[q.question_type].append(bool(row["correct_after"]))
        predictions.append(row)

    def accuracy_or_nan(correct: Sequence[bool]) -> float:
        if len(correct) == 0:
            return float("nan")
        return sum(1 for x in correct if x) / len(correct)

    def gap_or_nan(acc_ms: float, acc_other: float) -> float:
        if np.isnan(acc_ms) or np.isnan(acc_other):
            return float("nan")
        return float(acc_ms - acc_other)

    def compute_simpletom_metrics_safe(correct_by_task: Dict[SimpleToMTask, Sequence[bool]]) -> Dict[str, float]:
        acc_ms = accuracy_or_nan(correct_by_task["mental_state"])
        acc_b = accuracy_or_nan(correct_by_task["behavior"])
        acc_j = accuracy_or_nan(correct_by_task["judgment"])
        if np.isnan(acc_ms) or np.isnan(acc_b) or np.isnan(acc_j):
            acc_avg = float("nan")
        else:
            acc_avg = (acc_ms + acc_b + acc_j) / 3.0
        return {
            "mental_state_accuracy": float(acc_ms),
            "behavior_accuracy": float(acc_b),
            "judgment_accuracy": float(acc_j),
            "average_accuracy": float(acc_avg),
            "ms_minus_behavior": gap_or_nan(acc_ms, acc_b),
            "ms_minus_judgment": gap_or_nan(acc_ms, acc_j),
        }

    metrics_before = compute_simpletom_metrics_safe(correct_by_task=correct_before_by_task)  # type: ignore[arg-type]
    metrics_after = compute_simpletom_metrics_safe(correct_by_task=correct_after_by_task)  # type: ignore[arg-type]

    # Merge metrics with prefixes for a single table row.
    metrics: Dict[str, float] = {}
    metrics.update({f"before_{k}": float(v) for k, v in metrics_before.items()})
    metrics.update({f"after_{k}": float(v) for k, v in metrics_after.items()})
    for t in query_tasks:
        metrics[f"before_{t}_num_examples"] = float(len(correct_before_by_task[t]))
        metrics[f"after_{t}_num_examples"] = float(len(correct_after_by_task[t]))

    def bootstrap_ci_or_nan(values: Sequence[float]) -> Tuple[float, float, float]:
        if len(values) == 0:
            nan = float("nan")
            return (nan, nan, nan)
        return bootstrap_ci(values, n_resamples=2000, ci=0.95, seed=seed)

    bootstrap_cis: Dict[str, Tuple[float, float, float]] = {}
    for t in query_tasks:
        values_after = [1.0 if x else 0.0 for x in correct_after_by_task[t]]
        bootstrap_cis[f"after_{t}_accuracy_ci95"] = bootstrap_ci_or_nan(values_after)

    return predictions, metrics, bootstrap_cis


def save_predictions(predictions: Sequence[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_qualitative_examples(
    predictions: Sequence[Dict[str, Any]],
    out_path: Path,
    n_per_task: int = 2,
) -> None:
    """Save a few formatted examples for qualitative inspection."""
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for p in predictions:
        t = p.get("query_task", "unknown")
        by_task.setdefault(t, []).append(p)

    lines = ["# Qualitative Examples\n"]
    for task in ["mental_state", "behavior", "judgment"]:
        examples = by_task.get(task, [])[:n_per_task]
        if not examples:
            continue
        lines.append(f"\n## {task}\n")
        for i, ex in enumerate(examples, 1):
            lines.append(f"### Example {i}\n")
            lines.append(f"- **Question:** {ex.get('question', '')}\n")
            lines.append(f"- **Choices:** {ex.get('choices', [])}\n")
            lines.append(f"- **Gold:** {ex.get('answer', '')}\n")
            lines.append(f"- **Predicted:** {ex.get('pred_choice', '')}\n")
            lines.append(f"- **Correct:** {ex.get('correct', False)}\n")
            if ex.get("prompt"):
                lines.append(f"- **Prompt:**\n```\n{ex['prompt'][:800]}\n```\n")
            if ex.get("raw_output"):
                lines.append(f"- **Model output:**\n```\n{ex['raw_output'][:800]}\n```\n")
            if ex.get("ms_hint"):
                lines.append(f"- **MS hint:** {ex['ms_hint']}\n")
            lines.append("\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def save_tables(metrics_row: Dict[str, Any], *, out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame([metrics_row])
    df.to_csv(out_csv, index=False)

    # Avoid pandas' `to_markdown()` (requires optional `tabulate`).
    cols = list(df.columns)
    row = df.iloc[0].to_dict()

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "| " + " | ".join(fmt(row[c]) for c in cols) + " |"
    md = "\n".join([header, sep, body])
    out_md.write_text(md + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="simpletom", choices=["simpletom"])
    parser.add_argument(
        "--method",
        default="frozen",
        choices=["frozen", "vanilla", "cot", "ms_reminder", "bsttt", "ms_reminder_bsttt"],
        help="frozen/vanilla=option scoring; cot=CoT prompt + option scoring; ms_reminder=mental-state hint for behavior/judgment; bsttt=LoRA episodic adaptation; ms_reminder_bsttt=hybrid (MS reminder + BSTTT).",
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")

    # Quick prototype controls (keep small)
    parser.add_argument("--max-items-per-subset", type=int, default=3)
    parser.add_argument("--support-size", type=int, default=1)
    parser.add_argument("--num-episodes-per-task", type=int, default=1)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Force rebuilding cached SimpleToM processed examples.",
    )
    parser.add_argument(
        "--cache-stem",
        type=str,
        default="simpletom_processed_small",
        help="Cache stem used under bsttt/data/cache.",
    )

    # Runtime controls
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Kept for CLI compatibility; CoT baseline now uses option scoring (no generation).",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # BSTTT / LoRA adaptation controls
    parser.add_argument("--adapt-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--bsttt-loss",
        type=str,
        default="action_reconstruction",
        choices=["action_reconstruction", "action_reconstruction_plus_consistency", "next_token_loss"],
        help="BSTTT loss selection. Default is action reconstruction only. next_token_loss is generic TTT.",
    )
    parser.add_argument("--consistency-margin", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path(os.getcwd())
    out_dir = project_root / args.output_dir
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"

    # Load/prepare data
    processed = load_simpletom_processed(
        cache_stem=args.cache_stem,
        cache_max_items_per_subset=args.max_items_per_subset,
        force_rebuild=args.force_rebuild_cache,
        split=args.split,
        streaming=args.streaming,
        seed=args.seed,
    )

    # Load model
    model = HFCausalLMWrapper(
        model_name_or_path=args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device_map="auto",
    )

    eval_kw = dict(
        model=model,
        processed_examples=processed,
        support_size=args.support_size,
        num_episodes_per_task=args.num_episodes_per_task,
        seed=args.seed,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    t0 = time.time()
    if args.method in ("frozen", "vanilla"):
        predictions, metrics, bootstrap_cis = evaluate_frozen(**eval_kw)
    elif args.method == "cot":
        predictions, metrics, bootstrap_cis = evaluate_cot(**eval_kw, max_new_tokens=args.max_new_tokens)
    elif args.method == "ms_reminder":
        predictions, metrics, bootstrap_cis = evaluate_ms_reminder(**eval_kw)
    elif args.method == "bsttt":
        predictions, metrics, bootstrap_cis = evaluate_bsttt(
            **eval_kw,
            adapt_steps=args.adapt_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bsttt_loss=args.bsttt_loss,
            consistency_margin=args.consistency_margin,
            consistency_weight=args.consistency_weight,
            query_prompt_style="vanilla",
            support_prompt_style="vanilla",
        )
    elif args.method == "ms_reminder_bsttt":
        predictions, metrics, bootstrap_cis = evaluate_bsttt(
            **eval_kw,
            adapt_steps=args.adapt_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bsttt_loss=args.bsttt_loss,
            consistency_margin=args.consistency_margin,
            consistency_weight=args.consistency_weight,
            query_prompt_style="ms_reminder",
            support_prompt_style="ms_reminder",
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    dt = time.time() - t0

    run_meta = {
        "model": args.model_name_or_path,
        "method": args.method,
        "dataset": args.dataset,
        "max_items_per_subset": args.max_items_per_subset,
        "support_size": args.support_size,
        "num_episodes_per_task": args.num_episodes_per_task,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dtype": args.dtype,
        "split": args.split,
        "streaming": args.streaming,
        "cache_stem": args.cache_stem,
        "force_rebuild_cache": args.force_rebuild_cache,
        "runtime_seconds": dt,
    }
    if args.method == "cot":
        run_meta["max_new_tokens"] = args.max_new_tokens
    if args.method == "bsttt":
        run_meta.update(
            {
                "adapt_steps": args.adapt_steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "bsttt_loss": args.bsttt_loss,
                "consistency_margin": args.consistency_margin,
                "consistency_weight": args.consistency_weight,
            }
        )
    if args.method == "ms_reminder_bsttt":
        run_meta.update(
            {
                "adapt_steps": args.adapt_steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "bsttt_loss": args.bsttt_loss,
                "consistency_margin": args.consistency_margin,
                "consistency_weight": args.consistency_weight,
                "hybrid_prompt": "ms_reminder",
            }
        )

    # Save predictions
    ts = time.strftime("%Y%m%d_%H%M%S")
    pred_path = pred_dir / f"simpletom_{args.method}_{ts}.jsonl"
    save_predictions(predictions, pred_path)

    # Save qualitative examples
    qual_path = pred_dir / f"simpletom_{args.method}_qualitative_{ts}.md"
    save_qualitative_examples(predictions, qual_path, n_per_task=2)

    # Save metrics + CI
    metrics_row: Dict[str, Any] = {"method": args.method}
    metrics_row.update(metrics)
    for k, (mean, lower, upper) in bootstrap_cis.items():
        metrics_row[k] = mean
        metrics_row[f"{k}_lower"] = lower
        metrics_row[f"{k}_upper"] = upper

    metrics_row["runtime_seconds"] = dt

    out_csv = table_dir / f"simpletom_{args.method}_metrics_{ts}.csv"
    out_md = table_dir / f"simpletom_{args.method}_metrics_{ts}.md"
    save_tables(metrics_row, out_csv=out_csv, out_md=out_md)

    # Also write a single JSON summary for convenience.
    summary_path = out_dir / f"simpletom_{args.method}_summary_{ts}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"meta": run_meta, "metrics": metrics, "bootstrap_cis": bootstrap_cis}, indent=2), encoding="utf-8")

    print(f"=== {args.method} evaluation complete ===")
    print("Predictions:", pred_path)
    print("Metrics CSV:", out_csv)
    print("Metrics MD:", out_md)
    print("Summary JSON:", summary_path)


if __name__ == "__main__":
    main()
