"""
Evaluate external ToM MCQA datasets (Hi-ToM, OpenToM) with prompt baselines.

Methods:
  - frozen: direct MC option scoring
  - cot: chain-of-thought style prompt + MC option scoring
  - scratchpad_frozen: structured mental-state scratchpad prompt + MC option scoring
  - simtom: two-stage perspective-taking prompt baseline (proxy implementation)
  - symbolictom: two-stage symbolic belief-state prompt baseline (proxy implementation)
  - bsttt_ntl: LoRA episodic test-time training with next-token loss on in-scenario support
  - bsttt_ar: LoRA episodic test-time training with action-reconstruction loss on in-scenario support
  - scratchpad_bsttt_ar: scratchpad query prompt + BSTTT-AR adaptation on support
  - simtom_bsttt_ar: SimToM query prompt + BSTTT-AR adaptation on support
  - symbolictom_bsttt_ar: SymbolicToM query prompt + BSTTT-AR adaptation on support
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from bsttt.data.loaders.tom_external import ExternalToMExample, load_external_tom_processed
from bsttt.eval.bootstrap import bootstrap_ci
from bsttt.models.hf_lm_wrapper import HFCausalLMWrapper
from bsttt.trainers.prompt_baselines import build_cot_prompt, build_vanilla_prompt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_scratchpad_prompt(ex: ExternalToMExample) -> str:
    qtype = ex.question_type.lower()
    focus = "Track beliefs, emotions, intentions, and likely actions for each agent."
    if "attitude" in qtype or "emotion" in qtype or "sentiment" in qtype or "preference" in qtype:
        focus = "Prioritize latent mental states (beliefs, emotions, preferences, intentions)."
    elif "multihop" in qtype or "order" in qtype or "recursive" in qtype:
        focus = "Prioritize nested/recursive beliefs (who thinks what about whom)."
    elif "location" in qtype:
        focus = "Track physical state transitions and each agent's belief about locations."

    scratchpad = (
        "Mental-State Scratchpad:\n"
        "- Entities: identify key agents and objects.\n"
        "- State variables: belief / emotion / intention / action.\n"
        f"- Focus for this question: {focus}\n"
        "- Resolve conflicts between world state and each agent's belief before answering.\n\n"
    )
    return scratchpad + build_vanilla_prompt(ex.story, ex.question, ex.choices)


def _build_simtom_stage1_prompt(ex: ExternalToMExample) -> str:
    return (
        "You are performing perspective-taking for Theory of Mind.\n"
        "Task: extract only the facts that relevant agents can observe and infer.\n"
        "Do not answer the final question yet.\n\n"
        f"Story:\n{ex.story}\n\n"
        f"Question:\n{ex.question}\n\n"
        "Output format:\n"
        "1) Relevant agents\n"
        "2) Observable events per agent\n"
        "3) Inferred beliefs per agent\n"
    )


def _build_simtom_stage2_prompt(ex: ExternalToMExample, stage1_text: str) -> str:
    guidance = (
        "Perspective-Taking Notes:\n"
        f"{stage1_text}\n\n"
        "Use only these perspective-filtered notes to answer.\n\n"
    )
    return guidance + build_vanilla_prompt(ex.story, ex.question, ex.choices)


def _build_symbolictom_stage1_prompt(ex: ExternalToMExample) -> str:
    return (
        "Build a symbolic belief tracker for this story.\n"
        "Represent each state update as:\n"
        "[time] agent | belief | evidence\n"
        "Include nested beliefs if present.\n"
        "Do not answer the final question yet.\n\n"
        f"Story:\n{ex.story}\n\n"
        f"Question:\n{ex.question}\n\n"
        "Belief Tracker:\n"
    )


def _build_symbolictom_stage2_prompt(ex: ExternalToMExample, stage1_text: str) -> str:
    guidance = (
        "Belief Tracker State:\n"
        f"{stage1_text}\n\n"
        "Answer by following the tracker transitions and constraints.\n\n"
    )
    return guidance + build_vanilla_prompt(ex.story, ex.question, ex.choices)


def _canonical_story(story: str) -> str:
    # Normalize whitespace so semantically identical stories map to the same key.
    return re.sub(r"\s+", " ", (story or "").strip())


def _story_hash(story: str) -> str:
    canon = _canonical_story(story)
    return hashlib.md5(canon.encode("utf-8")).hexdigest()[:16]


def _build_bsttt_support_index(examples: Sequence[ExternalToMExample]) -> Tuple[Dict[str, List[int]], str]:
    by_scenario: Dict[str, List[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_scenario[ex.scenario_name].append(i)

    # If scenario names are effectively unique-per-question, fall back to story grouping.
    max_group = max((len(v) for v in by_scenario.values()), default=0)
    if max_group <= 1:
        by_story: Dict[str, List[int]] = defaultdict(list)
        for i, ex in enumerate(examples):
            key = f"{ex.dataset}_story_{_story_hash(ex.story)}"
            by_story[key].append(i)
        return by_story, "story_hash"

    return by_scenario, "scenario_name"


def _to_bsttt_support_row(ex: ExternalToMExample) -> Any:
    # Trainer expects SimpleToM-like fields; duck typing is sufficient.
    return SimpleNamespace(
        id=ex.id,
        story=ex.story,
        question=ex.question,
        choices=list(ex.choices),
        answer=ex.answer,
    )


def evaluate_prompt_method(
    *,
    model: HFCausalLMWrapper,
    examples: Sequence[ExternalToMExample],
    batch_size: int,
    max_seq_len: Optional[int],
    seed: int,
    method: str,
    stage1_max_new_tokens: int = 192,
    support_size: int = 4,
    support_policy: str = "any",
    adapt_steps: int = 3,
    lr: float = 1e-4,
    lora_top_fraction: float = 0.25,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Tuple[float, float, float]]]:
    stage1_texts: List[str] = []
    bsttt_losses: List[float] = []
    bsttt_support_sizes: List[int] = []
    if method == "frozen":
        prompts: List[str] = [build_vanilla_prompt(ex.story, ex.question, ex.choices) for ex in examples]
    elif method == "cot":
        prompts = [build_cot_prompt(ex.story, ex.question, ex.choices) for ex in examples]
    elif method == "scratchpad_frozen":
        prompts = [_build_scratchpad_prompt(ex) for ex in examples]
    elif method == "simtom":
        stage1_prompts = [_build_simtom_stage1_prompt(ex) for ex in examples]
        stage1_texts = model.generate(
            prompts=stage1_prompts,
            max_new_tokens=stage1_max_new_tokens,
            temperature=0.0,
            do_sample=False,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        prompts = [_build_simtom_stage2_prompt(ex, st) for ex, st in zip(examples, stage1_texts)]
    elif method == "symbolictom":
        stage1_prompts = [_build_symbolictom_stage1_prompt(ex) for ex in examples]
        stage1_texts = model.generate(
            prompts=stage1_prompts,
            max_new_tokens=stage1_max_new_tokens,
            temperature=0.0,
            do_sample=False,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        prompts = [_build_symbolictom_stage2_prompt(ex, st) for ex, st in zip(examples, stage1_texts)]
    elif method in (
        "bsttt_ntl",
        "bsttt_ar",
        "scratchpad_bsttt_ar",
        "simtom_bsttt_ar",
        "symbolictom_bsttt_ar",
    ):
        from bsttt.trainers.bsttt_simpletom import BSTTTLoRAConfig, BSTTTLoRATrainer

        cfg = BSTTTLoRAConfig(
            adapt_steps=adapt_steps,
            lr=lr,
            lora_top_fraction=lora_top_fraction,
            max_seq_len=max_seq_len,
        )
        trainer = BSTTTLoRATrainer(wrapper=model, cfg=cfg, seed=seed)
        by_group, support_grouping = _build_bsttt_support_index(examples)

        predictions: List[Dict[str, Any]] = []
        correct_all: List[bool] = []
        correct_by_qtype: Dict[str, List[bool]] = {}
        loss_name = "next_token_loss" if method == "bsttt_ntl" else "action_reconstruction"
        stage1_by_idx: Dict[int, str] = {}
        if method in ("symbolictom_bsttt_ar", "simtom_bsttt_ar"):
            if method == "symbolictom_bsttt_ar":
                stage1_prompts = [_build_symbolictom_stage1_prompt(ex) for ex in examples]
            else:
                stage1_prompts = [_build_simtom_stage1_prompt(ex) for ex in examples]
            stage1_texts = model.generate(
                prompts=stage1_prompts,
                max_new_tokens=stage1_max_new_tokens,
                temperature=0.0,
                do_sample=False,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )
            stage1_by_idx = {i: txt for i, txt in enumerate(stage1_texts)}

        for i, ex in enumerate(examples):
            trainer.reset_fast_weights()

            if support_grouping == "scenario_name":
                group_key = ex.scenario_name
            else:
                group_key = f"{ex.dataset}_story_{_story_hash(ex.story)}"

            support_idxs = [j for j in by_group.get(group_key, []) if j != i]
            if support_policy == "same_type":
                same_type = [j for j in support_idxs if examples[j].question_type == ex.question_type]
                if same_type:
                    support_idxs = same_type
            support_idxs = sorted(support_idxs, key=lambda j: examples[j].id)
            if support_size > 0:
                support_idxs = support_idxs[:support_size]
            support_rows = []
            for j in support_idxs:
                s = examples[j]
                # AR loss requires valid MC supervision; NTL can also use these rows safely.
                if s.choices and s.answer in s.choices:
                    support_rows.append(_to_bsttt_support_row(s))

            init_loss = None
            final_loss = None
            if support_rows:
                loss_curve, _, _, _, _, _ = trainer.adapt_on_support(
                    support_examples=support_rows,
                    bsttt_loss=loss_name,
                )
                if loss_curve:
                    init_loss = float(loss_curve[0])
                    final_loss = float(loss_curve[-1])
                    bsttt_losses.append(final_loss)
            bsttt_support_sizes.append(len(support_rows))

            if method == "symbolictom_bsttt_ar":
                prompt = _build_symbolictom_stage2_prompt(ex, stage1_by_idx.get(i, ""))
            elif method == "simtom_bsttt_ar":
                prompt = _build_simtom_stage2_prompt(ex, stage1_by_idx.get(i, ""))
            elif method == "scratchpad_bsttt_ar":
                prompt = _build_scratchpad_prompt(ex)
            else:
                prompt = build_vanilla_prompt(ex.story, ex.question, ex.choices)
            mc = trainer.wrapper.score_options(
                prompts=[prompt],
                options=[list(ex.choices)],
                batch_size=1,
                max_seq_len=max_seq_len,
            )[0]
            pred_choice = ex.choices[mc.pred_index]
            is_correct = pred_choice == ex.answer
            correct_all.append(is_correct)
            correct_by_qtype.setdefault(ex.question_type, []).append(is_correct)

            predictions.append(
                {
                    "id": ex.id,
                    "dataset": ex.dataset,
                    "scenario_name": ex.scenario_name,
                    "question_type": ex.question_type,
                    "question": ex.question,
                    "choices": ex.choices,
                    "answer": ex.answer,
                    "answer_index": ex.answer_index,
                    "prompt": prompt,
                    "pred_choice": pred_choice,
                    "pred_index": mc.pred_index,
                    "option_scores": mc.option_scores,
                    "correct": is_correct,
                    "stage1_text": stage1_by_idx.get(i),
                    "support_size_used": len(support_rows),
                    "support_grouping": support_grouping,
                    "support_group_key": group_key,
                    "support_policy": support_policy,
                    "adapt_loss_init": init_loss,
                    "adapt_loss_final": final_loss,
                }
            )

        metrics: Dict[str, float] = {
            "accuracy": float(np.mean(correct_all)) if correct_all else 0.0,
            "num_examples": float(len(correct_all)),
            "support_size_used_mean": float(np.mean(bsttt_support_sizes)) if bsttt_support_sizes else 0.0,
        }
        metrics["support_grouping_story_hash"] = 1.0 if support_grouping == "story_hash" else 0.0
        if bsttt_losses:
            metrics["adapt_final_loss_mean"] = float(np.mean(bsttt_losses))
        for qtype, vals in sorted(correct_by_qtype.items()):
            metrics[f"{qtype}_accuracy"] = float(np.mean(vals))
            metrics[f"{qtype}_num_examples"] = float(len(vals))

        bootstrap_cis: Dict[str, Tuple[float, float, float]] = {}
        vals = [1.0 if x else 0.0 for x in correct_all]
        if vals:
            mean, lower, upper = bootstrap_ci(vals, n_resamples=2000, ci=0.95, seed=seed)
            bootstrap_cis["accuracy_ci95"] = (mean, lower, upper)

        return predictions, metrics, bootstrap_cis
    else:
        raise ValueError(f"Unknown method: {method}")

    options: List[List[str]] = [list(ex.choices) for ex in examples]
    mc_outs = model.score_options(prompts=prompts, options=options, batch_size=batch_size, max_seq_len=max_seq_len)

    predictions: List[Dict[str, Any]] = []
    correct_all: List[bool] = []
    correct_by_qtype: Dict[str, List[bool]] = {}

    for i, (ex, mc, prompt) in enumerate(zip(examples, mc_outs, prompts)):
        pred_choice = ex.choices[mc.pred_index]
        is_correct = pred_choice == ex.answer
        correct_all.append(is_correct)
        correct_by_qtype.setdefault(ex.question_type, []).append(is_correct)

        predictions.append(
            {
                "id": ex.id,
                "dataset": ex.dataset,
                "scenario_name": ex.scenario_name,
                "question_type": ex.question_type,
                "question": ex.question,
                "choices": ex.choices,
                "answer": ex.answer,
                "answer_index": ex.answer_index,
                "prompt": prompt,
                "pred_choice": pred_choice,
                "pred_index": mc.pred_index,
                "option_scores": mc.option_scores,
                "correct": is_correct,
                "stage1_text": stage1_texts[i] if stage1_texts else None,
            }
        )

    metrics: Dict[str, float] = {
        "accuracy": float(np.mean(correct_all)) if correct_all else 0.0,
        "num_examples": float(len(correct_all)),
    }
    for qtype, vals in sorted(correct_by_qtype.items()):
        metrics[f"{qtype}_accuracy"] = float(np.mean(vals))
        metrics[f"{qtype}_num_examples"] = float(len(vals))

    bootstrap_cis: Dict[str, Tuple[float, float, float]] = {}
    vals = [1.0 if x else 0.0 for x in correct_all]
    if vals:
        mean, lower, upper = bootstrap_ci(vals, n_resamples=2000, ci=0.95, seed=seed)
        bootstrap_cis["accuracy_ci95"] = (mean, lower, upper)

    return predictions, metrics, bootstrap_cis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hitom", "opentom"], required=True)
    parser.add_argument(
        "--method",
        choices=[
            "frozen",
            "cot",
            "scratchpad_frozen",
            "simtom",
            "symbolictom",
            "bsttt_ntl",
            "bsttt_ar",
            "scratchpad_bsttt_ar",
            "simtom_bsttt_ar",
            "symbolictom_bsttt_ar",
        ],
        default="frozen",
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--split", default="auto")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-max-new-tokens", type=int, default=192)
    parser.add_argument("--support-size", type=int, default=4)
    parser.add_argument("--support-policy", choices=["any", "same_type"], default="any")
    parser.add_argument("--adapt-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-top-fraction", type=float, default=0.25)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/external_tom")
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_external_tom_processed(
        dataset_name=args.dataset,
        split=args.split,
        streaming=args.streaming,
        limit=args.limit,
        force_rebuild=args.force_rebuild,
    )

    model = HFCausalLMWrapper(
        model_name_or_path=args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device_map="auto",
    )

    t0 = time.time()
    predictions, metrics, bootstrap_cis = evaluate_prompt_method(
        model=model,
        examples=examples,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        method=args.method,
        stage1_max_new_tokens=args.stage1_max_new_tokens,
        support_size=args.support_size,
        support_policy=args.support_policy,
        adapt_steps=args.adapt_steps,
        lr=args.lr,
        lora_top_fraction=args.lora_top_fraction,
    )
    runtime = time.time() - t0

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "model": args.model_name_or_path,
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "runtime": runtime,
        "support_size": args.support_size,
        "support_policy": args.support_policy,
        "adapt_steps": args.adapt_steps,
        "lr": args.lr,
        "lora_top_fraction": args.lora_top_fraction,
        "metrics": metrics,
        "bootstrap_cis": bootstrap_cis,
    }

    pred_fp = out_dir / f"{args.dataset}_{args.method}_predictions_seed{args.seed}.json"
    sum_fp = out_dir / f"{args.dataset}_{args.method}_summary_seed{args.seed}.json"
    pred_fp.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    sum_fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{args.dataset} {args.method} accuracy: {metrics['accuracy']:.4f}")
    print(f"Predictions: {pred_fp}")
    print(f"Summary: {sum_fp}")


if __name__ == "__main__":
    main()
