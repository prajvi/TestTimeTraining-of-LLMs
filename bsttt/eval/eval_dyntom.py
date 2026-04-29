"""
DynToM evaluation script.
Methods:
  - frozen: no adaptation
  - cot: chain-of-thought style prompt + MC option scoring
  - bsttt: LoRA episodic adaptation using Action Reconstruction on type_a questions
  - scratchpad_frozen: 2-pass scratchpad built from model-predicted type_a answers
  - scratchpad_oracle: 2-pass oracle scratchpad for type_c/type_d transformation questions
  - scratchpad_ttt: 2-pass scratchpad with trial-level LoRA adaptation on type_a
  - hierarchical_ttt: trial-level LoRA adaptation on type_a, evaluate on type_c/type_d
"""

from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

from bsttt.data.loaders.dyntom import load_dyntom_processed, DynToMExample
from bsttt.data.episode_builders.dyntom_episode_builder import DynToMEpisodeBuilder

if TYPE_CHECKING:
    from bsttt.models.hf_lm_wrapper import HFCausalLMWrapper


SCRATCHPAD_STATES: Tuple[str, ...] = ("belief", "emotion", "intention", "action")
SCRATCHPAD_SCENARIOS: Tuple[int, ...] = (1, 2, 3, 4, 5)
TRANSFORMATION_TYPES = {"type_c", "type_d"}
TOTAL_SCRATCHPAD_CELLS = len(SCRATCHPAD_STATES) * len(SCRATCHPAD_SCENARIOS)
STATE_TO_INDEX = {state: idx for idx, state in enumerate(SCRATCHPAD_STATES)}


def _extract_state_and_scenario(question: str) -> Optional[Tuple[str, int]]:
    q = question.lower()
    state_m = re.search(r"\b(belief|emotion|intention|action)\b", q)
    scenario_m = re.search(r"\bscenario\s+(\d+)\b", q)
    if state_m is None or scenario_m is None:
        return None
    return state_m.group(1), int(scenario_m.group(1))


def _strip_option_prefix(text: str) -> str:
    t = text.strip()
    if len(t) >= 3 and t[0].isalpha() and t[1] == ".":
        return t[3:].strip()
    return t


def _normalize_choices_with_letters(choices: Sequence[str]) -> Tuple[List[str], Dict[int, str]]:
    clean_options: List[str] = []
    option_letter_map: Dict[int, str] = {}
    for idx, opt in enumerate(choices):
        if len(opt) >= 3 and opt[1] == ".":
            letter = opt[0].lower()
            clean_text = opt[3:].strip()
        else:
            letter = chr(ord("a") + idx)
            clean_text = opt
        clean_options.append(clean_text)
        option_letter_map[idx] = letter
    return clean_options, option_letter_map


def _score_mc_question(
    *,
    model: "HFCausalLMWrapper",
    prompt: str,
    choices: Sequence[str],
    max_seq_len: Optional[int] = None,
) -> Tuple[int, str, List[str], List[float]]:
    clean_options, option_letter_map = _normalize_choices_with_letters(choices)
    mc_out = model.score_options(
        prompts=[prompt],
        options=[clean_options],
        batch_size=1,
        max_seq_len=max_seq_len,
    )[0]
    pred_idx = mc_out.pred_index
    pred_letter = option_letter_map.get(pred_idx, chr(ord("a") + pred_idx))
    return pred_idx, pred_letter, clean_options, mc_out.option_scores


def _type_a_sort_key(ex: DynToMExample) -> Tuple[int, int, str]:
    slot = _extract_state_and_scenario(ex.question)
    if slot is None:
        return (99, 99, ex.id)
    state, scenario = slot
    return (scenario, STATE_TO_INDEX.get(state, 99), ex.id)


def _snapshot_fast_weights(trainer: Any) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for name, p in trainer.wrapper.model.named_parameters():
        if p.requires_grad:
            snap[name] = p.detach().cpu().clone()
    return snap


def _load_fast_weights(trainer: Any, snapshot: Dict[str, Any]) -> None:
    for name, p in trainer.wrapper.model.named_parameters():
        if not p.requires_grad:
            continue
        init = snapshot.get(name)
        if init is None:
            continue
        p.data.copy_(init.to(p.device))


def _make_ttt_support_example(ex: DynToMExample, all_scenario_texts: Dict[int, str]) -> Any:
    story_ctx = all_scenario_texts.get(ex.turn_id, ex.story)
    return SimpleNamespace(
        id=ex.id,
        story=story_ctx,
        question=ex.question,
        choices=list(ex.choices),
        answer=ex.answer,
    )


def build_oracle_state_table(
    type_a_examples: Sequence[DynToMExample],
) -> Tuple[Dict[int, Dict[str, str]], float, List[str]]:
    """
    Build a deterministic 5x4 oracle table:
      rows: scenarios 1..5
      cols: belief/emotion/intention/action
    """
    table: Dict[int, Dict[str, str]] = {
        s: {state: "[MISSING]" for state in SCRATCHPAD_STATES}
        for s in SCRATCHPAD_SCENARIOS
    }
    filled = set()
    duplicate_slots: List[str] = []

    for ex in sorted(type_a_examples, key=_type_a_sort_key):
        slot = _extract_state_and_scenario(ex.question)
        if slot is None:
            continue
        state, scenario = slot
        if scenario not in table:
            continue
        answer_text = _strip_option_prefix(ex.answer)
        if not answer_text:
            continue

        key = (scenario, state)
        if key in filled:
            duplicate_slots.append(f"scenario {scenario}:{state}")
            continue

        table[scenario][state] = answer_text
        filled.add(key)

    coverage = len(filled) / float(TOTAL_SCRATCHPAD_CELLS)
    return table, coverage, duplicate_slots


def build_frozen_state_table(
    *,
    model: "HFCausalLMWrapper",
    type_a_examples: Sequence[DynToMExample],
    all_scenario_texts: Dict[int, str],
    char_info: str,
    max_seq_len: Optional[int] = None,
) -> Tuple[Dict[int, Dict[str, str]], float, List[str]]:
    """
    Build a deterministic 5x4 scratchpad from model predictions on type_a questions.
    """
    table: Dict[int, Dict[str, str]] = {
        s: {state: "[MISSING]" for state in SCRATCHPAD_STATES}
        for s in SCRATCHPAD_SCENARIOS
    }
    filled = set()
    duplicate_slots: List[str] = []

    for ex in sorted(type_a_examples, key=_type_a_sort_key):
        slot = _extract_state_and_scenario(ex.question)
        if slot is None:
            continue
        state, scenario = slot
        if scenario not in table:
            continue
        if not ex.choices:
            continue

        key = (scenario, state)
        if key in filled:
            duplicate_slots.append(f"scenario {scenario}:{state}")
            continue

        ep_like = SimpleNamespace(query=SimpleNamespace(question=ex.question, choices=ex.choices))
        prompt = build_dyntom_prompt(ep_like, all_scenario_texts, char_info)
        pred_idx, _pred_letter, clean_options, _scores = _score_mc_question(
            model=model,
            prompt=prompt,
            choices=ex.choices,
            max_seq_len=max_seq_len,
        )
        if not (0 <= pred_idx < len(clean_options)):
            continue

        pred_text = _strip_option_prefix(clean_options[pred_idx])
        if not pred_text:
            continue
        table[scenario][state] = pred_text
        filled.add(key)

    coverage = len(filled) / float(TOTAL_SCRATCHPAD_CELLS)
    return table, coverage, duplicate_slots


def render_oracle_state_table(table: Dict[int, Dict[str, str]]) -> str:
    lines = [
        "Mental State Scratchpad (Oracle from type_a ground truth):",
        "| Scenario | Belief | Emotion | Intention | Action |",
        "|---|---|---|---|---|",
    ]
    for s in SCRATCHPAD_SCENARIOS:
        cells = []
        for state in SCRATCHPAD_STATES:
            cell = table[s][state].replace("\n", " ").replace("|", "/").strip()
            cells.append(cell if cell else "[MISSING]")
        lines.append(f"| {s} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    return "\n".join(lines)


def build_dyntom_prompt(
    episode,
    all_scenario_texts: dict,
    char_info: str,
    scratchpad_text: Optional[str] = None,
    cot: bool = False,
) -> str:
    """
    Build a rich prompt that includes:
    1. Character information
    2. ALL relevant scenario contexts (full trajectory)
    3. The question with explicit MC instructions
    """
    parts = []
    
    # 1. Character & setting context
    if char_info:
        parts.append(f"Characters:\n{char_info}\n")

    # 1.5 Optional scratchpad block (for transformation questions in scratchpad methods)
    if scratchpad_text:
        parts.append(f"{scratchpad_text}\n")
    
    # 2. Include ALL scenario texts for the trial
    for s_num in sorted(all_scenario_texts.keys()):
        parts.append(all_scenario_texts[s_num])
    
    # 3. Question with explicit instructions
    parts.append(f"\nBased on the scenarios above, answer the following multiple-choice question.")
    if cot:
        parts.append(
            "\nReason step by step about beliefs, emotions, intentions, and actions before choosing the best option."
        )
    parts.append(f"\nQuestion: {episode.query.question}")
    
    # 4. List all options clearly
    parts.append("\nOptions:")
    for opt in episode.query.choices:
        parts.append(f"  {opt}")
    
    if cot:
        parts.append("\nAnswer (option text):")
    else:
        parts.append("\nAnswer:")
    
    return "\n".join(parts)


def evaluate_dyntom(
    *,
    model: "HFCausalLMWrapper",
    processed_examples: Sequence[Any],
    method: str,
    support_size: int,
    seed: int,
    batch_size: int,
    full_index_examples: Optional[Sequence[Any]] = None,
    max_seq_len: Optional[int] = None,
    adapt_steps: int = 3,
    lr: float = 1e-4,
    lora_top_fraction: float = 0.25,
    max_ttt_support: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
    print("Building episodes...")
    
    # Build episodes with mode="questions" so support = other questions from same trial  
    mode = "questions" if method == "bsttt" else "turns"
    builder = DynToMEpisodeBuilder(processed_examples, seed=seed)
    episodes = builder.build_episodes(support_size=support_size, mode=mode)
    if method == "hierarchical_ttt":
        episodes = [ep for ep in episodes if ep.query.question_type in TRANSFORMATION_TYPES]
    
    if not episodes:
        print(f"WARNING: No episodes built for DynToM. support_size={support_size}, num_examples={len(processed_examples)}")
        return [], {"accuracy": 0.0}, {}

    print(f"Evaluating {len(episodes)} episodes using {method} (mode={mode})...")
    
    # Pre-compute per-trial scenario texts and character info
    trial_scenarios: Dict[str, Dict[int, str]] = {}
    trial_char_info: Dict[str, str] = {}
    for ex in processed_examples:
        if ex.question_type == "dyntom":
            trial_scenarios.setdefault(ex.scenario_name, {})[ex.turn_id] = ex.story
        if ex.characters_info and ex.scenario_name not in trial_char_info:
            trial_char_info[ex.scenario_name] = ex.characters_info

    # Build per-trial type_a index from full (unlimited) set when available.
    index_source = full_index_examples if full_index_examples is not None else processed_examples
    trial_type_a: Dict[str, List[DynToMExample]] = {}
    for ex in index_source:
        if ex.question_type == "type_a":
            trial_type_a.setdefault(ex.scenario_name, []).append(ex)

    # Initialize Trainer once (prevents Peft adapter nesting)
    trainer = None
    if method in ("bsttt", "scratchpad_ttt", "hierarchical_ttt"):
        from bsttt.trainers.bsttt_simpletom import BSTTTLoRATrainer, BSTTTLoRAConfig
        cfg = BSTTTLoRAConfig(
            adapt_steps=adapt_steps,
            lr=lr,
            enable_temporal_smoothness=False,
            lora_top_fraction=lora_top_fraction,
            max_seq_len=max_seq_len,
        )
        trainer = BSTTTLoRATrainer(wrapper=model, cfg=cfg)

    # Build trial-level scratchpads once and reuse in Pass 2.
    trial_scratchpad: Dict[str, Dict[str, Any]] = {}
    trial_fast_weights: Dict[str, Dict[str, Any]] = {}
    if method in ("scratchpad_oracle", "scratchpad_frozen", "scratchpad_ttt"):
        for trial_id in trial_scenarios.keys():
            type_a_examples = trial_type_a.get(trial_id, [])
            all_scenarios = trial_scenarios.get(trial_id, {})
            char_info = trial_char_info.get(trial_id, "")
            if method == "scratchpad_oracle":
                table, coverage, duplicate_slots = build_oracle_state_table(type_a_examples)
            elif method == "scratchpad_frozen":
                table, coverage, duplicate_slots = build_frozen_state_table(
                    model=model,
                    type_a_examples=type_a_examples,
                    all_scenario_texts=all_scenarios,
                    char_info=char_info,
                    max_seq_len=max_seq_len,
                )
            else:
                if trainer is None:
                    raise RuntimeError("scratchpad_ttt requires LoRA trainer initialization.")
                trainer.reset_fast_weights()
                ttt_support = []
                for ex in sorted(type_a_examples, key=_type_a_sort_key):
                    if ex.choices and ex.answer and ex.answer in ex.choices:
                        ttt_support.append(_make_ttt_support_example(ex, all_scenarios))
                if max_ttt_support is not None and max_ttt_support > 0:
                    ttt_support = ttt_support[:max_ttt_support]
                if ttt_support:
                    trainer.adapt_on_support(
                        support_examples=ttt_support,
                        bsttt_loss="action_reconstruction",
                    )
                table, coverage, duplicate_slots = build_frozen_state_table(
                    model=trainer.wrapper,
                    type_a_examples=type_a_examples,
                    all_scenario_texts=all_scenarios,
                    char_info=char_info,
                    max_seq_len=max_seq_len,
                )
                trial_fast_weights[trial_id] = _snapshot_fast_weights(trainer)
            trial_scratchpad[trial_id] = {
                "text": render_oracle_state_table(table),
                "coverage": coverage,
                "filled_cells": int(round(coverage * TOTAL_SCRATCHPAD_CELLS)),
                "duplicate_slots_count": len(duplicate_slots),
            }
    elif method == "hierarchical_ttt":
        if trainer is None:
            raise RuntimeError("hierarchical_ttt requires LoRA trainer initialization.")
        for trial_id in trial_scenarios.keys():
            all_scenarios = trial_scenarios.get(trial_id, {})
            type_a_examples = trial_type_a.get(trial_id, [])
            trainer.reset_fast_weights()
            ttt_support = []
            for ex in sorted(type_a_examples, key=_type_a_sort_key):
                if ex.choices and ex.answer and ex.answer in ex.choices:
                    ttt_support.append(_make_ttt_support_example(ex, all_scenarios))
            if max_ttt_support is not None and max_ttt_support > 0:
                ttt_support = ttt_support[:max_ttt_support]
            if ttt_support:
                trainer.adapt_on_support(
                    support_examples=ttt_support,
                    bsttt_loss="action_reconstruction",
                )
            trial_fast_weights[trial_id] = _snapshot_fast_weights(trainer)

    predictions = []
    correct_list = []
    scratchpad_coverages: List[float] = []

    for i, ep in enumerate(episodes):
        if i % 50 == 0:
            print(f"Episode {i}/{len(episodes)}...")
            
        if method == "bsttt" and trainer is not None:
            trainer.reset_fast_weights()
            
            # Filter support to only examples that have valid choices + answers
            valid_support = [s for s in ep.support if s.choices and s.answer and s.answer in s.choices]
            if max_ttt_support is not None and max_ttt_support > 0:
                valid_support = valid_support[:max_ttt_support]
            
            if valid_support:
                loss_curve = trainer.adapt_on_support(
                    support_examples=valid_support,
                    bsttt_loss="action_reconstruction"
                )
                if i % 200 == 0 and loss_curve[1]:
                    init_l = loss_curve[1][0]
                    final_l = loss_curve[1][-1]
                    print(f"  AR Loss: {init_l:.4f} -> {final_l:.4f} (support={len(valid_support)})")
        elif method == "scratchpad_ttt" and trainer is not None:
            snapshot = trial_fast_weights.get(ep.scenario_name)
            if snapshot is not None:
                _load_fast_weights(trainer, snapshot)
        elif method == "hierarchical_ttt" and trainer is not None:
            snapshot = trial_fast_weights.get(ep.scenario_name)
            if snapshot is not None:
                _load_fast_weights(trainer, snapshot)
        
        scratchpad_text = None
        table_coverage = None
        table_filled_cells = None
        table_duplicate_slots_count = None
        if method in ("scratchpad_oracle", "scratchpad_frozen", "scratchpad_ttt") and ep.query.question_type in TRANSFORMATION_TYPES:
            trial_sp = trial_scratchpad.get(ep.scenario_name)
            if trial_sp is not None:
                scratchpad_text = trial_sp["text"]
                table_coverage = trial_sp["coverage"]
                table_filled_cells = trial_sp["filled_cells"]
                table_duplicate_slots_count = trial_sp["duplicate_slots_count"]
                scratchpad_coverages.append(float(table_coverage))

        # Build rich prompt with full context (plus optional scratchpad)
        all_scenarios = trial_scenarios.get(ep.scenario_name, {})
        char_info = trial_char_info.get(ep.scenario_name, "")
        prompt = build_dyntom_prompt(
            ep,
            all_scenarios,
            char_info,
            scratchpad_text=scratchpad_text,
            cot=(method == "cot"),
        )
        
        # Score each option using log-likelihood
        # Strip the letter prefix ("a. ", "b. ", etc.) so model scores just the content
        pred_idx, pred_letter, clean_options, option_scores = _score_mc_question(
            model=model,
            prompt=prompt,
            choices=ep.query.choices,
            max_seq_len=max_seq_len,
        )
        
        # True answer is stored as full choice string like "g. Suspicious activity -> ..."
        # Extract the letter prefix from the stored answer
        true_answer_str = ep.query.answer.strip()
        if len(true_answer_str) >= 2 and true_answer_str[1] == '.':
            true_letter = true_answer_str[0].lower()
        else:
            true_letter = true_answer_str.lower()
        
        is_correct = (pred_letter == true_letter)
        
        correct_list.append(is_correct)
        predictions.append({
            "id": ep.episode_id,
            "pred_index": pred_idx,
            "pred": pred_letter,
            "answer": true_letter,
            "correct": is_correct,
            "question_type": ep.query.question_type,
            "question": ep.query.question[:100],
            "scores": option_scores,
            "scratchpad_used": bool(scratchpad_text),
            "table_coverage": table_coverage,
            "table_filled_cells": table_filled_cells,
            "table_duplicate_slots_count": table_duplicate_slots_count,
        })

    acc = sum(correct_list) / len(correct_list) if correct_list else 0
    metrics = {"accuracy": acc}
    print(f"Accuracy: {acc:.4f}")
    
    # Print per-question-type breakdown
    if predictions:
        type_correct: Dict[str, List[bool]] = {}
        for p in predictions:
            type_correct.setdefault(p["question_type"], []).append(p["correct"])
        for qtype, vals in sorted(type_correct.items()):
            tp = sum(vals)
            print(f"  {qtype}: {tp/len(vals):.4f} ({tp}/{len(vals)})")
        if "type_c" in type_correct:
            metrics["type_c_accuracy"] = sum(type_correct["type_c"]) / len(type_correct["type_c"])
        if "type_d" in type_correct:
            metrics["type_d_accuracy"] = sum(type_correct["type_d"]) / len(type_correct["type_d"])
        transform_vals = type_correct.get("type_c", []) + type_correct.get("type_d", [])
        if transform_vals:
            metrics["transformation_accuracy"] = sum(transform_vals) / len(transform_vals)

    if method in ("scratchpad_oracle", "scratchpad_frozen", "scratchpad_ttt"):
        avg_cov = (sum(scratchpad_coverages) / len(scratchpad_coverages)) if scratchpad_coverages else 0.0
        metrics["table_coverage"] = avg_cov
        print(f"  table_coverage: {avg_cov:.4f}")

    return predictions, metrics, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="frozen",
        choices=["frozen", "cot", "bsttt", "scratchpad_frozen", "scratchpad_oracle", "scratchpad_ttt", "hierarchical_ttt"],
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--support-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapt-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-top-fraction", type=float, default=0.25)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument(
        "--max-ttt-support",
        type=int,
        default=None,
        help="Optional cap on support examples used for adaptation (memory/time control).",
    )
    parser.add_argument("--output-dir", default="outputs/dyntom")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to load")
    args = parser.parse_args()

    # Load data
    processed = load_dyntom_processed(limit=args.limit)
    print(f"Loaded {len(processed)} total examples (turns + questions).")

    full_index_examples = None
    if args.method in ("scratchpad_oracle", "scratchpad_frozen", "scratchpad_ttt", "hierarchical_ttt"):
        if args.limit is None:
            full_index_examples = processed
        else:
            eval_trials = {
                ex.scenario_name
                for ex in processed
                if ex.question_type != "dyntom"
            }
            full_processed = load_dyntom_processed(limit=None)
            full_index_examples = [ex for ex in full_processed if ex.scenario_name in eval_trials]
        print(
            "Scratchpad index prepared from "
            f"{len(full_index_examples)} examples across "
            f"{len({ex.scenario_name for ex in full_index_examples})} trials."
        )

    # Load model
    from bsttt.models.hf_lm_wrapper import HFCausalLMWrapper
    model = HFCausalLMWrapper(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        device_map="auto",
    )

    t0 = time.time()
    predictions, metrics, cis = evaluate_dyntom(
        model=model,
        processed_examples=processed,
        method=args.method,
        support_size=args.support_size,
        seed=args.seed,
        batch_size=4,
        full_index_examples=full_index_examples,
        max_seq_len=args.max_seq_len,
        adapt_steps=args.adapt_steps,
        lr=args.lr,
        lora_top_fraction=args.lora_top_fraction,
        max_ttt_support=args.max_ttt_support,
    )
    dt = time.time() - t0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with (out_dir / f"dyntom_{args.method}_predictions.json").open("w") as f:
        json.dump(predictions, f, indent=2, default=str)
    
    with (out_dir / f"dyntom_{args.method}_summary.json").open("w") as f:
        json.dump(
            {
                "metrics": metrics,
                "runtime": dt,
                "method": args.method,
                "meta": {
                    "model": args.model_name_or_path,
                    "seed": args.seed,
                    "limit": args.limit,
                    "dtype": args.dtype,
                    "max_seq_len": args.max_seq_len,
                    "adapt_steps": args.adapt_steps,
                    "lr": args.lr,
                    "lora_top_fraction": args.lora_top_fraction,
                    "max_ttt_support": args.max_ttt_support,
                },
            },
            f,
            indent=2,
        )

    print(f"DynToM {args.method} Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
