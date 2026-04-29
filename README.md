# Anonymous Code Artifact for BSTTT

This repository is a submission-safe, anonymous companion artifact for a NeurIPS 2026 Evaluations and Datasets paper on Belief-State Test-Time Training (BSTTT) and the explicit-to-applied Theory-of-Mind gap in large language models.

The artifact is intentionally curated rather than copied wholesale from the working research repository:

- included: executable core code, configs, paper assets, and lightweight utility scripts
- excluded: personal notes, cluster-specific launch scripts, caches, local tokens, and large private intermediate outputs

## Repository Layout

- `bsttt/`: Python package with loaders, evaluation code, model wrapper, and trainers
- `configs/`: configuration templates for the main benchmarks
- `scripts/`: small utility scripts for aggregating and summarizing results
- `paper/`: anonymous manuscript source and figure assets
- `docs/`: anonymity, reproducibility, and track-compliance notes

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Example Commands

SimpleToM frozen evaluation:

```bash
python -m bsttt.eval.eval_simpletom \
  --method frozen \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --max-items-per-subset 50 \
  --num-episodes-per-task 20
```

SimpleToM BSTTT with action reconstruction:

```bash
python -m bsttt.eval.eval_simpletom \
  --method bsttt \
  --bsttt-loss action_reconstruction \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --adapt-steps 3 \
  --lora-rank 8 \
  --lr 1e-4
```

External ToM benchmarks:

```bash
python -m bsttt.eval.eval_external_tom \
  --dataset hitom \
  --method frozen \
  --model-name-or-path openai/gpt-oss-20b
```

DynToM:

```bash
python -m bsttt.eval.eval_dyntom \
  --method frozen \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct
```

## Notes

- This artifact is structured for anonymous review. Before publishing a remote copy, read `docs/ANONYMITY_CHECKLIST.md`.
- Setup and execution guidance is in `docs/REPRODUCIBILITY.md`.
- Track-specific compliance notes are in `docs/TRACK_REQUIREMENTS.md`.
- The paper source references conference style and bibliography files that are not bundled in this artifact.
