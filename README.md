# BSTTT: Belief-State Test-Time Training

**Diagnosing the Explicit-to-Applied Theory of Mind Gap in Large Language Models**

BSTTT is a diagnostic evaluation protocol for Theory of Mind (ToM) in large language models. It distinguishes whether observed failures on applied ToM tasks (behavior prediction, judgment) reflect **absent knowledge** or **recoverable latent competence** by contrasting domain-aligned weight adaptation (action reconstruction) against matched generic adaptation (next-token prediction) under identical conditions.

---

## Key Idea

| Component | Description |
|-----------|-------------|
| **Explicit ToM** | Recognizing that an agent holds a particular mental state |
| **Applied ToM** | Using that recognition to predict behavior or evaluate actions |
| **BSTTT-AR** | Episodic LoRA adaptation with action reconstruction loss |
| **BSTTT-NTL** | Matched control: same setup but with next-token prediction loss |
| **Diagnostic** | If only AR recovers applied accuracy → domain-specific latent competence |

## Repository Structure

```
bsttt_anonymous_submission/
├── bsttt/                        # Core Python package
│   ├── data/
│   │   ├── loaders/              # Dataset loaders (SimpleToM, Hi-ToM, OpenToM, DynToM)
│   │   └── episode_builders/     # Episodic evaluation constructors
│   ├── eval/                     # Evaluation scripts per benchmark
│   │   ├── eval_simpletom.py     # SimpleToM evaluation (all methods)
│   │   ├── eval_external_tom.py  # Hi-ToM and OpenToM evaluation
│   │   ├── eval_dyntom.py        # DynToM evaluation
│   │   ├── metrics.py            # Accuracy and gap metrics
│   │   └── bootstrap.py          # Bootstrap confidence intervals
│   ├── models/
│   │   └── hf_lm_wrapper.py      # HuggingFace causal LM wrapper (scoring + generation)
│   └── trainers/
│       ├── bsttt_simpletom.py    # BSTTT LoRA trainer (episodic adaptation)
│       └── prompt_baselines.py   # Prompt templates (vanilla, CoT, MS-Reminder)
├── configs/                      # Benchmark and training configurations
├── scripts/                      # Result aggregation and reproduction utilities
│   └── reproduce_main_results.sh # Reproduce all main paper results
├── requirements.txt
└── pyproject.toml
```

## Setup

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1
- A GPU with ≥ 24 GB VRAM (for 7–8B models) or ≥ 48 GB (for 20B models)

### Installation

```bash
# Clone and install
git clone <anonymous-repo-url>
cd bsttt_anonymous_submission

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Frozen Baseline (SimpleToM)

Evaluate a model on SimpleToM with frozen inference (no adaptation):

```bash
python -m bsttt.eval.eval_simpletom \
  --method frozen \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --split test \
  --num-episodes-per-task 500 \
  --support-size 1 \
  --seed 42
```

### 2. BSTTT with Action Reconstruction

Episodic LoRA adaptation using the ToM-aligned objective:

```bash
python -m bsttt.eval.eval_simpletom \
  --method bsttt \
  --bsttt-loss action_reconstruction \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --split test \
  --num-episodes-per-task 500 \
  --support-size 1 \
  --adapt-steps 3 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lr 1e-4 \
  --seed 42
```

### 3. BSTTT with Next-Token Loss (Control)

Same adaptation setup, but generic next-token prediction:

```bash
python -m bsttt.eval.eval_simpletom \
  --method bsttt \
  --bsttt-loss next_token_loss \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --split test \
  --num-episodes-per-task 500 \
  --support-size 1 \
  --adapt-steps 3 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lr 1e-4 \
  --seed 42
```

### 4. MS-Reminder + BSTTT (Composed)

Combines mental-state prompting with weight adaptation:

```bash
python -m bsttt.eval.eval_simpletom \
  --method ms_reminder_bsttt \
  --bsttt-loss action_reconstruction \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --split test \
  --num-episodes-per-task 500 \
  --support-size 1 \
  --adapt-steps 3 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lr 1e-4 \
  --seed 42
```

### 5. External ToM Benchmarks (Hi-ToM, OpenToM)

```bash
python -m bsttt.eval.eval_external_tom \
  --dataset hitom \
  --method frozen \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --seed 42
```

### 6. DynToM

```bash
python -m bsttt.eval.eval_dyntom \
  --method frozen \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --seed 42
```

## Reproducing Main Results

To reproduce all results reported in the paper:

```bash
bash scripts/reproduce_main_results.sh
```

This runs all method × model × benchmark × seed combinations and writes results to `outputs/`. See the script for details and expected runtime.

## Method Configurations

| Method | CLI | Description |
|--------|-----|-------------|
| Frozen | `--method frozen` | Direct multiple-choice scoring, no adaptation |
| CoT | `--method cot` | Chain-of-thought prompt + option scoring |
| MS-Reminder | `--method ms_reminder` | Inject predicted mental state into applied prompts |
| BSTTT-AR | `--method bsttt --bsttt-loss action_reconstruction` | Episodic LoRA adaptation, ToM-aligned objective |
| BSTTT-NTL | `--method bsttt --bsttt-loss next_token_loss` | Episodic LoRA adaptation, generic objective (control) |
| MS-Reminder+BSTTT | `--method ms_reminder_bsttt` | Composed: prompting + weight adaptation |

## Hyperparameters

All results in the paper use these defaults (also in `configs/train_bsttt.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `adapt_steps` | 3 | Gradient steps per episode |
| `lr` | 1e-4 | Learning rate (AdamW) |
| `lora_rank` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling factor |
| `lora_top_fraction` | 0.25 | Apply LoRA to top 25% of layers |
| `weight_decay` | 0.0 | No weight decay |
| Seeds | {42, 43, 44} | 3 seeds for all reported results |

## Models

| Model | HuggingFace ID |
|-------|----------------|
| Qwen-2.5-7B | `Qwen/Qwen2.5-7B-Instruct` |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` |
| GPT-oss-20B | `openai/gpt-oss-20b` |

## Benchmarks

All benchmarks are loaded dynamically from HuggingFace Datasets and require no manual data download.

| Benchmark | Source | Questions |
|-----------|--------|-----------|
| SimpleToM | `allenai/SimpleToM` | 3,441 (1,147 scenarios × 3 types) |
| Hi-ToM | `mraheja/hi-tom` | 1,200 |
| OpenToM | `SeacowX/OpenToM` | 2,000 |
| DynToM | `dyntom/DynToM` | 3,000 |

## Output Format

Each evaluation run produces:

- `outputs/predictions/`: Per-item predictions (JSONL)
- `outputs/tables/`: Metrics summary (CSV + Markdown)
- `outputs/*_summary_*.json`: Run metadata + aggregate metrics + bootstrap CIs

## License

MIT
