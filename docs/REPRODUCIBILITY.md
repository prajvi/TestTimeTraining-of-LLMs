# Reproducibility Notes

## Environment

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Main Entry Points

- `python -m bsttt.eval.eval_simpletom`
- `python -m bsttt.eval.eval_external_tom`
- `python -m bsttt.eval.eval_dyntom`

## Expected Dependencies

The code expects:

- PyTorch
- Transformers
- Datasets
- PEFT
- Accelerate

Some benchmark/model combinations may also require:

- model-specific tokenizer support
- access credentials for gated or rate-limited model providers
- sufficient GPU memory for the selected model size

## Artifact Scope

This anonymous repo is intended to make the codebase reviewable and runnable.
It does not bundle the full private working repository, complete experiment history, or every large intermediate artifact produced during development.
