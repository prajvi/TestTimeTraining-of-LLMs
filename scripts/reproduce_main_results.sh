#!/usr/bin/env bash
# ============================================================================
# reproduce_main_results.sh
#
# Reproduce the main results reported in the paper.
# This script runs all method × model × seed combinations for SimpleToM,
# Hi-ToM, OpenToM, and DynToM.
#
# Usage:
#   bash scripts/reproduce_main_results.sh
#
# Requirements:
#   - GPU with ≥ 48 GB VRAM (for 20B model) or run models sequentially
#   - All dependencies installed (pip install -r requirements.txt && pip install -e .)
#   - HuggingFace access tokens set for gated models (export HF_TOKEN=...)
#
# Estimated runtime: ~24 GPU-hours on a single B200/H200 GPU.
# ============================================================================

set -euo pipefail

OUTPUT_DIR="${1:-outputs/reproduction}"
SEEDS=(42 43 44)

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "openai/gpt-oss-20b"
)

# ---- SimpleToM ----
SIMPLETOM_METHODS=(frozen cot ms_reminder bsttt ms_reminder_bsttt)
BSTTT_LOSSES_SIMPLETOM=(action_reconstruction next_token_loss)

for model in "${MODELS[@]}"; do
    model_short=$(echo "$model" | sed 's|.*/||')
    for seed in "${SEEDS[@]}"; do
        # Non-adaptation methods
        for method in frozen cot ms_reminder; do
            echo "=== SimpleToM | ${model_short} | ${method} | seed=${seed} ==="
            python -m bsttt.eval.eval_simpletom \
                --method "$method" \
                --model-name-or-path "$model" \
                --split test \
                --num-episodes-per-task 500 \
                --support-size 1 \
                --max-items-per-subset 9999 \
                --seed "$seed" \
                --output-dir "${OUTPUT_DIR}/simpletom/${model_short}/${method}/seed_${seed}"
        done

        # BSTTT methods (AR and NTL)
        for loss in "${BSTTT_LOSSES_SIMPLETOM[@]}"; do
            echo "=== SimpleToM | ${model_short} | bsttt-${loss} | seed=${seed} ==="
            python -m bsttt.eval.eval_simpletom \
                --method bsttt \
                --bsttt-loss "$loss" \
                --model-name-or-path "$model" \
                --split test \
                --num-episodes-per-task 500 \
                --support-size 1 \
                --max-items-per-subset 9999 \
                --adapt-steps 3 \
                --lora-rank 8 \
                --lora-alpha 16 \
                --lr 1e-4 \
                --seed "$seed" \
                --output-dir "${OUTPUT_DIR}/simpletom/${model_short}/bsttt_${loss}/seed_${seed}"
        done

        # MS-Reminder + BSTTT (composed)
        echo "=== SimpleToM | ${model_short} | ms_reminder_bsttt | seed=${seed} ==="
        python -m bsttt.eval.eval_simpletom \
            --method ms_reminder_bsttt \
            --bsttt-loss action_reconstruction \
            --model-name-or-path "$model" \
            --split test \
            --num-episodes-per-task 500 \
            --support-size 1 \
            --max-items-per-subset 9999 \
            --adapt-steps 3 \
            --lora-rank 8 \
            --lora-alpha 16 \
            --lr 1e-4 \
            --seed "$seed" \
            --output-dir "${OUTPUT_DIR}/simpletom/${model_short}/ms_reminder_bsttt/seed_${seed}"
    done
done

# ---- Hi-ToM and OpenToM ----
EXTERNAL_DATASETS=(hitom opentom)
EXTERNAL_METHODS=(frozen cot scratchpad simtom symbolictom bsttt)

for model in "${MODELS[@]}"; do
    model_short=$(echo "$model" | sed 's|.*/||')
    for dataset in "${EXTERNAL_DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for method in "${EXTERNAL_METHODS[@]}"; do
                echo "=== ${dataset} | ${model_short} | ${method} | seed=${seed} ==="
                python -m bsttt.eval.eval_external_tom \
                    --dataset "$dataset" \
                    --method "$method" \
                    --model-name-or-path "$model" \
                    --seed "$seed" \
                    --output-dir "${OUTPUT_DIR}/${dataset}/${model_short}/${method}/seed_${seed}"
            done
        done
    done
done

# ---- DynToM ----
DYNTOM_METHODS=(frozen cot scratchpad bsttt hierarchical_ttt)

for model in "${MODELS[@]}"; do
    model_short=$(echo "$model" | sed 's|.*/||')
    for seed in "${SEEDS[@]}"; do
        for method in "${DYNTOM_METHODS[@]}"; do
            echo "=== DynToM | ${model_short} | ${method} | seed=${seed} ==="
            python -m bsttt.eval.eval_dyntom \
                --method "$method" \
                --model-name-or-path "$model" \
                --seed "$seed" \
                --output-dir "${OUTPUT_DIR}/dyntom/${model_short}/${method}/seed_${seed}"
        done
    done
done

echo ""
echo "============================================"
echo " All reproduction runs complete."
echo " Results saved to: ${OUTPUT_DIR}/"
echo "============================================"
