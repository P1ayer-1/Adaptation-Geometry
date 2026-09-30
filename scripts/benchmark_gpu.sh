#!/usr/bin/env bash
# One-hour GPU benchmark before committing to a card (e.g. A40 vs A100 vs H100).
#   bash scripts/benchmark_gpu.sh <price $/h> [minutes=45] [base config] [task config]
# Trains one Stage-0-sized run of the largest Stage-0 base (Llama-3.2-3B; accept its licence
# on the hub and `huggingface-cli login` first), verifies ΔW, evaluates, and prints cost per
# run and a Stage-0 projection at that price. Run it once per GPU type and compare.
set -euo pipefail
PRICE="${1:?usage: benchmark_gpu.sh <price per hour> [minutes] [base config] [task config]}"
MIN="${2:-45}"
BASE="${3:-configs/bases/stage0_B_llama3.2-3b.yaml}"
TASK="${4:-configs/tasks/v2/T4_json.yaml}"
nvidia-smi --query-gpu=name,memory.total --format=csv
uag benchmark --price "$PRICE" --minutes "$MIN" --base-config "$BASE" --task-config "$TASK" \
  --train-config configs/train/gpu_24_48gb.yaml | tee "results/benchmark_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' /' '__').txt"
