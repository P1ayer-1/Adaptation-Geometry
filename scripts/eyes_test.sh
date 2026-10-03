#!/usr/bin/env bash
# Smart-eyes test on a rented machine: 5 variants x 2 dev bases x {T1, T2, T3}. The 10
# (variant, base) jobs are dealt round-robin to the GPUs; each trains and evaluates its runs.
#   bash scripts/eyes_test.sh [n_gpus] [jobs_per_gpu]   (defaults: all GPUs, 1) -> results/eyes_summary.txt
# These dev models use a small fraction of a 40-80 GB GPU, so 3 jobs per GPU is ~2.5x faster.
# Resumable: rerun after any crash. Per-slot logs: results/eyes_test.slot<i>.log
set -euo pipefail
G="${1:-$(nvidia-smi -L | wc -l)}"
K="${2:-1}"
N=$((G * K))
EXPS=(eyes_random16 eyes_whitened16 eyes_pca16 eyes_random64 eyes_trainable16)
BASES=(dev_qwen2.5-0.5b dev_llama3.2-1b)
uag make-data -e configs/experiments/eyes/eyes_random16.yaml
python3 -c "from uag.eyes import calibration_texts; print(len(calibration_texts('wikitext')), 'calibration texts')"
JOBS=(); for e in "${EXPS[@]}"; do for b in "${BASES[@]}"; do JOBS+=("$e $b"); done; done
PIDS=()
trap 'echo "interrupted: stopping GPU jobs"; kill "${PIDS[@]}" 2>/dev/null; exit 130' INT TERM
for slot in $(seq 0 $((N - 1))); do
  g=$((slot % G))
  (
    for ((j = slot; j < ${#JOBS[@]}; j += N)); do
      read -r e b <<< "${JOBS[$j]}"
      cfg=configs/experiments/eyes/$e.yaml
      for attempt in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$g uag train -e "$cfg" --base "$b" && \
        CUDA_VISIBLE_DEVICES=$g uag eval -e "$cfg" --stage direct --base "$b" && break
        echo "job $e/$b failed (attempt $attempt), retrying in 30 s"; sleep 30
      done
    done
  ) >> "results/eyes_test.slot$slot.log" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
python3 scripts/eyes_summary.py | tee results/eyes_summary.txt
