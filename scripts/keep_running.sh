#!/usr/bin/env bash
# Rerun a resumable script until its log shows a done marker (max 20 runs). For flaky local GPUs.
#   nohup bash scripts/keep_running.sh scripts/scaling_test.sh results/scaling_test.log "scaling_test: done" &
SCRIPT="$1"; LOG="$2"; DONE="$3"
cd "$(dirname "$0")/.."
export PATH=~/micromamba/envs/lora_research/bin:$PATH
unset PYTORCH_CUDA_ALLOC_CONF   # expandable_segments breaks CUDA memory mapping under WSL
for i in $(seq 1 20); do
  grep -q "$DONE" "$LOG" 2>/dev/null && { echo "keep_running: done" >> "$LOG"; exit 0; }
  echo "=== keep_running: start $i $(date) ===" >> "$LOG"
  bash "$SCRIPT" >> "$LOG" 2>&1 || true
  sleep 30
done
echo "keep_running: gave up after 20 runs" >> "$LOG"
