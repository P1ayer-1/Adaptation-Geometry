#!/usr/bin/env bash
# Run an experiment on N GPUs of one machine (e.g. 4x RTX 5090): tasks are dealt round-robin
# to GPUs, one `uag` process per GPU. Same steps and outputs as dry_run.sh; resumable.
#   bash scripts/multi_gpu.sh <experiment config> <number of GPUs>
# Per-GPU logs: results/<config>.gpu<i>.log
set -euo pipefail
CFG="${1:?usage: multi_gpu.sh <experiment config> <n_gpus>}"
N="${2:?usage: multi_gpu.sh <experiment config> <n_gpus>}"
NAME="${CFG##*/}"
uag validate -e "$CFG" > /dev/null
uag make-data -e "$CFG"                       # once, before the parallel steps
mapfile -t CHUNKS < <(uag validate -e "$CFG" | python3 -c "
import json, sys
tasks = json.load(sys.stdin)['tasks']; n = int(sys.argv[1])
for i in range(n): print(' '.join(tasks[i::n]))" "$N")

# Background jobs of a script ignore Ctrl-C, so forward it: stop every GPU process, then exit.
PIDS=()
trap 'echo "interrupted: stopping GPU processes"; kill "${PIDS[@]}" 2>/dev/null; exit 130' INT TERM

parallel_step() {  # parallel_step <uag args...>: run once per GPU on its task chunk, wait, fail if any failed
  local i
  for i in "${!CHUNKS[@]}"; do
    [ -z "${CHUNKS[$i]}" ] && continue
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$i uag "$@" --task ${CHUNKS[$i]} >> "results/${NAME}.gpu$i.log" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p" || { echo "a GPU process failed; see results/${NAME}.gpu*.log"; exit 1; }; done
  PIDS=()
}

echo "tasks per GPU:"; for i in "${!CHUNKS[@]}"; do echo "  GPU $i: ${CHUNKS[$i]}"; done
parallel_step train -e "$CFG"
parallel_step eval -e "$CFG" --stage base
parallel_step eval -e "$CFG" --stage direct
CUDA_VISIBLE_DEVICES=0 uag fit-maps -e "$CFG"   # fits on all training tasks; one process
parallel_step eval -e "$CFG" --stage transfer
uag graded-transfer -e "$CFG"
uag analyze -e "$CFG"
uag dry-run-report -e "$CFG" | tee "results/${NAME}.dry_run_report.txt"
