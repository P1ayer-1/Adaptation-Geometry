#!/usr/bin/env bash
# One rental session: (1) the smart-eyes test on all GPUs, then (2) the shared-adapter test
# (code width 64 and 128 in parallel on GPUs 0 and 1). Resumable; rerun after any crash.
#   bash scripts/rental_eyes_then_shared.sh [n_gpus] [eyes_jobs_per_gpu]
# Results: results/eyes_summary.txt, results/shared_adapter_d64/shared_adapter.txt, ..._d128/...
set -euo pipefail
N="${1:-$(nvidia-smi -L | wc -l)}"
bash scripts/eyes_test.sh "$N" "${2:-1}"
run_shared() {  # run_shared <gpu> <config>
  for attempt in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$1 uag shared-adapter -e "$2" && return 0
    echo "shared adapter $2 failed (attempt $attempt), retrying in 30 s"; sleep 30
  done
  return 1
}
if [ "$N" -ge 2 ]; then
  run_shared 0 configs/experiments/shared_adapter_d64.yaml > results/shared_adapter_d64.log 2>&1 &
  P1=$!
  run_shared 1 configs/experiments/shared_adapter_d128.yaml > results/shared_adapter_d128.log 2>&1 &
  P2=$!
  wait $P1; wait $P2
else
  run_shared 0 configs/experiments/shared_adapter_d64.yaml > results/shared_adapter_d64.log 2>&1
  run_shared 0 configs/experiments/shared_adapter_d128.yaml > results/shared_adapter_d128.log 2>&1
fi
echo "== smart eyes"; cat results/eyes_summary.txt
for d in 64 128; do echo "== shared adapter d$d"; cat results/shared_adapter_d$d/shared_adapter.txt; done
