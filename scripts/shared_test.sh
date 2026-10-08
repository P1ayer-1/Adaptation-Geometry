#!/usr/bin/env bash
# Shared-adapter test alone: width 64 on GPU 0 and 128 on GPU 1 (or both on GPU 0 with one GPU).
# Resumable; retries failures. Logs: results/shared_adapter_d{64,128}.log
set -euo pipefail
G="$(nvidia-smi -L | wc -l)"
run_shared() {  # run_shared <gpu> <width>
  for attempt in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$1 uag shared-adapter -e "configs/experiments/shared_adapter_d$2.yaml" && return 0
    echo "shared adapter d$2 failed (attempt $attempt), retrying in 30 s"; sleep 30
  done
  return 1
}
run_shared 0 64 > results/shared_adapter_d64.log 2>&1 &
P1=$!
run_shared $((G > 1 ? 1 : 0)) 128 > results/shared_adapter_d128.log 2>&1 &
P2=$!
wait $P1; wait $P2
for d in 64 128; do echo "== shared adapter d$d"; cat "results/shared_adapter_d$d/shared_adapter.txt"; done
