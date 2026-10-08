#!/usr/bin/env bash
# Zero-default shared adapter on a local GPU (resumable per phase; retries CUDA failures).
#   nohup bash scripts/keep_running.sh scripts/shared_zero_test.sh results/shared_zero_test.log "shared_zero_test: done" &
set -euo pipefail
for attempt in 1 2 3 4 5; do
  rc=0; uag shared-adapter -e configs/experiments/shared_adapter_zero_d64.yaml || rc=$?
  [ "$rc" -eq 0 ] && { echo "shared_zero_test: done"; exit 0; }
  [ "$rc" -ne 3 ] && exit "$rc"
  echo "CUDA failure, retry $attempt/5 in 30 s" >&2; sleep 30
done
exit 3
