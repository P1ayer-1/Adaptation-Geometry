#!/usr/bin/env bash
# Crossed task x initialisation test (universal-subspace note). Resumable: training is idempotent.
#   nohup bash scripts/keep_running.sh scripts/crossed_test.sh results/crossed_test.log "crossed_test: done" &
set -e
cd "$(dirname "$0")/.."
UAG=~/micromamba/envs/lora_research/bin/uag
for arm in shared indep; do
  $UAG validate -e configs/experiments/dev_3080_crossed_$arm.yaml --allow-unpinned
  $UAG make-data -e configs/experiments/dev_3080_crossed_$arm.yaml
done
$UAG train -e configs/experiments/dev_3080_crossed_indep.yaml
$UAG train -e configs/experiments/dev_3080_crossed_shared.yaml
echo "crossed_test: done"
