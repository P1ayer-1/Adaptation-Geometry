#!/usr/bin/env bash
# End-to-end pipeline validation on tiny CPU bases (2 bases × 2 tasks × 2 seeds).
set -euo pipefail
CFG=configs/experiments/smoke.yaml
bash scripts/stage0_train_grid.sh "$CFG"
bash scripts/stage0_fit_maps.sh "$CFG"
bash scripts/stage0_eval.sh "$CFG"
