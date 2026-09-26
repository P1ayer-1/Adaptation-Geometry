#!/usr/bin/env bash
# Stage 0, step 1: data, the LoRA grid (with ΔW extraction + verification), raw-base and
# direct-LoRA evaluations. Idempotent: finished runs/evals are reused.
set -euo pipefail
CFG="${1:-configs/experiments/stage0.yaml}"
uag validate -e "$CFG"
uag make-data -e "$CFG"
uag train -e "$CFG"
uag eval -e "$CFG" --stage base
uag eval -e "$CFG" --stage direct
