#!/usr/bin/env bash
# Stage 0, step 3: evaluate predicted updates on untouched target test sets, then write
# results/<experiment>_decision.md from the frozen analysis script.
set -euo pipefail
CFG="${1:-configs/experiments/stage0.yaml}"
uag eval -e "$CFG" --stage transfer
uag analyze -e "$CFG"
