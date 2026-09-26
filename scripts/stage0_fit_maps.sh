#!/usr/bin/env bash
# Stage 0, step 2: freeze the gate, fit source→target maps on training transformations
# only, and predict held-out target updates.
set -euo pipefail
CFG="${1:-configs/experiments/stage0.yaml}"
uag declare-gate -e "$CFG"
uag fit-maps -e "$CFG"
