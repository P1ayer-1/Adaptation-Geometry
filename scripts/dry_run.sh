#!/usr/bin/env bash
# Real-model dry run on a home GPU (e.g. RTX 3080 8 GB) before renting.
# Prereqs: CUDA PyTorch installed, `huggingface-cli login`, Llama-3.2 licence accepted.
set -euo pipefail
CFG="${1:-configs/experiments/dev_3080_v2.yaml}"
uag validate -e "$CFG"
uag train -e "$CFG"
uag eval -e "$CFG" --stage base
uag eval -e "$CFG" --stage direct
uag fit-maps -e "$CFG"
uag eval -e "$CFG" --stage transfer
uag analyze -e "$CFG"
uag dry-run-report -e "$CFG" | tee "results/${CFG##*/}.dry_run_report.txt"
