#!/usr/bin/env bash
# Local (8 GB GPU) test of the frozen-A design against the trainable-A v2 dry run.
#   1. dev_3080_frozenA: train 24 frozen-A runs, evaluate, fit maps on T3/T7/T9/T10, predict T4/T8.
#   2. dev_a100_v2_cmp: the same maps / baselines on the dry run's trainable-A adapters
#      (needs artifacts/dev_a100_v2 from the dry run; runs and base/direct evals are reused).
#   3. dev_3080_patience: the T1/T2 runs that never took off, retrained with patient stopping.
# Resumable. Logs: results/frozen_a_test.log
set -euo pipefail
copy_evals() {  # copy_evals <from experiment> <to experiment> <glob>...: reuse finished evaluations
  local from="results/raw/$1" to="results/raw/$2"; shift 2
  mkdir -p "$to"
  for g in "$@"; do for f in "$from"/$g; do [ -e "$f" ] && cp -n "$f" "$to/"; done; done
}
TASKS="T3_paraphrase T7_python T9_clinical T10_format T4_json T8_arithmetic"

# 1. frozen A (base-model evaluations are identical to the dry run's, so they are reused)
F=configs/experiments/dev_3080_frozenA.yaml
for t in $TASKS; do copy_evals dev_a100_v2 dev_3080_frozenA "base*__${t}__v1.*"; done
uag train -e "$F"
uag eval -e "$F" --stage base
uag eval -e "$F" --stage direct
uag fit-maps -e "$F"
uag eval -e "$F" --stage transfer
uag analyze -e "$F"
uag dry-run-report -e "$F" > results/dev_3080_frozenA.yaml.dry_run_report.txt

# 2. trainable-A reference on the same tasks / split / baselines
C=configs/experiments/dev_a100_v2_cmp.yaml
mkdir -p artifacts/dev_a100_v2_cmp
[ -e artifacts/dev_a100_v2_cmp/runs ] || ln -s ../dev_a100_v2/runs artifacts/dev_a100_v2_cmp/runs
for t in $TASKS; do copy_evals dev_a100_v2 dev_a100_v2_cmp "base*__${t}__v1.*" "direct__*__${t}__*"; done
uag fit-maps -e "$C"
uag eval -e "$C" --stage transfer
uag analyze -e "$C"

# 3. patience: only the runs that failed to take off in the dry run
P=configs/experiments/dev_3080_patience.yaml
uag make-data -e "$P"
uag train -e "$P" --base dev_qwen2.5-0.5b --task T1_hidden_rule --seed 0
uag train -e "$P" --base dev_llama3.2-1b --task T1_hidden_rule --seed 0
uag train -e "$P" --base dev_llama3.2-1b --task T2_nli --seed 0
uag train -e "$P" --base dev_qwen2.5-0.5b --task T2_nli --seed 1
uag learning-curves -e "$P"
echo "frozen_a_test: done"
