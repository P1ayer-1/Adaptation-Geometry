#!/usr/bin/env bash
# Task-count scaling test (frozen A, local GPU). Reuses dev_3080_frozenA's seed-0 runs, trains
# T1/T2/T5/T6, fits nested maps (2/4/6/8 training tasks), scores them with graded transfer and
# writes the coverage curve. Resumable; retries transient CUDA failures. Log: results/scaling_test.log
set -euo pipefail
uag() {
  local i rc
  for i in 1 2 3 4 5; do
    rc=0; command uag "$@" || rc=$?
    [ "$rc" -eq 0 ] && return 0
    [ "$rc" -ne 3 ] && return "$rc"
    echo "uag $1: CUDA failure, retry $i/5 in 30 s" >&2; sleep 30
  done
  return 3
}
E=configs/experiments/dev_3080_frozenA_scale.yaml
mkdir -p artifacts/dev_3080_frozenA_scale/runs
for b in dev_qwen2.5-0.5b dev_llama3.2-1b; do
  for t in T3_paraphrase T7_python T9_clinical T10_format T4_json T8_arithmetic; do
    r=${b}_${t}_seed0_r16
    [ -e "artifacts/dev_3080_frozenA_scale/runs/$r" ] || ln -s "../../dev_3080_frozenA/runs/$r" "artifacts/dev_3080_frozenA_scale/runs/$r"
  done
done
uag make-data -e "$E"
uag train -e "$E" --task T1_hidden_rule T2_nli T5_concise T6_verbose
uag fit-maps -e "$E"
uag graded-transfer -e "$E" --n-examples 100
python scripts/coverage_scaling.py "$E" --order T3_paraphrase T7_python T9_clinical T10_format T1_hidden_rule T2_nli T5_concise T6_verbose --heldout T4_json T8_arithmetic
echo "scaling_test: done"
