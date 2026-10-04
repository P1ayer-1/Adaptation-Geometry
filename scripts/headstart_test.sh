#!/usr/bin/env bash
# Zero-default shared adapter + head-start test on a 2-GPU rental. Resumable; rerun after a crash.
#   bash scripts/headstart_test.sh            -> results/headstart_summary.txt
# GPU placement (defaults: shared on 0, LoRA on 1, head-start jobs round-robin), e.g. when one GPU
# is partly occupied:  SHARED_GPU=1 LORA_GPU=1 JOB_GPUS="1 1 1 1" bash scripts/headstart_test.sh
set -euo pipefail
E=configs/experiments/shared_adapter_zero_d64_gpu.yaml
G="$(nvidia-smi -L | wc -l)"
SHARED_GPU="${SHARED_GPU:-0}"
LORA_GPU="${LORA_GPU:-$((G > 1 ? 1 : 0))}"
read -r -a JG <<< "${JOB_GPUS:-$(seq -s ' ' 0 $((G - 1)))}"
retry() { for a in 1 2 3; do "$@" && return 0; echo "failed (attempt $a): $*"; sleep 30; done; return 1; }
uag make-data -e "$E"
# GPU 0: phase 1 (connectors), held-out cores (full data), zero-default transfer evaluation
( export CUDA_VISIBLE_DEVICES=$SHARED_GPU; retry uag shared-adapter -e "$E" --phase all ) > results/headstart_shared.log 2>&1 &
P0=$!
# GPU 1 meanwhile: ordinary LoRA from scratch on N = 32 / 128 / 512 examples (references)
( export CUDA_VISIBLE_DEVICES=$LORA_GPU; for n in 32 128 512; do
    c=configs/experiments/headstart/headstart_lora_n$n.yaml
    retry uag train -e "$c" && retry uag eval -e "$c" --stage direct
  done ) > results/headstart_lora.log 2>&1 &
P1=$!
wait $P0; wait $P1
# Head start: 4 (task, target) jobs, 2 per GPU
PIDS=(); i=0
for t in T4_json T8_arithmetic; do for b in dev_qwen2.5-0.5b dev_llama3.2-1b; do
  ( export CUDA_VISIBLE_DEVICES=${JG[$((i % ${#JG[@]}))]}; retry uag shared-adapter -e "$E" --phase headstart --task "$t" --base "$b" ) \
    > "results/headstart_$t.$b.log" 2>&1 &
  PIDS+=($!); i=$((i + 1))
done; done
for p in "${PIDS[@]}"; do wait "$p"; done
uag shared-adapter -e "$E" --phase headstart > /dev/null   # merge the four parts
python3 scripts/headstart_summary.py | tee results/headstart_summary.txt
