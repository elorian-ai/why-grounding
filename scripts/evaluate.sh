#!/usr/bin/env bash
# Evaluate one checkpoint on one GPU: 9 spatial/perception benchmarks at temperature 1 per decode seed (gr_sampled),
# then CountQA and FSC-147 greedily (gr_greedy) -> $GR_RESULTS/eval/<run>/{sampled/seed<S>,greedy}/.
#   bash scripts/evaluate.sh $GR_RUNS/sft/grounded_s17 sft/grounded_s17
#   LIMIT=8 bash scripts/evaluate.sh Qwen/Qwen3-VL-8B-Instruct smoke
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
CKPT=${1:?usage: evaluate.sh <checkpoint dir or hub id> <run-name>}
RUN=${2:?usage: evaluate.sh <checkpoint dir or hub id> <run-name>}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS-1000 2000 3000}   # SEEDS= skips gr_sampled
GREEDY=${GREEDY:-1}             # GREEDY=0 skips gr_greedy
LIMIT=${LIMIT:-}                # first N items per benchmark
export GR_DATA=${GR_DATA:-$REPO/data}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
OUT=${GR_RESULTS:-$REPO/results}/eval/$RUN
run() { "$PYTHON" -m grounded_reasoning.evaluate --model "$CKPT" ${LIMIT:+--limit "$LIMIT"} "$@"; }
for SEED in $SEEDS; do run --tasks gr_sampled --seed "$SEED" --out "$OUT/sampled/seed$SEED"; done
if [[ $GREEDY == 1 ]]; then run --tasks gr_greedy --out "$OUT/greedy"; fi
