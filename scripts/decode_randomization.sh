#!/usr/bin/env bash
# Decode-time coordinate randomization (Table 1, Figure 3 usage curve): one (arm, bench, seed) unit per GPU, finished
# units are skipped on a rerun, then the report.   bash scripts/decode_randomization.sh [arm ...]
# Env: NUM_GPUS (all visible), SEEDS (1000..20000), BENCHES (all four).
set -euo pipefail
PYTHON=${PYTHON:-python}
# vLLM's FlashInfer sampler JIT-builds with ninja from the env's bin/ (what `uv run` puts on PATH)
[[ $PYTHON == */* ]] && export PATH="$(dirname "$PYTHON"):$PATH"
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
SEEDS=${SEEDS:-$(seq 1000 1000 20000)}
BENCHES=${BENCHES:-countqa multihopspatial spatialmqa vstar}
export GR_RESULTS=${GR_RESULTS:-$PWD/results}
OUT=$GR_RESULTS/decode_randomization
TABLE_ARMS=(treevgr_ci treevgr_rl ivgr deepeyes appended grounded)
CURVE_ARMS=(grounded_rand05 grounded_rand20 grounded_rand50 grounded_rand100)
ARMS=("$@")
[[ ${#ARMS[@]} -gt 0 ]] || ARMS=("${TABLE_ARMS[@]}" "${CURVE_ARMS[@]}")

units=()  # "arm bench seed", CountQA first (the longest units)
for bench in $BENCHES; do
  for arm in "${ARMS[@]}"; do
    [[ $arm == grounded_rand* && $bench != multihopspatial ]] && continue
    for seed in $SEEDS; do units+=("$arm $bench $seed"); done
  done
done

# Download the datasets and cache the CountQA images once, before parallel jobs read them.
"$PYTHON" -m grounded_reasoning.randomize prepare
mkdir -p "$OUT/logs"
CLAIMS=$(mktemp -d)  # one directory per unit started on this node
trap 'rm -rf "$CLAIMS"' EXIT

worker() {  # GPU index -> runs unclaimed units until none are left
  local arm bench seed
  for unit in "${units[@]}"; do
    read -r arm bench seed <<< "$unit"
    [[ -f $OUT/rollouts/${bench}_${arm}_randomize_s${seed}.json ]] && continue  # written last by a finished unit
    mkdir "$CLAIMS/${bench}_${arm}_s${seed}" 2> /dev/null || continue
    echo "[gpu $1] $unit"
    # vLLM points Inductor at <VLLM_CACHE_ROOT>/torch_compile_cache/<model hash>/; one root per GPU keeps workers
    # compiling the same model from racing on its kernel files
    CUDA_VISIBLE_DEVICES=$1 TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor_gpu$1 VLLM_CACHE_ROOT=${TMPDIR:-/tmp}/vllm_gpu$1 \
      VLLM_LOGGING_LEVEL=WARNING \
      "$PYTHON" -m grounded_reasoning.randomize run --arm "$arm" --bench "$bench" --seed "$seed" \
      > "$OUT/logs/${bench}_${arm}_s${seed}.log" 2>&1 || { echo "[gpu $1] FAILED $unit" >&2 && touch "$CLAIMS/failed"; }
  done
}

for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do worker "$gpu" & done
wait
[[ ! -e $CLAIMS/failed ]] || { echo "some units failed, see $OUT/logs; rerun to retry them" >&2 && exit 1; }
if [[ $# -eq 0 ]]; then
  "$PYTHON" -m grounded_reasoning.randomize report
else
  echo "done: $*. Once every arm is decoded: $PYTHON -m grounded_reasoning.randomize report"
fi
