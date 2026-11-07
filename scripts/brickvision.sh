#!/usr/bin/env bash
# BrickVision (Table 4), one stage per call:  bash scripts/brickvision.sh generate|restyle|api|sft-data|eval|table
#   generate  synthetic bench (CPU)          restyle   photo restyle + judge (GEMINI_API_KEY)
#   api       API model rows (API keys)      sft-data  training scenes + SFT sets (CPU); then train_sft.sh brickvision_*
#   eval      Qwen3.5-9B base + 3 fine-tunes (vLLM, 1 GPU)                       table     Table 4
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PYTHON=${PYTHON:-python}
DATA=${GR_DATA:-$REPO/data}/brickvision
API_ROWS=${API_ROWS:-gpt-6-astra claude-opus-5.5 gpt-5.6-sol gemini-3.8-flash claude-opus-5 qwen3.8-max}
EVAL_RUNS=${EVAL_RUNS:-qwen3.5-9b brickvision_direct_s17 brickvision_text_s17 brickvision_grounded_s17}
declare -A BENCH_SEED=([crop_id]=66002 [counting]=61002 [duplicate]=64004 [unpaired]=56002)
declare -A SFT_SEED=([crop_id]=760000 [counting]=770000 [duplicate]=740000 [unpaired]=750000)   # + 1000 * shard
pids=()
wait_all() { local pid; for pid in "${pids[@]}"; do wait "$pid"; done; }   # fails if any generator failed

case "${1:?usage: brickvision.sh generate|restyle|api|sft-data|eval|table}" in
  generate)
    for task in "${!BENCH_SEED[@]}"; do
      "$PYTHON" -m brickvision.generate "$task" --seed "${BENCH_SEED[$task]}" --out "$DATA/synthetic/$task" & pids+=($!)
    done
    wait_all ;;
  restyle) "$PYTHON" -m brickvision.restyle ;;
  api) for row in $API_ROWS; do "$PYTHON" -m brickvision.models api "$row"; done ;;
  sft-data)   # 10 shards x 4 tasks, then 2,500 scenes per task -> sft/{direct,text,grounded}.json
    for task in "${!SFT_SEED[@]}"; do
      for shard in {0..9}; do
        seed=$((SFT_SEED[$task] + 1000 * shard)) extra=()
        [[ $task == counting ]] && extra=(--n-items 252)   # all 3 x 84 scenes; the bench keeps 250
        "$PYTHON" -m brickvision.generate "$task" --seed "$seed" --out "$DATA/sft/scenes/${task}_seed$seed" "${extra[@]}" & pids+=($!)
      done
    done
    wait_all
    "$PYTHON" -m brickvision.build_sft ;;
  eval)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
    for run in $EVAL_RUNS; do "$PYTHON" -m brickvision.models qwen "$run"; done ;;
  table) "$PYTHON" -m brickvision.evaluate tables ;;
  *) echo "unknown stage: $1" >&2; exit 2 ;;
esac
