#!/usr/bin/env bash
# Full fine-tune of one SFT arm with LLaMA-Factory on all GPUs of one node (configs/sft.yaml, global batch 64).
#   bash scripts/train_sft.sh <arm> [seed=17] [key=value ...]   -> $GR_RUNS/sft/<arm>_s<seed>/
set -euo pipefail
ARMS="text grounded appended grounded_box grounded_rand05 grounded_rand20 grounded_rand50 grounded_rand100
      text_unfiltered grounded_unfiltered brickvision_direct brickvision_text brickvision_grounded"
ARM=${1:?usage: train_sft.sh <arm> [seed] [key=value ...]}
SEED=${2:-17}
shift $(($# < 2 ? $# : 2))
[[ " $(echo $ARMS) " == *" $ARM "* ]] || { echo "unknown arm '$ARM'; arms: $(echo $ARMS)" >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
export GR_DATA=${GR_DATA:-$REPO/data} GR_RUNS=${GR_RUNS:-$REPO/runs}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ $ARM == brickvision_* ]]; then
  SET=$GR_DATA/brickvision/sft/${ARM#brickvision_}.json
  ARGS=(model_name_or_path=Qwen/Qwen3.5-9B template=qwen3_5_nothink)
else
  SET=$GR_DATA/sft/$ARM.json
  ARGS=()
fi
DATASET_DIR=$(dirname "$SET")/lf/$(basename "$SET" .json)          # LLaMA-Factory dataset dir, built once
[[ -s $DATASET_DIR/train.json ]] || "$PYTHON" -m grounded_reasoning.data llamafactory "$SET"

GLOBAL_BATCH=64 PER_DEVICE=$(sed -n "s/^per_device_train_batch_size: *//p" "$REPO/configs/sft.yaml")
NGPUS=${NPROC_PER_NODE:-$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')}
(( NGPUS >= 1 && GLOBAL_BATCH % (NGPUS * PER_DEVICE) == 0 )) || { echo "need a GPU count dividing $GLOBAL_BATCH" >&2; exit 2; }
RUN=${ARM}_s${SEED}
cd "$REPO"   # deepspeed config path is relative to the repo root
# rank 0 tokenizes while the other ranks wait in a collective, which outlasts the default NCCL heartbeat
export NPROC_PER_NODE=$NGPUS FORCE_TORCHRUN=1 TOKENIZERS_PARALLELISM=false TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
llamafactory-cli train configs/sft.yaml "${ARGS[@]}" dataset_dir="$DATASET_DIR" seed="$SEED" \
  gradient_accumulation_steps=$((GLOBAL_BATCH / (NGPUS * PER_DEVICE))) output_dir="$GR_RUNS/sft/$RUN" run_name="$RUN" "$@"
