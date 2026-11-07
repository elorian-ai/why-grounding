#!/usr/bin/env bash
# One probe task end to end on one 8-GPU node: data -> SFT sets -> fine-tune + eval per run -> summary.json.
#   bash scripts/probes.sh wire_tracing|prompted_counting|arrow_following
#   STAGES="eval summarize" RUN_NAMES="grounded grounded_labeled" bash scripts/probes.sh arrow_following
set -euo pipefail
TASK=${1:?usage: probes.sh wire_tracing|prompted_counting|arrow_following}
[[ $TASK =~ ^(wire_tracing|prompted_counting|arrow_following)$ ]] || { echo "unknown task $TASK" >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PYTHON=${PYTHON:-python}
STAGES=" ${STAGES:-generate sft train eval summarize} "
RUN_NAMES=${RUN_NAMES:-direct text grounded direct_labeled text_labeled grounded_labeled}
NPROC=${NPROC:-8}   # 8 GPUs x batch 1 x grad-accum 8 = global batch 64
export GR_DATA=${GR_DATA:-$REPO/data} GR_RUNS=${GR_RUNS:-$REPO/runs} GR_RESULTS=${GR_RESULTS:-$REPO/results}
M=grounded_reasoning.probes.$TASK

[[ $STAGES == *" generate "* ]] && "$PYTHON" -m "$M" generate
[[ $STAGES == *" sft "* ]] && "$PYTHON" -m "$M" sft
for RUN in $RUN_NAMES; do
  FORMAT=${RUN%_labeled} LABELED=()
  [[ $RUN == *_labeled ]] && LABELED=(--labeled)
  CKPT=$GR_RUNS/probes/${TASK}_$RUN
  if [[ $STAGES == *" train "* ]]; then
    "$PYTHON" -m torch.distributed.run --nproc_per_node "$NPROC" -m grounded_reasoning.probes.finetune \
      --data_path "$GR_DATA/probes/$TASK/sft/$RUN.json" --output_dir "$CKPT"
    rm -rf "$CKPT"/checkpoint-*   # intermediate saves; final weights are in $CKPT
  fi
  if [[ $STAGES == *" eval "* ]]; then
    "$PYTHON" -m "$M" eval --format "$FORMAT" "${LABELED[@]}"
  fi
done
[[ $STAGES == *" summarize "* ]] && "$PYTHON" -m "$M" summarize
exit 0
