#!/usr/bin/env bash
# RL answer judge: Qwen3.5-9B behind vLLM's OpenAI server on one GPU (envs/gpu). Prints JUDGE_URL=... when ready
# (also to $JUDGE_URL_FILE if set) and serves until killed.   CUDA_VISIBLE_DEVICES=0 bash scripts/serve_judge.sh
set -euo pipefail
PORT=${PORT:-8100}
export TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
"${VLLM:-vllm}" serve Qwen/Qwen3.5-9B --host 0.0.0.0 --port "$PORT" --served-model-name Qwen/Qwen3.5-9B \
  --gpu-memory-utilization 0.85 --max-model-len 4096 &
SERVER=$!
trap 'kill "$SERVER" 2> /dev/null || true' EXIT
until curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 "$SERVER" || exit 1
  sleep 10
done
URL="http://$(hostname -I | awk '{print $1}'):$PORT/v1"
echo "JUDGE_URL=$URL"
if [ -n "${JUDGE_URL_FILE:-}" ]; then echo "$URL" > "$JUDGE_URL_FILE"; fi
wait "$SERVER"
