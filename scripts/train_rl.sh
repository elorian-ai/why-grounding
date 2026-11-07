#!/usr/bin/env bash
# GRPO (verl) of one SFT init on one 8-GPU node inside the envs/rl image; the merged HF weights go to
# $GR_RUNS/rl/<arm>_s<seed>/. Needs the answer judge at JUDGE_URL (scripts/serve_judge.sh). Extra args: hydra overrides.
#   JUDGE_URL=http://<judge host>:8100/v1 bash scripts/train_rl.sh grounded 17
#   arm: text (grounding reward weight 0) | grounded;  seed: SFT init $GR_RUNS/sft/<arm>_s<seed> and data-order seed
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
ARM=${1:?usage: train_rl.sh text|grounded <seed> [hydra overrides ...]}
SEED=${2:?usage: train_rl.sh text|grounded <seed> [hydra overrides ...]}
shift 2
PYTHON=${PYTHON:-python}
export GR_DATA=${GR_DATA:-$REPO/data} GR_RUNS=${GR_RUNS:-$REPO/runs}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"  # the reward function imports grounded_reasoning
MODEL=${MODEL:-$GR_RUNS/sft/${ARM}_s${SEED}}
RUN=${ARM}_s${SEED}
OUT=$GR_RUNS/rl/$RUN
DATA=$GR_DATA/rl/train
case "$ARM" in
  text) GROUND_WEIGHT=0.0 ;;
  grounded) GROUND_WEIGHT=1.0 ;;
  *) echo "arm must be text or grounded" >&2; exit 2 ;;
esac
# seed 17: torch's default generator seed, the data order of the paper's seed-17 runs
if [ "$SEED" = 17 ]; then DATA_SEED=67280421310721; else DATA_SEED=$SEED; fi
JUDGE_URL=${JUDGE_URL:?set JUDGE_URL to the judge endpoint printed by scripts/serve_judge.sh}
export JUDGE_URL

ls "$MODEL"/*.safetensors > /dev/null || { echo "no SFT weights in $MODEL" >&2; exit 2; }
[ -s "$GR_DATA/rl/treevgr_rl.parquet" ] || "$PYTHON" -m grounded_reasoning.rl build-data treevgr
[ -s "$DATA/train.parquet" ] || "$PYTHON" -m grounded_reasoning.rl build-data verl  # absolute image paths: built here
for _ in $(seq 180); do  # wait up to 30 min for the judge
  curl -sf --max-time 5 "$JUDGE_URL/models" > /dev/null && break
  sleep 10
done
curl -sf --max-time 5 "$JUDGE_URL/models" > /dev/null || { echo "judge at $JUDGE_URL is not answering" >&2; exit 1; }
echo "train_rl: run=$RUN model=$MODEL grounding_weight=$GROUND_WEIGHT data.seed=$DATA_SEED judge=$JUDGE_URL out=$OUT"

# never HF_HUB_OFFLINE=1: verl's HFModelConfig calls the hub API even for cached weights.
# sglang's torch_memory_saver is incompatible with expandable segments.
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
ulimit -n 65536 || echo "warning: could not raise the open-file limit to 65536" >&2
RAY_TMP=/tmp/ray-$$  # Ray needs a short absolute tmp dir
mkdir -p "$OUT/checkpoints" "$RAY_TMP"
export WANDB_DIR=${WANDB_DIR:-$OUT}  # wandb/ and the hydra log (hydra/) go to the run dir, not the working directory
"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=True algorithm.use_kl_in_reward=False \
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/val.parquet" data.train_max_samples=-1 \
  data.seed="$DATA_SEED" data.train_batch_size=512 data.val_batch_size=512 \
  data.max_prompt_length=8192 data.max_response_length=4096 \
  data.filter_overlong_prompts=False data.truncation=error data.image_key=images data.trust_remote_code=True \
  actor_rollout_ref.model.path="$MODEL" actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=0 actor_rollout_ref.actor.freeze_vision_tower=False \
  actor_rollout_ref.actor.optim.lr=1e-5 actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=512 actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  "+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[\"Qwen3VLTextDecoderLayer\"]" \
  actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 actor_rollout_ref.rollout.max_model_len=12288 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 actor_rollout_ref.rollout.max_num_seqs=512 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=null \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=null \
  actor_rollout_ref.ref.fsdp_config.param_offload=False actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  "+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[\"Qwen3VLTextDecoderLayer\"]" \
  reward.custom_reward_function.path="$REPO/grounded_reasoning/rl.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.success_weight=1.0 \
  +reward.custom_reward_function.reward_kwargs.format_weight=1.0 \
  +reward.custom_reward_function.reward_kwargs.grounding_weight="$GROUND_WEIGHT" \
  reward.reward_manager.name=naive reward.num_workers=8 \
  trainer.critic_warmup=0 trainer.total_epochs=5 \
  trainer.save_freq=50 trainer.test_freq=-1 trainer.val_before_train=False \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.logger="[console,wandb]" \
  trainer.project_name="${WANDB_PROJECT:-grounded_reasoning}" trainer.experiment_name="$RUN" \
  trainer.default_local_dir="$OUT/checkpoints" hydra.run.dir="$OUT/hydra" \
  trainer.resume_mode=auto trainer.max_actor_ckpt_to_keep=4 \
  +ray_kwargs.ray_init._temp_dir="$RAY_TMP" +ray_kwargs.ray_init.num_gpus=8 "$@"

# Merge the last checkpoint into one model.safetensors; copy the tokenizer and processor files from the init, but not
# its safetensors index (it names shards the merge does not write).
VERL_DIR=${VERL_DIR:-$("$PYTHON" -c 'import os, verl; print(os.path.dirname(os.path.dirname(verl.__file__)))')}
CKPT=$OUT/checkpoints/global_step_$(cat "$OUT/checkpoints/latest_checkpointed_iteration.txt")/actor
"$PYTHON" "$VERL_DIR/scripts/merge_fsdp_checkpoint.py" --ckpt_dir "$CKPT"
for f in added_tokens.json chat_template.jinja merges.txt preprocessor_config.json special_tokens_map.json \
  tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json; do
  if [ ! -e "$CKPT/huggingface/$f" ] && [ -e "$MODEL/$f" ]; then cp "$MODEL/$f" "$CKPT/huggingface/"; fi
done
rm -f "$CKPT/huggingface/model.safetensors.index.json"
ln -f "$CKPT"/huggingface/* "$OUT/"  # hard links: the HF weights sit directly in the run dir
echo "train_rl: HF weights of $(basename "$(dirname "$CKPT")") in $OUT"
