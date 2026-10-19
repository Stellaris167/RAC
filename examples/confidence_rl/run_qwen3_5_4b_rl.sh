#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <baseline|pair|rank|pair_rank> <grpo|reinforce_plus_plus> [Hydra overrides...]" >&2
  exit 1
fi

VARIANT=$1
ALGO=$2
shift 2

case "$VARIANT" in
  baseline) RANK_COEF=0.0; PAIR_COEF=0.0 ;;
  pair) RANK_COEF=0.0; PAIR_COEF=0.3 ;;
  rank) RANK_COEF=0.2; PAIR_COEF=0.0 ;;
  pair_rank) RANK_COEF=0.2; PAIR_COEF=0.3 ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 1 ;;
esac

case "$ALGO" in
  grpo|reinforce_plus_plus) ;;
  *) echo "Unknown algorithm: $ALGO" >&2; exit 1 ;;
esac

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/home/yby/miniforge3/envs/skillopt/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/yby/datasets/SCS_data}
MODEL=${MODEL:-$PROJECT_DIR/checkpoints/qwen3.5-4b-direct-prefix-sft/huggingface}
TRAIN_DATA=$DATA_ROOT/train/train_pair_main.parquet
VAL_DATA=${VAL_DATA:-$DATA_ROOT/rac_qwen35_eval.parquet}
RUN_NAME=${RUN_NAME:-qwen3.5-4b-${ALGO}-${VARIANT}}
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints/$RUN_NAME
VALIDATION_DIR=$PROJECT_DIR/logs/val_dumps/$RUN_NAME
ROLLOUT_DIR=$PROJECT_DIR/logs/rollouts/$RUN_NAME
METRICS_FILE=$PROJECT_DIR/logs/mlp_metrics/$RUN_NAME.jsonl

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export VERL_FILE_LOGGER_PATH="$METRICS_FILE"

mkdir -p "$CHECKPOINT_DIR" "$VALIDATION_DIR" "$ROLLOUT_DIR" "$(dirname "$METRICS_FILE")"
touch "$METRICS_FILE"

restore_outputs() {
  if [[ "$(id -u)" == 0 ]]; then
    chown -R "$(stat -c '%u:%g' /home/yby)" \
      "$CHECKPOINT_DIR" "$VALIDATION_DIR" "$ROLLOUT_DIR" "$METRICS_FILE"
  fi
}
trap restore_outputs EXIT
restore_outputs

cmd=(
  "$PYTHON_BIN" -B -m verl.trainer.main_ppo
  "algorithm.adv_estimator=$ALGO"
  "data.train_files=$TRAIN_DATA"
  "data.val_files=$VAL_DATA"
  "data.custom_cls.path=$PROJECT_DIR/verl/trainer/reward_fn/confidence_dataset.py"
  "data.custom_cls.name=ConfidenceRLDataset"
  "data.train_batch_size=128"
  "data.shuffle=false"
  "data.seed=42"
  "data.max_prompt_length=7680"
  "data.max_response_length=128"
  "data.prompt_key=message_qwenvl"
  "data.trust_remote_code=true"
  "data.reward_fn_key=data_source"
  "++data.max_model_len=8192"
  "++data.local_data_root=$DATA_ROOT"
  "++data.inject_confidence=true"
  "++data.response_schema=boxed_direct"
  "++data.apply_chat_template_kwargs.enable_thinking=false"
  "++data.split_data_source_by_corruption=true"
  "data.truncation=error"
  "data.image_key=images"
  "actor_rollout_ref.model.path=$MODEL"
  "actor_rollout_ref.model.enable_gradient_checkpointing=true"
  "actor_rollout_ref.model.trust_remote_code=true"
  "actor_rollout_ref.model.use_remove_padding=false"
  "++actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=5"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=cosine"
  "actor_rollout_ref.actor.optim.min_lr_ratio=0.5"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=false"
  "actor_rollout_ref.actor.use_torch_compile=false"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.actor.use_kl_loss=true"
  "actor_rollout_ref.actor.kl_loss_coef=0.005"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.fsdp_config.param_offload=true"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.n=8"
  "actor_rollout_ref.rollout.temperature=0.6"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "++actor_rollout_ref.rollout.repetition_penalty=1.0"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.65"
  "actor_rollout_ref.rollout.max_model_len=8192"
  "actor_rollout_ref.rollout.max_num_seqs=64"
  "actor_rollout_ref.rollout.enforce_eager=true"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576"
  "actor_rollout_ref.ref.fsdp_config.param_offload=true"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152"
  "++reward.custom_reward_function.path=$PROJECT_DIR/verl/trainer/reward_fn/confidence_reward.py"
  "++reward.custom_reward_function.name=compute_score"
  "++reward.custom_reward_function.reward_kwargs.format_reward_coef=0.3"
  "++reward.custom_reward_function.reward_kwargs.format_reward_warmup_start=1.0"
  "++reward.custom_reward_function.reward_kwargs.format_reward_warmup_steps=10"
  "++reward.custom_reward_function.reward_kwargs.tail_text_penalty_coef=0.1"
  "++reward.custom_reward_function.reward_kwargs.response_schema=boxed"
  "++reward.reward_manager.source=register"
  "++reward.reward_manager.name=naive"
  "++reward.reward_shaping.path=$PROJECT_DIR/verl/trainer/reward_fn/reward_shaping.py"
  "++reward.reward_shaping.name=apply_reward_shaping"
  "++reward.reward_shaping.rank_reward_coef=$RANK_COEF"
  "++reward.reward_shaping.rank_reward_margin=0.05"
  "++reward.reward_shaping.corr_reward_coef=$PAIR_COEF"
  "++reward.reward_shaping.corr_reward_margin=0.05"
  "++reward.reward_shaping.corr_reward_alpha=0.1"
  "++reward.reward_shaping.len_reward_coef=0.0"
  "algorithm.use_kl_in_reward=false"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.6"
  "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
  "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=false"
  "trainer.total_epochs=2"
  "trainer.total_training_steps=32"
  "trainer.save_freq=16"
  "trainer.test_freq=4"
  "trainer.val_before_train=false"
  "trainer.resume_mode=disable"
  "trainer.max_actor_ckpt_to_keep=2"
  "trainer.project_name=confidence-rl"
  "trainer.experiment_name=$RUN_NAME"
  "trainer.logger=['console','file']"
  "trainer.n_gpus_per_node=8"
  "trainer.nnodes=1"
  "trainer.validation_data_dir=$VALIDATION_DIR"
  "trainer.default_local_dir=$CHECKPOINT_DIR"
  "trainer.rollout_data_dir=$ROLLOUT_DIR"
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH=$METRICS_FILE"
  "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONDONTWRITEBYTECODE=1"
  "hydra.run.dir=$CHECKPOINT_DIR/hydra"
  "$@"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

cd "$PROJECT_DIR"
"${cmd[@]}"
