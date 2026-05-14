#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
ALGO=${ALGO:-grpo}
: "${DATA_ROOT:=}"

if [[ -z "$DATA_ROOT" ]]; then
  echo "Set DATA_ROOT to a local RAC dataset directory." >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
fi

join_path() {
  local root="${1%/}"
  local rel="${2#/}"
  printf '%s/%s' "$root" "$rel"
}

build_data_path() {
  join_path "$DATA_ROOT" "$1"
}

build_val_files() {
  local benchmarks=(
    m3cot
    mathverse
    mathvision
    mmmu
    scienceqa
    we_math
  )
  local severities=("" "_T0.2" "_T0.4" "_T0.6" "_T0.8" "_T1.0")
  local files=()
  local benchmark severity suffix file

  for benchmark in "${benchmarks[@]}"; do
    for severity in "${severities[@]}"; do
      suffix="_test_processed${severity}"
      file=$(build_data_path "${benchmark}${suffix}/test.parquet")
      files+=("\"${file}\"")
    done
  done

  local joined=""
  for file in "${files[@]}"; do
    joined+="${file},"
  done
  printf '[%s]' "${joined%,}"
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi

EXP=${EXP:-pair_main}

case $EXP in
  baseline)
    TRAIN_DATA=$(build_data_path "train/train.parquet")
    RANK_COEF=0.0
    CORR_COEF=0.0
    ;;
  ranking)
    TRAIN_DATA=$(build_data_path "train/train.parquet")
    RANK_COEF=0.2
    CORR_COEF=0.0
    ;;
  pair_T0.2)
    TRAIN_DATA=$(build_data_path "train/train_pair_T0.2.parquet")
    RANK_COEF=0.2
    CORR_COEF=0.3
    ;;
  pair_T0.4)
    TRAIN_DATA=$(build_data_path "train/train_pair_T0.4.parquet")
    RANK_COEF=0.2
    CORR_COEF=0.3
    ;;
  pair_T0.6)
    TRAIN_DATA=$(build_data_path "train/train_pair_T0.6.parquet")
    RANK_COEF=0.2
    CORR_COEF=0.3
    ;;
  pair_T0.8)
    TRAIN_DATA=$(build_data_path "train/train_pair_T0.8.parquet")
    RANK_COEF=0.2
    CORR_COEF=0.3
    ;;
  pair_T1.0)
    TRAIN_DATA=$(build_data_path "train/train_pair_T1.0.parquet")
    RANK_COEF=0.2
    CORR_COEF=0.3
    ;;
  pair)
    TRAIN_DATA=$(build_data_path "train/train_pair_main.parquet")
    RANK_COEF=0.0
    CORR_COEF=0.3
    ;;
  pair_main)
    TRAIN_DATA=$(build_data_path "train/train_pair_main.parquet")
    RANK_COEF=0.2
    CORR_COEF=0.3
    ;;
  *)
    echo "Unknown EXP=$EXP. Use: baseline, ranking, pair_T0.2, pair_T0.4, pair_T0.6, pair_T0.8, pair_T1.0, pair, pair_main"
    exit 1
    ;;
esac

VAL_FILES=$(build_val_files)
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}
VALIDATION_DIR=${VALIDATION_DIR:-$(join_path "$PROJECT_DIR" "logs/val_dumps/intern3.5vl8b-${ALGO}-${EXP}")}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$(join_path "$PROJECT_DIR" "checkpoints/intern3.5vl8b-${ALGO}-${EXP}")}

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
if [[ "${VLLM_ATTENTION_BACKEND:-}" == "XFORMERS" || "${VLLM_ATTENTION_BACKEND:-}" == "TORCH_SDPA" ]]; then
  unset VLLM_ATTENTION_BACKEND
fi
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
PROJECT_DIR=$PROJECT_DIR
PYTHON_BIN=$PYTHON_BIN
MODEL=$MODEL
DATA_ROOT=$DATA_ROOT
TRAIN_DATA=$TRAIN_DATA
VAL_FILES=$VAL_FILES
TRAINER_LOGGER=$TRAINER_LOGGER
VALIDATION_DIR=$VALIDATION_DIR
CHECKPOINT_DIR=$CHECKPOINT_DIR
EOF
  exit 0
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ALGO \
    data.train_files=\"$TRAIN_DATA\" \
    data.val_files=$VAL_FILES \
    data.custom_cls.path=\"$PROJECT_DIR/verl/trainer/reward_fn/confidence_dataset.py\" \
    data.custom_cls.name=ConfidenceRLDataset \
    data.train_batch_size=512 \
    data.max_prompt_length=2500 \
    data.max_response_length=3072 \
    data.prompt_key=message_internvl \
    data.trust_remote_code=true \
    data.reward_fn_key=data_source \
    ++data.max_model_len=60000 \
    ++data.inject_confidence=true \
    ++data.split_data_source_by_corruption=true \
    data.image_key=images \
    actor_rollout_ref.model.path=\"$MODEL\" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.use_remove_padding=true \
    ++actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    ++actor_rollout_ref.rollout.repetition_penalty=1.10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    ++reward.custom_reward_function.path=\"$PROJECT_DIR/verl/trainer/reward_fn/confidence_reward.py\" \
    ++reward.custom_reward_function.name=compute_score \
    ++reward.custom_reward_function.reward_kwargs.format_reward_coef=0.3 \
    ++reward.custom_reward_function.reward_kwargs.format_reward_warmup_start=1.0 \
    ++reward.custom_reward_function.reward_kwargs.format_reward_warmup_steps=10 \
    ++reward.custom_reward_function.reward_kwargs.tail_text_penalty_coef=0.1 \
    ++reward.reward_manager.source=register \
    ++reward.reward_manager.name=naive \
    ++reward.reward_shaping.path=\"$PROJECT_DIR/verl/trainer/reward_fn/reward_shaping.py\" \
    ++reward.reward_shaping.name=apply_reward_shaping \
    ++reward.reward_shaping.rank_reward_coef=$RANK_COEF \
    ++reward.reward_shaping.rank_reward_margin=0.05 \
    ++reward.reward_shaping.corr_reward_coef=$CORR_COEF \
    ++reward.reward_shaping.corr_reward_margin=0.05 \
    ++reward.reward_shaping.corr_reward_alpha=0.1 \
    ++reward.reward_shaping.len_reward_coef=0.0 \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    trainer.total_epochs=2 \
    trainer.save_freq=6 \
    trainer.test_freq=6 \
    trainer.project_name=confidence-rl \
    trainer.experiment_name=intern3.5vl8b-${ALGO}-${EXP} \
    trainer.logger=$TRAINER_LOGGER \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.validation_data_dir=\"$VALIDATION_DIR\" \
    trainer.default_local_dir=\"$CHECKPOINT_DIR\"
