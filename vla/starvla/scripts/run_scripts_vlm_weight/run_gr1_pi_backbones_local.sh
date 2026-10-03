#!/bin/bash
set -euo pipefail

# Single-node GR1 smoke/formal training for the fixed StarVLA pi action expert.
# Only the perception backbone changes; data and trainer follow the GR1 recipe.
#
# Usage:
#   SMOKE=1 STYLE=qwen3_pi bash scripts/run_scripts_vlm_weight/run_gr1_pi_backbones_local.sh
#   SMOKE=1 STYLE=wan_pi   bash scripts/run_scripts_vlm_weight/run_gr1_pi_backbones_local.sh

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ACCELERATE="${ACCELERATE:-${REPO_ROOT}/.venv/bin/accelerate}"
BASE_CONFIG=starVLA/config/training/pi_backbone_train_gr1.yaml
DEEPSPEED_CONFIG=starVLA/config/deepseeds/deepspeed_zero2.yaml
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/Checkpoints}"

STYLE="${STYLE:-qwen3_pi}"
SMOKE="${SMOKE:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29600}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
EXPERT_LR="${EXPERT_LR:-1e-4}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
NUM_SHOT="${NUM_SHOT:-100}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

common_model_args=(
  --framework.action_model.action_dim 29
  --framework.action_model.state_dim 58
  --framework.action_model.future_action_window_size 15
  --framework.action_model.repeated_diffusion_steps 2
  --framework.action_model.num_inference_timesteps 4
  --datasets.vla_data.data_mix gr1_multi_concat
  --datasets.vla_data.num_shot "${NUM_SHOT}"
  --trainer.learning_rate.base "${EXPERT_LR}"
  --trainer.learning_rate.backbone_conditioner "${EXPERT_LR}"
  --trainer.learning_rate.action_model "${EXPERT_LR}"
)

case "${STYLE}" in
  qwen3_pi)
    framework_name=QwenPI
    model_tag=qwen3_pi
    freeze_modules=qwen_vl_interface.model.model.visual
    model_args=(
      --framework.qwenvl.base_vlm "${QWEN_MODEL:-/path/to/backbones/VLM/Qwen3-VL-4B}"
      --framework.qwenvl.vl_hidden_dim 2560
      --framework.qwenvl.select_layer -1
      --framework.qwenvl.attn_implementation flash_attention_2
      --trainer.learning_rate.qwen_vl_interface "${BACKBONE_LR}"
    )
    ;;
  wan_pi)
    framework_name=WanPI
    model_tag=wan_pi
    # WanPI hard-freezes VAE and UMT5; the Wan DiT backbone remains trainable.
    freeze_modules=""
    model_args=(
      --framework.world_model.base_wm "${WAN_MODEL:-/path/to/backbones/WAM/Wan2.2-TI2V-5B-Diffusers}"
      --framework.action_model.action_expert_fp32 True
      --trainer.enable_gradient_checkpointing False
      --trainer.learning_rate.backbone "${BACKBONE_LR}"
      --trainer.learning_rate.qwen_vl_interface 0.0
    )
    ;;
  *)
    echo "ERROR: STYLE must be qwen3_pi or wan_pi; got ${STYLE}" >&2
    exit 2
    ;;
esac

if [[ "${SMOKE}" == "1" ]]; then
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-1}"
  max_train_steps="${MAX_TRAIN_STEPS:-5}"
  save_interval=100000
  eval_interval=100000
  smoke_tag=smoke
else
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-32}"
  max_train_steps="${MAX_TRAIN_STEPS:-60000}"
  save_interval="${SAVE_INTERVAL:-5000}"
  eval_interval="${EVAL_INTERVAL:-1000}"
  smoke_tag=train
fi

run_id="${model_tag}_gr1_${smoke_tag}_bs${per_device_batch_size}_bb${BACKBONE_LR}_expert${EXPERT_LR}${RUN_SUFFIX}"

echo "========================================================"
echo "StarVLA GR1 pi training"
echo "STYLE:          ${STYLE}"
echo "FRAMEWORK:      ${framework_name}"
echo "DATA MIX:       gr1_multi_concat"
echo "NUM SHOT:       ${NUM_SHOT}"
echo "FREEZE:         ${freeze_modules:-Wan VAE + UMT5 (hard frozen)}"
echo "BACKBONE LR:    ${BACKBONE_LR}"
echo "EXPERT/COND LR: ${EXPERT_LR}"
echo "GPUs / BS:      ${NUM_GPUS} / ${per_device_batch_size} per GPU"
echo "STEPS:          ${max_train_steps}"
echo "RUN ID:         ${run_id}"
echo "========================================================"

wandb_args=(
  --wandb_project starVLA-GR1-pi-local
  # The trainer requires the key to exist. OmegaConf parses null as None, which
  # lets W&B select the account associated with WANDB_API_KEY.
  --wandb_entity "${WANDB_ENTITY:-null}"
)

cd "${REPO_ROOT}"
"${ACCELERATE}" launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --main_process_port "${MASTER_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${BASE_CONFIG}" \
  --framework.name "${framework_name}" \
  "${model_args[@]}" \
  "${common_model_args[@]}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --trainer.freeze_modules "${freeze_modules}" \
  --trainer.pretrained_checkpoint null \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.eval_interval "${eval_interval}" \
  --trainer.skip_nonfinite_loss True \
  --trainer.max_consecutive_nonfinite_skips 100 \
  --run_root_dir "${OUTPUT_ROOT}" \
  --run_id "${run_id}" \
  "${wandb_args[@]}"
