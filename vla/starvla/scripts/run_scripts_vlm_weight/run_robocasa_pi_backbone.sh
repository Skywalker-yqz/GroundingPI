#!/usr/bin/env bash
set -euo pipefail

# Train the new StarVLA PI architecture on RoboCasa with a selectable backbone.
#
# Examples:
#   BACKBONE=qwen3 bash scripts/run_scripts_vlm_weight/run_robocasa_pi_backbone.sh
#   BACKBONE=cosmos DATA_PERCENT=50 bash scripts/run_scripts_vlm_weight/run_robocasa_pi_backbone.sh
#   BACKBONE=wan BASE_MODEL=/path/to/Wan2.2-TI2V-5B-Diffusers bash scripts/run_scripts_vlm_weight/run_robocasa_pi_backbone.sh
#
# Global batch size is:
#   PER_DEVICE_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# Optional: a shell file to source before launching (e.g. a uv/conda activation script).
UV_ACTIVATE="${UV_ACTIVATE:-}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

BACKBONE="${BACKBONE:-qwen3}"
DATA_ROOT="${DATA_ROOT:-/path/to/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot}"
DATA_MIX="${DATA_MIX:-gr1_multi_concat}"
DATA_PERCENT="${DATA_PERCENT:-100}"
SUBSET_SEED="${SUBSET_SEED:-42}"

case "${DATA_PERCENT}" in
  25) DEFAULT_MAX_TRAIN_STEPS=10000; DEFAULT_NUM_WARMUP_STEPS=500 ;;
  50) DEFAULT_MAX_TRAIN_STEPS=20000; DEFAULT_NUM_WARMUP_STEPS=1000 ;;
  75) DEFAULT_MAX_TRAIN_STEPS=30000; DEFAULT_NUM_WARMUP_STEPS=1500 ;;
  100) DEFAULT_MAX_TRAIN_STEPS=40000; DEFAULT_NUM_WARMUP_STEPS=2000 ;;
  *) echo "ERROR: DATA_PERCENT must be 25, 50, 75, or 100; got '${DATA_PERCENT}'" >&2; exit 2 ;;
esac

# Topology aliases match OpenWAM/scripts/train.sh and common cloud schedulers.
# NUM_GPUS is GPUs/processes per machine, while WORLD_SIZE below is global.
detected_gpus="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
[[ "${detected_gpus}" -ge 1 ]] 2>/dev/null || detected_gpus=1
NUM_GPUS="${NUM_GPUS:-${NPROC_PER_NODE:-${HOST_GPU_NUM:-${detected_gpus}}}}"
NUM_MACHINES="${NUM_MACHINES:-${NNODES:-${HOST_NUM:-1}}}"
MACHINE_RANK="${MACHINE_RANK:-${NODE_RANK:-${RANK:-0}}}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
if [[ -n "${GRADIENT_ACCUMULATION_STEPS+x}" ]]; then
  GRAD_ACCUM_EXPLICIT=1
else
  GRAD_ACCUM_EXPLICIT=0
  GRADIENT_ACCUMULATION_STEPS=1
fi
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-${DEFAULT_MAX_TRAIN_STEPS}}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-${DEFAULT_NUM_WARMUP_STEPS}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"

BACKBONE_LR="${BACKBONE_LR:-1e-5}"
EXPERT_LR="${EXPERT_LR:-1e-4}"
MASTER_PORT="${MASTER_PORT:-29600}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-starvla-robocasa-backbones}"
WANDB_ENTITY="${WANDB_ENTITY:-null}"   # null = the account behind WANDB_API_KEY
OUTPUT_ROOT_BASE="${OUTPUT_ROOT:-${REPO_ROOT}/results/data_efficiency}"

case "${BACKBONE}" in
  qwen3|qwen3_pi)
    BACKBONE=qwen3
    FRAMEWORK=QwenPI
    BASE_MODEL="${BASE_MODEL:-/path/to/backbones/VLM/Qwen3-VL-4B}"
    FREEZE_MODULES="qwen_vl_interface.model.model.visual"
    MAX_PER_DEVICE_BATCH=32
    backbone_args=(
      --framework.qwenvl.base_vlm "${BASE_MODEL}"
      --framework.qwenvl.vl_hidden_dim 2560
      --framework.qwenvl.select_layer -1
      --framework.qwenvl.attn_implementation flash_attention_2
      --trainer.enable_gradient_checkpointing True
      --trainer.learning_rate.qwen_vl_interface "${BACKBONE_LR}"
    )
    ;;
  wan|wan_pi)
    BACKBONE=wan
    FRAMEWORK=WanPI
    BASE_MODEL="${BASE_MODEL:-/path/to/backbones/WAM/Wan2.2-TI2V-5B-Diffusers}"
    FREEZE_MODULES=""
    MAX_PER_DEVICE_BATCH=32
    backbone_args=(
      --framework.world_model.base_wm "${BASE_MODEL}"
      --framework.action_model.action_expert_fp32 True
      --trainer.enable_gradient_checkpointing False
      --trainer.learning_rate.backbone "${BACKBONE_LR}"
      --trainer.learning_rate.qwen_vl_interface 0.0
    )
    ;;
  cosmos|cosmos_pi)
    BACKBONE=cosmos
    FRAMEWORK=CosmosPI
    BASE_MODEL="${BASE_MODEL:-/path/to/backbones/WAM/Cosmos-Predict2.5-2B}"
    VENV_DIR="${COSMOS_VENV_DIR:-/path/to/envs/cosmos}"
    FREEZE_MODULES="backbone.vae,backbone.reason1"
    MAX_PER_DEVICE_BATCH=8
    COSMOS_SOURCE="${COSMOS_SOURCE:-${REPO_ROOT}/third_party/cosmos-predict2.5}"
    backbone_args=(
      --framework.world_model.type cosmos
      --framework.world_model.base_wm "${BASE_MODEL}"
      --framework.world_model.model_variant base/post-trained
      --framework.world_model.text_encoder "${COSMOS_TEXT_ENCODER:-/path/to/backbones/WAM/Cosmos-Reason1-7B}"
      --framework.world_model.cosmos_source "${COSMOS_SOURCE}"
      --framework.action_model.action_expert_fp32 True
      --trainer.enable_gradient_checkpointing False
      --trainer.learning_rate.backbone "${BACKBONE_LR}"
      --trainer.learning_rate.qwen_vl_interface 0.0
    )
    ;;
  rex|rex_omni)
    BACKBONE=rex_omni
    FRAMEWORK=VLMBackbonePI
    BASE_MODEL="${BASE_MODEL:-/path/to/backbones/VLM/Rex-Omni-3B}"
    FREEZE_MODULES=""
    MAX_PER_DEVICE_BATCH=16
    backbone_args=(
      --framework.qwenvl.base_vlm "${BASE_MODEL}"
      --framework.qwenvl.vlm_type rex_omni
      --framework.qwenvl.attn_implementation flash_attention_2
      --trainer.enable_gradient_checkpointing False
      --trainer.learning_rate.qwen_vl_interface "${BACKBONE_LR}"
    )
    ;;
  paligemma|pali)
    BACKBONE=paligemma
    FRAMEWORK=VLMBackbonePI
    BASE_MODEL="${BASE_MODEL:-/path/to/backbones/VLM/Paligemma-3B}"
    FREEZE_MODULES=""
    MAX_PER_DEVICE_BATCH=32
    backbone_args=(
      --framework.qwenvl.base_vlm "${BASE_MODEL}"
      --framework.qwenvl.vlm_type paligemma
      # PaliGemma is a prefix-LM (bidirectional over image + prompt). HF's FA2 path
      # drops its prefix mask and runs GemmaAttention causal, so never use FA2 here.
      --framework.qwenvl.attn_implementation sdpa
      --trainer.enable_gradient_checkpointing False
      --trainer.learning_rate.qwen_vl_interface "${BACKBONE_LR}"
    )
    ;;
  locate|locate_anything)
    BACKBONE=locate_anything
    FRAMEWORK=VLMBackbonePI
    BASE_MODEL="${BASE_MODEL:-/path/to/backbones/VLM/LocateAnything-3B}"
    FREEZE_MODULES=""
    MAX_PER_DEVICE_BATCH=8
    backbone_args=(
      --framework.qwenvl.base_vlm "${BASE_MODEL}"
      --framework.qwenvl.vlm_type locateanything
      --framework.qwenvl.allow_remote_code True
      --framework.qwenvl.attn_implementation sdpa
      --trainer.enable_gradient_checkpointing False
      --trainer.learning_rate.qwen_vl_interface "${BACKBONE_LR}"
    )
    ;;
  rynn2|rynnbrain2)
    BACKBONE=rynnbrain2; FRAMEWORK=VLMBackbonePI
    BASE_MODEL="${BASE_MODEL:-/path/to/backbones/VLM/RynnBrain-2B}"
    VENV_DIR="${RYNN_VENV_DIR:-${REPO_ROOT}/.venv-rynn}"
    FREEZE_MODULES=""; MAX_PER_DEVICE_BATCH=8
    backbone_args=(--framework.qwenvl.base_vlm "${BASE_MODEL}" --framework.qwenvl.vlm_type rynnbrain2 --framework.qwenvl.attn_implementation sdpa --trainer.enable_gradient_checkpointing False --trainer.learning_rate.qwen_vl_interface "${BACKBONE_LR}")
    ;;
  *)
    echo "ERROR: BACKBONE must be qwen3, wan, cosmos, rex, paligemma, locate, or rynn2; got '${BACKBONE}'" >&2
    exit 2
    ;;
esac

if [[ "${BACKBONE}" == "cosmos" ]]; then
  # transformer-engine is built against the cuDNN wheel torch ships, not the
  # system one; put the environment's cuDNN first (override with COSMOS_CUDNN_LIB).
  COSMOS_CUDNN_LIB="${COSMOS_CUDNN_LIB:-$("${VENV_DIR}/bin/python" -c 'import importlib.util, os; s = importlib.util.find_spec("nvidia.cudnn"); print(os.path.join(os.path.dirname(s.origin), "lib") if s else "")' 2>/dev/null || true)}"
  if [[ -n "${COSMOS_CUDNN_LIB}" ]]; then
    export LD_LIBRARY_PATH="${COSMOS_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
  fi
  export NVTE_FUSED_ATTN=0
  unset PYTORCH_CUDA_ALLOC_CONF
fi

WORLD_SIZE=$((NUM_GPUS * NUM_MACHINES))
if (( GRAD_ACCUM_EXPLICIT == 0 )); then
  # Pick the smallest accumulation that keeps the per-GPU batch within the
  # backbone's measured safe bound while preserving the exact global batch.
  for candidate in $(seq 1 "${GLOBAL_BATCH_SIZE}"); do
    denominator=$((WORLD_SIZE * candidate))
    if (( GLOBAL_BATCH_SIZE % denominator == 0 )) && \
       (( GLOBAL_BATCH_SIZE / denominator <= MAX_PER_DEVICE_BATCH )); then
      GRADIENT_ACCUMULATION_STEPS="${candidate}"
      break
    fi
  done
fi
batch_denominator=$((WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( GLOBAL_BATCH_SIZE % batch_denominator != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by world_size*gradient_accumulation=${batch_denominator}" >&2
  exit 2
fi
PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / batch_denominator))
if (( PER_DEVICE_BATCH_SIZE < 1 || PER_DEVICE_BATCH_SIZE > MAX_PER_DEVICE_BATCH )); then
  echo "ERROR: computed PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}; ${BACKBONE} requires 1..${MAX_PER_DEVICE_BATCH}" >&2
  exit 2
fi

if [[ -n "${UV_ACTIVATE}" && ! -f "${UV_ACTIVATE}" ]]; then
  echo "ERROR: activation script not found: ${UV_ACTIVATE}" >&2
  exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/accelerate" ]]; then
  echo "ERROR: StarVLA environment not found: ${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: dataset root not found: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "ERROR: ${BACKBONE} backbone not found: ${BASE_MODEL}" >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/${BACKBONE}"
RUN_ID="${RUN_ID:-${BACKBONE}_robocasa_pct${DATA_PERCENT}_seed${SUBSET_SEED}_${MAX_TRAIN_STEPS}steps}"

export WANDB_MODE TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

cd "${REPO_ROOT}"
if [[ -n "${UV_ACTIVATE}" ]]; then source "${UV_ACTIVATE}"; fi

echo "============================================================"
echo "StarVLA RoboCasa PI training"
echo "backbone/framework : ${BACKBONE} / ${FRAMEWORK}"
echo "base model         : ${BASE_MODEL}"
echo "dataset/mix        : ${DATA_ROOT} / ${DATA_MIX}"
echo "data subset        : ${DATA_PERCENT}% per task (seed ${SUBSET_SEED})"
echo "GPUs               : ${NUM_GPUS}"
echo "machines/rank      : ${NUM_MACHINES} / ${MACHINE_RANK}"
echo "rendezvous         : ${MAIN_PROCESS_IP}:${MASTER_PORT}"
echo "batch              : ${PER_DEVICE_BATCH_SIZE}/GPU x ${NUM_GPUS} GPUs/node x ${NUM_MACHINES} nodes x ${GRADIENT_ACCUMULATION_STEPS} accumulation = ${GLOBAL_BATCH_SIZE} global"
echo "steps/warmup       : ${MAX_TRAIN_STEPS} / ${NUM_WARMUP_STEPS}"
echo "output             : ${OUTPUT_ROOT}/${RUN_ID}"
echo "============================================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: configuration validated; training was not started."
  exit 0
fi

"${VENV_DIR}/bin/accelerate" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${WORLD_SIZE}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MASTER_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/pi_backbone_train_gr1.yaml \
  --framework.name "${FRAMEWORK}" \
  "${backbone_args[@]}" \
  --framework.action_model.action_dim 29 \
  --framework.action_model.state_dim 58 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.repeated_diffusion_steps 2 \
  --framework.action_model.num_inference_timesteps 4 \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.num_shot null \
  --datasets.vla_data.data_percent "${DATA_PERCENT}" \
  --datasets.vla_data.subset_seed "${SUBSET_SEED}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.pretrained_checkpoint null \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --trainer.learning_rate.base "${EXPERT_LR}" \
  --trainer.learning_rate.backbone_conditioner "${EXPERT_LR}" \
  --trainer.learning_rate.action_model "${EXPERT_LR}" \
  --run_root_dir "${OUTPUT_ROOT}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"
