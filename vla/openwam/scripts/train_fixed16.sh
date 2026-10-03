#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Fixed-16 π-style backbone comparison — one launch per backbone.
#
#   bash scripts/train_fixed16.sh wan22_ti2v_5b
#   bash scripts/train_fixed16.sh qwen3_vl_4b
#   BATCH_SIZE=4 EPOCHS=10 bash scripts/train_fixed16.sh cosmos_predict25
#
# Two nodes (run on each, with NODE_RANK 0 and 1):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> BATCH_SIZE=16 \
#     bash scripts/train_fixed16.sh rex_omni_3b
#
# Every backbone must be launched through this script, with the same knobs, or
# the comparison stops being one. What varies per run is the backbone name and
# nothing else; the protocol (LRs, schedule, lambda_video=0, action width) comes
# from configs/experiment/fixed16_comparison.yaml and the defaults below.
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

BACKBONE="${1:?usage: $0 <backbone>   # wan22_ti2v_5b | cosmos_predict25 | qwen3_vl_4b | paligemma_3b | rex_omni_3b | locate_anything_3b | rynnbrain_2b}"

# ── Which side of the comparison this backbone sits on ──
case "$BACKBONE" in
    wan22_ti2v_5b|cosmos_predict25)
        MODEL_ARGS="model=dual_system_fixed16 model/video_backbone=${BACKBONE}" ;;
    qwen3_vl_4b|qwen3_vl_2b|paligemma_3b|rex_omni_3b|locate_anything_3b|rynnbrain_2b)
        MODEL_ARGS="model=vlm_system model/vlm_backbone=${BACKBONE}" ;;
    *)  echo "error: unknown backbone '${BACKBONE}'." >&2; exit 1 ;;
esac

# ── Environment ──
# Two active venvs, because Cosmos and the VLM/Wan group require different transformers:
#   cosmos    cosmos-oss pins transformers==4.51.3
#   t457      Qwen3-VL / RynnBrain / PaliGemma / Rex-Omni / LocateAnything need
#             4.57. RynnBrain-2B is itself a Qwen3-VL derivative.
# See envs/*.env for the settings each one carries.
ENV_ROOT="${ENV_ROOT:-/path/to/envs}"
case "$BACKBONE" in
    cosmos_predict25) export OPENWAM_ENV="${OPENWAM_ENV:-${ENV_ROOT}/cosmos}"; . envs/cosmos.env ;;
    *)                export OPENWAM_ENV="${OPENWAM_ENV:-${ENV_ROOT}/t457}";   . envs/vlm.env ;;
esac
[ -x "${OPENWAM_ENV}/bin/torchrun" ] || {
    echo "error: ${OPENWAM_ENV}/bin/torchrun missing. Without the shim, scripts/train.sh" >&2
    echo "       picks up the system torchrun and runs the system interpreter — a" >&2
    echo "       different transformers, without decord/lmdb. See README." >&2
    exit 1
}

# ── Data ──
# RoboTwin, clean split only. unify_action=false with action_dim=20: the unified
# 80-D space is mostly padding, and since padding is masked out of the loss the
# action numbers stop being comparable across benchmarks. 20 = RoboTwin eef
# (xyz + rot6d + grip, both arms); use 14 for action_mode=joint.
DATASET_DIR="${DATASET_DIR:-/path/to/RoboTwin2.0/dataset}"
VARIANT="${VARIANT:-clean_50}"
ACTION_DIM="${ACTION_DIM:-20}"

# ── Scale ──
# GLOBAL_BATCH is the number that has to match across backbones; BATCH_SIZE is
# per GPU and is whatever that backbone's memory allows. The gap is closed with
# gradient accumulation, computed here so the global value cannot drift when the
# per-GPU batch or the GPU count changes.
#
# Measured peaks at 384x320 RoboTwin, 8 x H200 (143.8 GiB), ZeRO-2:
#   wan22_ti2v_5b      48 -> 48 GiB      cosmos_predict25    8 -> 32 GiB
#   qwen3_vl_4b        32 -> 93 GiB      paligemma_3b       32 -> 95 GiB
#   rex_omni_3b        32 -> 120 GiB (tight; 16 is the safer choice)
#   locate_anything_3b  8 -> 80 GiB, OOM at 12
# LocateAnything is the outlier because MoonViT packs the whole batch into one
# attention sequence, so its cost grows with the square of the batch, not
# linearly. Gradient checkpointing does not reach it (the tower exposes no
# toggle, and nothing here forces one onto it), so 8 is its ceiling.
# Resolved exactly as scripts/train.sh does, including the cloud scheduler
# variables: this script sizes gradient accumulation against the world, so if it
# read a different topology than the one torchrun launches, the global batch
# would silently differ from the banner again.
NPROC_PER_NODE="${NPROC_PER_NODE:-${HOST_GPU_NUM:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"
# Multi-node: the world size is nodes x GPUs-per-node, and accumulation has to be
# computed against the world, not against one node. Dividing by NPROC_PER_NODE
# alone silently multiplied the real global batch by the node count -- two nodes
# with BATCH_SIZE=16 produced 512 while the banner still claimed 256, which makes
# that run incomparable to every other backbone with nothing in the logs to say
# so. NNODES mirrors scripts/train.sh, which already reads it for torchrun.
NNODES="${NNODES:-${HOST_NUM:-${WORLD_SIZE:-1}}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"

WORLD_GPUS=$(( NPROC_PER_NODE * NNODES ))
PER_STEP=$(( BATCH_SIZE * WORLD_GPUS ))
if [ $(( GLOBAL_BATCH % PER_STEP )) -ne 0 ]; then
    echo "error: GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by" >&2
    echo "       BATCH_SIZE x (${NPROC_PER_NODE} GPUs x ${NNODES} nodes) = ${PER_STEP}." >&2
    echo "       Pick a per-GPU batch that divides it, or the runs stop being comparable." >&2
    exit 1
fi
GRAD_ACCUM=$(( GLOBAL_BATCH / PER_STEP ))
EPOCHS="${EPOCHS:-10}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
OUTPUT_PATH="${OUTPUT_PATH:-/path/to/checkpoints/fixed16_${BACKBONE}_${VARIANT}}"

echo "╔══════════════════════════════════════════════════════╗"
printf "║  Fixed-16 comparison — %-30s║\n" "${BACKBONE}"
printf "║  data: RoboTwin %-37s║\n" "${VARIANT}"
printf "║  env:  %-46s║\n" "${OPENWAM_ENV}"
printf "║  %-52s║\n" \
    "${NNODES} nodes x ${NPROC_PER_NODE} GPUs x batch ${BATCH_SIZE} x accum ${GRAD_ACCUM} = global ${GLOBAL_BATCH}"
printf "║  out:  %-46s║\n" "${OUTPUT_PATH}"
echo "╚══════════════════════════════════════════════════════╝"

NPROC_PER_NODE="${NPROC_PER_NODE}" NNODES="${NNODES}" exec bash scripts/train.sh \
    +experiment=fixed16_comparison \
    ${MODEL_ARGS} \
    dataloader.dataset_dir="${DATASET_DIR}" \
    dataloader.variant="${VARIANT}" \
    dataloader.unify_action=false \
    model.architecture.action_dim="${ACTION_DIM}" \
    model.architecture.state_dim="${ACTION_DIM}" \
    training.num_epochs="${EPOCHS}" \
    training.batch_size="${BATCH_SIZE}" \
    training.gradient_accumulation_steps="${GRAD_ACCUM}" \
    training.save_steps="${SAVE_STEPS}" \
    training.output_path="${OUTPUT_PATH}" \
    "${@:2}"
