#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# OpenWAM Training — torchrun + Accelerate DeepSpeed
#
# DeepSpeed ZeRO stage is set in train.yaml (training.zero_stage: 2),
# or overridden from the CLI:
#   bash scripts/train.sh training.zero_stage=1
#
# ── Single-node (auto-detect GPUs) ──
#   bash scripts/train.sh
#   bash scripts/train.sh training.learning_rate=5e-5
#   NPROC_PER_NODE=4 bash scripts/train.sh
#
# ── Multi-node (env vars set by cloud scheduler) ──
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash scripts/train.sh
# Schedulers that export HOST_NUM, HOST_GPU_NUM and RANK are understood too, so the
# topology resolves with none of these set.
# ───────────────────────���──────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Quiet logging defaults ──
# Force NCCL_DEBUG to WARN to suppress the per-channel/per-rank INFO spam
# (RingP2P, comm init, topology probes) that buries training progress. We
# unconditionally override here because cloud environments commonly export
# NCCL_DEBUG=INFO by default. Opt back in with OPENWAM_VERBOSE_NCCL=1.
if [[ "${OPENWAM_VERBOSE_NCCL:-0}" == "1" ]]; then
    export NCCL_DEBUG=INFO
else
    export NCCL_DEBUG=WARN
fi

# ── GPU / Node topology ──
# Priority: explicit override > cloud scheduler vars (HOST_GPU_NUM/HOST_NUM) >
# auto-detect. HOST_NUM has to come before WORLD_SIZE: torchrun's convention is
# that WORLD_SIZE counts *processes*, not nodes, so a launcher exporting
# WORLD_SIZE=16 for a 2x8 job would otherwise make NNODES 16 and torchrun would
# sit waiting for 128 ranks.
NPROC_PER_NODE="${NPROC_PER_NODE:-${HOST_GPU_NUM:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-${HOST_NUM:-${WORLD_SIZE:-1}}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  OpenWAM Training                                   ║"
echo "║  Nodes: ${NNODES}  GPUs/node: ${NPROC_PER_NODE}  Rank: ${NODE_RANK}            ║"
echo "║  Master: ${MASTER_ADDR}:${MASTER_PORT}                   ║"
echo "╚══════════════════════════════════════════════════════╝"

torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    scripts/train.py \
    "$@"
