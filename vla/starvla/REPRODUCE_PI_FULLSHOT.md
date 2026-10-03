# Qwen3-PI and WanPI Full-Shot Training

This recipe reproduces the GR1 full-shot Qwen3-PI / WanPI runs.
Run Qwen3-PI and WanPI on separate 8-GPU nodes. Do not queue both models on
one node unless that behavior is explicitly wanted.

## Inputs

- Dataset root: `/path/to/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot`
- Dataset mixture: `gr1_multi_concat` (24 tasks)
- Full-shot setting: `NUM_SHOT=null` (1000 trajectories per task)
- Qwen backbone: `/path/to/backbones/VLM/Qwen3-VL-4B`
- Wan backbone: `/path/to/backbones/WAM/Wan2.2-TI2V-5B-Diffusers`

The default `NUM_SHOT=100` remains the 100-shot recipe. Full-shot must pass
`NUM_SHOT=null`; `load_all_data_for_training=true` alone does not disable the
trajectory limit.

## Shared Training Settings

- 8 GPUs with DeepSpeed ZeRO-2 and bf16
- batch size 32 per GPU, global batch size 256
- 40,000 optimizer steps
- checkpoints at steps 20,000 and 40,000
- backbone LR `1e-5`
- conditioner and action expert LR `1e-4`
- action/state dimensions 29/58
- action horizon 16
- two flow-noise repeats during training and four inference integration steps

Configure W&B without committing the API key, for example with `wandb login`
or a private `WANDB_API_KEY` environment variable.

## Qwen3-PI

```bash
cd /path/to/starvla
STYLE=qwen3_pi \
SMOKE=0 \
NUM_GPUS=8 \
MASTER_PORT=29620 \
PER_DEVICE_BATCH_SIZE=32 \
MAX_TRAIN_STEPS=40000 \
SAVE_INTERVAL=20000 \
EVAL_INTERVAL=2000 \
NUM_SHOT=null \
RUN_SUFFIX=_fullshot \
WANDB_MODE=online \
bash scripts/run_scripts_vlm_weight/run_gr1_pi_backbones_local.sh
```

## WanPI

```bash
cd /path/to/starvla
STYLE=wan_pi \
SMOKE=0 \
NUM_GPUS=8 \
MASTER_PORT=29621 \
PER_DEVICE_BATCH_SIZE=32 \
MAX_TRAIN_STEPS=40000 \
SAVE_INTERVAL=20000 \
EVAL_INTERVAL=2000 \
NUM_SHOT=null \
RUN_SUFFIX=_fullshot \
WANDB_MODE=online \
bash scripts/run_scripts_vlm_weight/run_gr1_pi_backbones_local.sh
```

## Startup Checks

Confirm all of the following before leaving a run unattended:

1. The launcher prints `NUM SHOT: null`, `GPUs / BS: 8 / 32`, and 40,000 steps.
2. Dataset initialization reports 1000 trajectories for every task.
3. W&B creates the intended run.
4. At least 10 optimizer steps produce finite action loss.
5. All eight GPUs are active and no traceback, OOM, CUDA, or NCCL error appears.

The source snapshot intentionally excludes checkpoints, W&B files, videos,
caches, credentials, and Git metadata.
