# Fixed-16 Backbone Comparison — StarVLA side

> Part of the **GroundingPI** release. The unified environment / training / testing guide is [`../README.md`](../README.md); this file holds the per-backbone details of the RoboCasa side.

The StarVLA side of the VLM-vs-world-model comparison: seven pretrained backbones
drive one identical π-style layerwise action expert on the RoboCasa GR1 tabletop
tasks (`nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim`, 24 tasks), trained at
25 / 50 / 75 / 100 % of the demonstrations to measure data efficiency.

This is a trimmed derivative of [StarVLA](https://github.com/starVLA/starVLA).
Only the code on the PI-backbone path is kept.

| framework | backbone(s) | files |
|---|---|---|
| `QwenPI` | Qwen3-VL-4B | `starVLA/model/framework/QwenPI.py`, `modules/vlm/QWen3.py` |
| `WanPI` | Wan2.2-TI2V-5B (Diffusers layout) | `framework/WanPI.py`, `modules/world_model/Wan2.py` |
| `CosmosPI` | Cosmos-Predict2.5-2B + Cosmos-Reason1-7B | `framework/CosmosPI.py`, `modules/world_model/CosmosPredict25.py` |
| `VLMBackbonePI` | Rex-Omni-3B, PaliGemma-3B, LocateAnything-3B, RynnBrain-2B | `framework/VLMBackbonePI.py`, `modules/vlm/comparison_backbones.py`, `modules/vlm/locate_anything.py` |

Shared by all of them: `modules/projector/PiBackboneConditioner.py` (eight hidden
states tapped at normalized depths, projected to 1024) and
`modules/action_model/LayerwiseFM_ActionHeader.py` (flow-matching action head,
16 DiT layers, width 1024, 16 heads × 64, fp32 action expert, 4 inference
steps). Every framework overwrites the action-model geometry in its constructor,
so the expert is identical regardless of what the yaml says.

## 1. Environments

Tested with Python 3.10, torch 2.6.0 (cu124), transformers 4.57.1, DeepSpeed 0.16.9.

**Main env (Qwen3-VL, Wan2.2, Rex-Omni, PaliGemma, LocateAnything):**

```bash
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools ninja packaging psutil
.venv/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements.txt
# FlashAttention-2 (used by the Qwen3-VL and Rex-Omni launch presets; PaliGemma / LocateAnything use SDPA):
.venv/bin/pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# or build it: MAX_JOBS=32 .venv/bin/pip install --no-build-isolation flash-attn==2.7.1.post4
```

The launch script uses `VENV_DIR` (default `<repo>/.venv`) and calls
`accelerate` from it directly, so no activation is needed.

**Cosmos env (`COSMOS_VENV_DIR`)**: Python 3.12, torch 2.7.1 (cu128),
transformers 4.51.3 pinned by cosmos-oss, plus transformer-engine and a checkout
of cosmos-predict2.5 v1.5.2 (`COSMOS_SOURCE`). Build it with the
`install_cosmos_predict25.sh` recipe from the OpenWAM side of this comparison, or
follow the cosmos-predict2.5 README; then `pip install -r requirements.txt`
without the torch line.

**RynnBrain env (`RYNN_VENV_DIR`)**: same as the main env with
`transformers>=5.12`; RynnBrain-2B was trained there. Details and the reasons
for the three environments are in `docs/BACKBONE_ENVIRONMENTS.md`.

## 2. Weights and data

Weights are not included. Defaults expect:

```
/path/to/backbones/VLM/{Qwen3-VL-4B, Rex-Omni-3B, Paligemma-3B, LocateAnything-3B, RynnBrain-2B}
/path/to/backbones/WAM/{Wan2.2-TI2V-5B-Diffusers, Cosmos-Predict2.5-2B, Cosmos-Reason1-7B}
```

Override with `BASE_MODEL` (and `COSMOS_TEXT_ENCODER`, `COSMOS_SOURCE` for Cosmos).

Data: the LeRobot root of `PhysicalAI-Robotics-GR00T-Teleop-Sim`
(`DATA_ROOT`), mixture `gr1_multi_concat` (`starVLA/dataloader/gr00t_lerobot/mixtures.py`),
robot type `fourier_gr1_arms_waist_concat_starvla`: ego view resized to 224×224,
29-D actions, 58-D state, horizon 16, min-max action normalization.
`datasets.vla_data.data_percent` selects a deterministic per-task episode subset
(seed 42 by default); the 25 % subset is a prefix of the 50 % subset, and so on.

## 3. Training

```bash
export WANDB_API_KEY=...                       # or WANDB_MODE=disabled
BACKBONE=qwen3 DATA_PERCENT=50 \
DATA_ROOT=/path/to/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot \
BASE_MODEL=/path/to/backbones/VLM/Qwen3-VL-4B \
bash scripts/run_scripts_vlm_weight/run_robocasa_pi_backbone.sh
```

`BACKBONE` ∈ `qwen3 | wan | cosmos | rex | paligemma | locate | rynn2`;
`DATA_PERCENT` ∈ `25 | 50 | 75 | 100`. The script fixes the protocol and derives
gradient accumulation from the GPU count (`NUM_GPUS`, `NUM_MACHINES`,
`MACHINE_RANK`, `MAIN_PROCESS_IP` for multi-node):

- global batch 256; steps ∝ data: 10k / 20k / 30k / 40k, warmup 5 %, cosine;
- backbone LR 1e-5, conditioner + action expert LR 1e-4, AdamW;
- DeepSpeed ZeRO-2, bf16 backbone, fp32 action expert;
- `repeated_diffusion_steps 2`, `num_inference_timesteps 4`.

`DRY_RUN=1` prints the resolved configuration without launching. Runs land in
`results/data_efficiency/<backbone>/<run_id>/` (`OUTPUT_ROOT` to change) with
`config.yaml`, `dataset_statistics.json`, `checkpoints/steps_*_pytorch_model.pt`
and `final_model/pytorch_model.pt`.

The GR1 full-shot / 100-shot recipe for Qwen3-PI and WanPI is in
`REPRODUCE_PI_FULLSHOT.md` (`scripts/run_scripts_vlm_weight/run_gr1_pi_backbones_local.sh`).

## 4. Testing

**Smoke training (one GPU, a few steps):**

```bash
WANDB_MODE=disabled BACKBONE=qwen3 DATA_PERCENT=25 GLOBAL_BATCH_SIZE=8 NUM_GPUS=1 \
MAX_TRAIN_STEPS=5 NUM_WARMUP_STEPS=1 SAVE_INTERVAL=100000 EVAL_INTERVAL=100000 \
DATA_ROOT=... BASE_MODEL=... OUTPUT_ROOT=/tmp/pi_smoke \
bash scripts/run_scripts_vlm_weight/run_robocasa_pi_backbone.sh
```

**Open-loop check / serving:** a trained run directory loads with
`baseframework.from_pretrained(run_dir)`; `predict_action(batch_images,
instructions, state)` returns normalized action chunks and `get_action(data)`
implements the StarVLA evaluation contract. To serve a checkpoint over WebSocket
and exercise the transport end to end:

```bash
python deployment/model_server/server_policy.py --ckpt_path results/.../final_model/pytorch_model.pt --port 10093
python deployment/model_server/debug_server_policy.py --host 127.0.0.1 --port 10093 --test infer
```

**Closed-loop RoboCasa evaluation** runs in the RoboCasa / GR00T simulation
harness against a policy that implements `get_action(data)` (batched video,
language annotation, normalized state in; normalized action chunk out). The
simulator harness is not part of this repository.

## Caveats recorded for the released runs

- PaliGemma must use `attn_implementation=sdpa`; under `flash_attention_2` HF runs
  Gemma attention causally and drops the prefix-LM mask. The adapter warns but
  keeps a configured FA2 so older checkpoints evaluate as trained.
- The Cosmos interface feeds the observation with `condition_video_input_mask`
  all zeros, unlike the OpenWAM Cosmos adapter (observation frames masked 1).
- Per-GPU batch ceilings: Cosmos and LocateAnything 8, Rex-Omni 16, others 32.

## Acknowledgements

- [StarVLA](https://github.com/starVLA/starVLA): this repository is a trimmed
  derivative of the StarVLA codebase (trainer, framework base classes, Qwen3-VL
  interface, model server, layerwise flow-matching head).
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T): the LeRobot
  dataloader under `starVLA/dataloader/gr00t_lerobot/` and the flow-matching head
  modules under `modules/action_model/flow_matching_head/` derive from GR00T N1.5.
- [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM): the OpenWAM side of this
  comparison; the Cosmos interface here mirrors OpenWAM's Cosmos adapter.
- [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) (NVIDIA),
  [Wan2.2](https://github.com/Wan-Video/Wan2.2) via Diffusers, Hugging Face
  `transformers` for the VLM backbones, and the RoboCasa GR1 dataset
  (`nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim`).

## License and redistribution

- Code: MIT License (`LICENSE`). Modifications made for the Fixed-16 comparison
  are released under the same terms.
- Third-party components (Apache-2.0, MIT and BSD-3-Clause parts) and the
  licenses of the model weights and datasets this code expects are listed in
  `THIRD_PARTY_NOTICES.md`; the full license texts are in `licenses/`.
- No model weights or datasets are distributed with this code; checkpoints
  fine-tuned from third-party weights inherit those weights' terms.
- When redistributing, keep `LICENSE`, `THIRD_PARTY_NOTICES.md`, `licenses/` and the SPDX / copyright headers in the source files.
