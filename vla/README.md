# GroundingPI — VLA backbone comparison (training + evaluation)

GroundingPI asks one question: **which pretrained representation is the better
foundation for robot control, a vision-language model (VLM) or a video world
model (WAM)?** Seven pretrained backbones are attached to one byte-identical
π-style Action Expert (16 DiT blocks, width 1024, 16 heads × 64, eight hidden
states tapped at normalized depths, fp32 flow matching) and fine-tuned
end-to-end under one protocol, so the backbone is the only variable.

| backbone | family | layers / hidden |
|---|---|---|
| Wan2.2-TI2V-5B | video world model | 30 / 3072 |
| Cosmos-Predict2.5-2B | video world model | 28 / 2048 |
| Qwen3-VL-4B | VLM | 36 / 2560 |
| PaliGemma-3B | VLM | 18 / 2048 |
| Rex-Omni-3B (Qwen2.5-VL) | VLM | 36 / 2048 |
| LocateAnything-3B | VLM | 36 / 2048 |
| RynnBrain-2B (Qwen3-VL) | VLM | 28 / 2048 |

The comparison runs on two benchmarks, each with its own code base:

| sub-project | benchmark | data | what it measures |
|---|---|---|---|
| [`openwam/`](openwam/README.md) | RoboTwin 2.0 (bimanual, 50 tasks, 20-D EEF actions) | RoboTwin 2.0 `clean_50` demonstrations | end-to-end fine-tuning, 10 epochs, closed-loop success rate |
| [`starvla/`](starvla/README.md) | RoboCasa GR1 (24 tabletop tasks, 29-D actions) | `nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim` | data efficiency at 25 / 50 / 75 / 100 % of the demonstrations |

`openwam/` is a trimmed derivative of [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM);
`starvla/` is a trimmed derivative of [StarVLA](https://github.com/starVLA/starVLA).
Each keeps only the code on the GroundingPI path and has its own README with
the per-backbone details. This file is the single entry point: read it top to
bottom and you can train and evaluate on both benchmarks.

---

## 1. Environment tutorial

Hardware used: 8 × H200 (143 GB) per node; a single 80 GB GPU is enough for the
smoke tests below. Linux, NVIDIA driver supporting CUDA 12.4+.

Four Python environments are needed in total, because the backbones pin
incompatible `transformers` versions. Build them once, then point the launch
scripts at them.

### 1.1 `openwam/` environments (RoboTwin side)

**A. `envs/vlm` — five VLMs + Wan2.2 (Python 3.12, torch 2.7.1 cu128, transformers 4.57.0)**

```bash
cd vla/openwam
python3.12 -m venv /path/to/envs/vlm
/path/to/envs/vlm/bin/pip install --upgrade pip
/path/to/envs/vlm/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
/path/to/envs/vlm/bin/pip install -e '.[dev]'
/path/to/envs/vlm/bin/pip install -r envs/vlm-requirements.txt
```

**B. `envs/cosmos` — Cosmos-Predict2.5 only (Python 3.12, torch 2.7.1 cu128, transformers 4.51.3)**

```bash
cd vla/openwam
python3.12 -m venv /path/to/envs/cosmos
/path/to/envs/cosmos/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
/path/to/envs/cosmos/bin/pip install -e .
git clone --branch v1.5.2 https://github.com/nvidia-cosmos/cosmos-predict2.5.git third_party/cosmos-predict2.5
PYBIN=/path/to/envs/cosmos/bin/python bash scripts/install_cosmos_predict25.sh   # needs a CUDA toolkit with cuDNN headers
```

`envs/vlm.env` and `envs/cosmos.env` hold the runtime variables each environment
needs; `scripts/train_fixed16.sh` sources the right one automatically. Never
merge the two environments (Rex-Omni's tokenizer breaks on transformers 5.x).

### 1.2 `starvla/` environments (RoboCasa side)

**C. `starvla/.venv` — Qwen3-VL, Wan2.2, Rex-Omni, PaliGemma, LocateAnything (Python 3.10, torch 2.6.0 cu124, transformers 4.57.1)**

```bash
cd vla/starvla
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools ninja packaging psutil
.venv/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

**D. Cosmos for `starvla/`** — reuse environment B: set `COSMOS_VENV_DIR=/path/to/envs/cosmos`,
`COSMOS_SOURCE=vla/openwam/third_party/cosmos-predict2.5`, and run
`/path/to/envs/cosmos/bin/pip install -r vla/starvla/requirements.txt` once
(skip the torch line; transformers stays at 4.51.3).

RynnBrain-2B on the starvla side used a variant of C with `transformers>=5.12`
(`RYNN_VENV_DIR`); see `starvla/docs/BACKBONE_ENVIRONMENTS.md`.

### 1.3 Weights and data

Nothing is bundled. Download the backbones and place (or symlink) them as:

```
/path/to/backbones/VLM/{Qwen3-VL-4B, Paligemma-3B, Rex-Omni-3B, LocateAnything-3B, RynnBrain-2B}
/path/to/backbones/WAM/{Wan2.2-TI2V-5B, Wan2.2-TI2V-5B-Diffusers, Cosmos-Predict2.5-2B, Cosmos-Reason1-7B}
```

`openwam/` reads the original Wan2.2 layout, `starvla/` the Diffusers layout of
the same weights. Any other location works through the yaml fields / `BASE_MODEL`.

Data:

- RoboTwin 2.0 demonstrations → `/path/to/RoboTwin2.0/dataset` (`<task>/aloha-agilex_clean_50/data/*.hdf5`).
- RoboCasa GR1 → the LeRobot root of `nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim`.

For closed-loop evaluation you also need the simulators, each in its own
environment: [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
(conda env `robotwin`) and the RoboCasa / GR00T simulation harness.

---

## 2. Training

### 2.1 RoboTwin side (`openwam/`)

```bash
cd vla/openwam
export OPENWAM_ENV=/path/to/envs/vlm            # the script switches to the cosmos env for cosmos_predict25
BATCH_SIZE=32 DATASET_DIR=/path/to/RoboTwin2.0/dataset OUTPUT_PATH=/path/to/checkpoints \
    bash scripts/train_fixed16.sh qwen3_vl_4b
```

Backbone names: `wan22_ti2v_5b | cosmos_predict25 | qwen3_vl_4b | paligemma_3b | rex_omni_3b | locate_anything_3b | rynnbrain_2b`.
Protocol (fixed by the script): global batch 256, 10 epochs, cosine + 5 % warmup,
backbone LR 1e-5, expert LR 1e-4, `lambda_video=0`, 20-D actions, fp32 expert.
Per-GPU `BATCH_SIZE` ceilings on H200: Wan 48, Cosmos 8, Qwen3-VL 32, PaliGemma 32,
Rex-Omni 16, LocateAnything 8, RynnBrain 16. Multi-node: run the same command on
each node with `NNODES`, `NODE_RANK`, `MASTER_ADDR`. Output:
`<OUTPUT_PATH>/<timestamp>/{checkpoint_step_*.safetensors, config.yaml, normalization_stats.npy}`.

### 2.2 RoboCasa side (`starvla/`)

```bash
cd vla/starvla
export WANDB_API_KEY=...                        # or WANDB_MODE=disabled
BACKBONE=qwen3 DATA_PERCENT=50 \
DATA_ROOT=/path/to/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot \
BASE_MODEL=/path/to/backbones/VLM/Qwen3-VL-4B \
    bash scripts/run_scripts_vlm_weight/run_robocasa_pi_backbone.sh
```

`BACKBONE ∈ qwen3 | wan | cosmos | rex | paligemma | locate | rynn2`,
`DATA_PERCENT ∈ 25 | 50 | 75 | 100`. Protocol: global batch 256, steps ∝ data
(10k / 20k / 30k / 40k), warmup 5 %, backbone LR 1e-5, expert LR 1e-4, ZeRO-2.
`DRY_RUN=1` prints the resolved configuration. Output:
`results/data_efficiency/<backbone>/<run_id>/{config.yaml, dataset_statistics.json, checkpoints/, final_model/pytorch_model.pt}`.

---

## 3. Testing

### 3.1 Quick checks (no simulator)

```bash
# openwam: CPU unit tests (Action Expert geometry, two-side equivalence, trainer, deploy)
cd vla/openwam && PYTHON=/path/to/envs/vlm/bin/python make test

# openwam: serve a checkpoint and query it with random images
export OPENWAM_ENV=/path/to/envs/vlm && . envs/vlm.env
bash scripts/deploy.sh /path/to/checkpoints/<run> --port 8848 &
python scripts/inference_single_test.py --test --server ws://127.0.0.1:8848   # prints a 20-D action

# starvla: 5-step training smoke on one GPU
cd vla/starvla
WANDB_MODE=disabled BACKBONE=qwen3 DATA_PERCENT=25 GLOBAL_BATCH_SIZE=8 NUM_GPUS=1 \
MAX_TRAIN_STEPS=5 NUM_WARMUP_STEPS=1 SAVE_INTERVAL=100000 EVAL_INTERVAL=100000 \
DATA_ROOT=... BASE_MODEL=... OUTPUT_ROOT=/tmp/pi_smoke \
    bash scripts/run_scripts_vlm_weight/run_robocasa_pi_backbone.sh

# starvla: serve the smoke checkpoint and exercise the transport
.venv/bin/python deployment/model_server/server_policy.py --ckpt_path /tmp/pi_smoke/qwen3/*/final_model/pytorch_model.pt --port 10093 &
.venv/bin/python deployment/model_server/debug_server_policy.py --host 127.0.0.1 --port 10093 --device cuda --test infer
```

### 3.2 Closed-loop evaluation

**RoboTwin** (start the policy server as above, then from the RoboTwin conda env):

```bash
cd vla/openwam/benchmarks/robotwin
export ROBOTWIN_PATH=/path/to/RoboTwin
bash single_eval.sh adjust_bottle demo_clean groundingpi 0 8848 127.0.0.1        # one task
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/checkpoints/<run> all       # all 50 tasks
```

`policy_config.yml` must match the checkpoint (`action_type: ee`, `state_dim: 20`
for the shipped configs). `parallel_eval.sh` + `scripts/deploy_multi.sh` run
episodes across GPUs; `export_results_csv.py` and `benchmarks/web_control.py`
summarize results.

**RoboCasa GR1**: load a run directory with `baseframework.from_pretrained(run_dir)`;
the model implements the StarVLA `get_action(data)` contract (batched video,
language annotation, normalized state in; normalized action chunk out), which
the RoboCasa / GR00T simulation harness consumes. The simulator harness itself
is not part of this repository.

---

## 4. Anonymization

This code was extracted from internal research repositories. Before release:

- every internal filesystem path, cluster/scheduler hook, account name,
  credential and W&B entity was removed; paths appear as `/path/to/...`
  placeholders or environment variables (`DATASET_DIR`, `DATA_ROOT`, `BASE_MODEL`,
  `OUTPUT_PATH`, `OUTPUT_ROOT`, `VENV_DIR`, ...);
- all architectures, datasets, benchmarks and tooling not on the GroundingPI
  path were deleted rather than left disabled;
- no model weights, datasets or checkpoints are distributed. Upstream copyright
  and license headers are kept intact.

---

## 5. Repository map

```
vla/
├── README.md                 this file
├── LICENSE                   MIT (GroundingPI authors)
├── THIRD_PARTY_NOTICES.md    every third-party component and asset license
├── openwam/                  RoboTwin side (derived from OpenWAM)
│   ├── openwam/model/action_backbone/fixed16_pi_action_dit.py     the shared Action Expert
│   ├── openwam/model/architectures/{dual_system,vlm_system}/      video side / VLM side
│   ├── openwam/model/{vlm_backbone,video_backbone}/               seven backbone adapters
│   ├── configs/experiment/fixed16_comparison.yaml                 the protocol
│   ├── scripts/train_fixed16.sh, scripts/deploy.sh                entry points
│   └── benchmarks/robotwin/                                       evaluation client
└── starvla/                  RoboCasa side (derived from StarVLA)
    ├── starVLA/model/framework/{QwenPI,WanPI,CosmosPI,VLMBackbonePI}.py
    ├── starVLA/model/modules/projector/PiBackboneConditioner.py
    ├── starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py
    ├── starVLA/dataloader/gr00t_lerobot/                          GR1 LeRobot reader (data_percent subsets)
    ├── scripts/run_scripts_vlm_weight/run_robocasa_pi_backbone.sh  entry point
    └── deployment/model_server/                                   WebSocket policy server
```

---

## Acknowledgements

- [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM) — `openwam/` is a trimmed
  derivative: trainer, policy server, RoboTwin client, Wan and Cosmos adapters.
- [StarVLA](https://github.com/starVLA/starVLA) — `starvla/` is a trimmed
  derivative: trainer, framework base classes, Qwen3-VL interface, model server,
  layerwise flow-matching head; the π-style Action Expert follows StarVLA's design.
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) — LeRobot dataloader
  and flow-matching head modules; [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
  and Cosmos-Reason1.
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) and
  [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — Wan backbone code.
- Hugging Face `transformers` / `diffusers` — VLM and Wan loading;
  [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) and the RoboCasa GR1
  dataset (`nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim`) — benchmarks.
- Qwen3-VL, PaliGemma, Rex-Omni, LocateAnything and RynnBrain — the evaluated backbones.

## License

Code is released under the MIT License (`LICENSE`). `openwam/` and `starvla/`
keep their upstream MIT `LICENSE` files (OpenWAM Team, StarVLA Team). Apache-2.0,
MIT and BSD-3-Clause third-party components are listed with their license texts
in `THIRD_PARTY_NOTICES.md` and the `licenses/` directories. Model weights and
datasets are not distributed and remain under their own terms; checkpoints
fine-tuned from third-party weights inherit those terms. Keep `LICENSE`,
`THIRD_PARTY_NOTICES.md`, the `licenses/` directories and the in-file copyright
headers when redistributing.
