# Fixed-16 Backbone Comparison — OpenWAM side

> Part of the **GroundingPI** release. The unified environment / training / testing guide is [`../README.md`](../README.md); this file holds the per-backbone details of the RoboTwin side.

Which representation is the better foundation for robot control, a vision-language
model (VLM) or a video world model? This repository is the OpenWAM side of that
comparison: seven pretrained backbones are attached to one **byte-identical**
Action Expert and fine-tuned end-to-end on RoboTwin 2.0 under one protocol, so
the backbone is the only variable.

The Action Expert is a Fixed-16 π-style layerwise Action DiT (16 blocks = 8
cross-attention + 8 self-attention, width 1024, 16 heads × 64, fp32 compute). It
reads eight hidden states tapped at normalized depths `ℓ_j = round(j·(N−1)/7)` of
whichever backbone it is attached to.

| backbone | side | launch name | layers / hidden |
|---|---|---|---|
| Wan2.2-TI2V-5B | video | `wan22_ti2v_5b` | 30 / 3072 |
| Cosmos-Predict2.5-2B | video | `cosmos_predict25` | 28 / 2048 |
| Qwen3-VL-4B | VLM | `qwen3_vl_4b` | 36 / 2560 |
| PaliGemma-3B | VLM | `paligemma_3b` | 18 / 2048 |
| Rex-Omni-3B (Qwen2.5-VL) | VLM | `rex_omni_3b` | 36 / 2048 |
| LocateAnything-3B | VLM | `locate_anything_3b` | 36 / 2048 |
| RynnBrain-2B (Qwen3-VL) | VLM | `rynnbrain_2b` | 28 / 2048 |

The code is a trimmed derivative of [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM):
only the two Fixed-16 architectures, the seven backbone adapters, the RoboTwin
dataloader, the trainer, the policy server and the RoboTwin evaluation client are kept.

## 1. Environments

Tested with Python 3.12, CUDA 12.8 drivers and torch 2.7.1. Two virtual
environments are required because Cosmos pins a different `transformers`.

**`vlm` env (five VLMs + Wan2.2):**

```bash
python3.12 -m venv /path/to/envs/vlm
/path/to/envs/vlm/bin/pip install --upgrade pip
/path/to/envs/vlm/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
/path/to/envs/vlm/bin/pip install -e '.[dev]'                   # openwam + pytest/ruff
/path/to/envs/vlm/bin/pip install -r envs/vlm-requirements.txt   # transformers==4.57.0, decord, lmdb
```

`torchrun` is installed into the venv by torch. If you build the venv with
`--system-site-packages` instead, add the shim described in
`envs/vlm-requirements.txt` so the venv interpreter is the one launched.
`flash-attn` is optional: the Action Expert and the VLMs fall back to SDPA.

**`cosmos` env (Cosmos-Predict2.5 only):**

```bash
python3.12 -m venv /path/to/envs/cosmos
/path/to/envs/cosmos/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
/path/to/envs/cosmos/bin/pip install -e .
git submodule update --init third_party/cosmos-predict2.5          # tag v1.5.2
PYBIN=/path/to/envs/cosmos/bin/python bash scripts/install_cosmos_predict25.sh   # cosmos-oss (transformers==4.51.3) + transformer-engine
```

`scripts/install_cosmos_predict25.sh` needs a CUDA toolkit with cuDNN headers to
compile transformer-engine. `envs/vlm.env` and `envs/cosmos.env` hold the runtime
variables each environment needs (notably `unset PYTORCH_CUDA_ALLOC_CONF`, which
otherwise turned the loss to NaN under ZeRO-2 + gradient checkpointing, and the
cuDNN library order for transformer-engine). Do not merge the two environments:
Rex-Omni's tokenizer resolves wrong token ids on transformers 5.x and the adapter
refuses to run.

## 2. Weights and data

Point the yaml fields (or CLI overrides) at the downloaded weights. Defaults expect:

```
/path/to/backbones/VLM/{Qwen3-VL-4B, Paligemma-3B, Rex-Omni-3B, LocateAnything-3B, RynnBrain-2B}
/path/to/backbones/WAM/{Wan2.2-TI2V-5B, Cosmos-Predict2.5-2B, Cosmos-Reason1-7B}
```

LocateAnything ships its own modeling code and loads with `trust_remote_code`
(`allow_remote_code: true` in its yaml); only point it at a checkpoint you trust.

Data: RoboTwin 2.0 demonstrations at `/path/to/RoboTwin2.0/dataset` (layout
`<task>/aloha-agilex_clean_50/data/*.hdf5`), variant `clean_50`, 50 tasks, 20-D
end-effector actions. Normalization statistics are computed on first load.

## 3. Training

One launch path for every backbone:

```bash
export OPENWAM_ENV=/path/to/envs/vlm          # cosmos_predict25 uses the cosmos env
BATCH_SIZE=32 bash scripts/train_fixed16.sh qwen3_vl_4b \
    DATASET_DIR=/path/to/RoboTwin2.0/dataset \
    training.output_path=/path/to/checkpoints

BATCH_SIZE=48 bash scripts/train_fixed16.sh wan22_ti2v_5b
BATCH_SIZE=8  OPENWAM_ENV=/path/to/envs/cosmos bash scripts/train_fixed16.sh cosmos_predict25
```

`DATASET_DIR`, `OUTPUT_PATH`, `EPOCHS`, `GLOBAL_BATCH` and `SAVE_STEPS` are
environment variables; anything after the backbone name is passed to Hydra.
The script fixes everything that must match across backbones and refuses to
start if the per-GPU batch does not divide the global batch:

- global batch 256 (gradient accumulation derived from `BATCH_SIZE × GPUs × nodes`),
  10 epochs, cosine schedule with 5 % warmup;
- backbone LR 1e-5, conditioner and Action Expert LR 1e-4;
- `lambda_video = 0`, `unify_action=false`, `action_dim = state_dim = 20`;
- fp32 Action Expert.

Multi-node: run the same command on every node with `NNODES`, `NODE_RANK` and
`MASTER_ADDR` set (schedulers that export `HOST_NUM` / `HOST_GPU_NUM` / `RANK`
are picked up automatically).

Per-GPU batch ceilings measured on 8×H200: Wan 48, Cosmos 8, Qwen3-VL 32,
PaliGemma 32, Rex-Omni 16, LocateAnything 8, RynnBrain 16.

Checkpoints land in `<output_path>/<timestamp>/` as `checkpoint_step_*.safetensors`
plus `config.yaml` and `normalization_stats.npy`; that directory is what the
policy server consumes.

## 4. Testing

**Unit tests (CPU, no weights needed):**

```bash
make test        # pytest -m "not gpu": Action Expert geometry, two-side equivalence, trainer, deploy paths
```

**Policy server smoke (GPU, one checkpoint):**

```bash
bash scripts/deploy.sh /path/to/checkpoints/<run> --port 8848
python scripts/inference_single_test.py --test --server ws://127.0.0.1:8848
```

The client pings the server, sends three random images plus a dummy state, and
prints the returned 20-D action.

**RoboTwin closed-loop evaluation:** install RoboTwin 2.0 in its own conda
environment, start the policy server as above, then from `benchmarks/robotwin/`:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
bash single_eval.sh adjust_bottle demo_clean groundingpi 0 8848 127.0.0.1      # one task
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/checkpoints/<run> all  # all 50 tasks
```

`benchmarks/robotwin/README.md` documents the episode-level parallel evaluator
(`parallel_eval.sh` + `scripts/deploy_multi.sh`), the result exporter and the
web dashboard. `policy_config.yml` must match the checkpoint's `action_mode`
(`ee` / `state_dim: 20` for the shipped configs).

## 5. Layout

```
openwam/model/action_backbone/fixed16_pi_action_dit.py   # the shared Action Expert
openwam/model/action_backbone/backbone_conditioner.py    # taps -> projector -> resampler -> depth embedding
openwam/model/architectures/{fixed16_pi_base,dual_system,vlm_system}  # harness + the two sides
openwam/model/vlm_backbone/                              # HF VLM adapters
openwam/model/video_backbone/                            # Wan2.2 and Cosmos-Predict2.5 adapters
openwam/dataloader/robotwin.py                           # RoboTwin 2.0 reader
openwam/train/, openwam/deploy/                          # trainer, policy server
configs/experiment/fixed16_comparison.yaml               # the protocol
configs/model/{dual_system_fixed16,vlm_system}.yaml      # the two sides; keep in sync
scripts/train_fixed16.sh, scripts/deploy.sh              # entry points
benchmarks/robotwin/                                     # evaluation client
```

## Acknowledgements

- [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM): this repository is a
  trimmed derivative of the OpenWAM codebase (trainer, deploy server, RoboTwin
  client, Wan/Cosmos adapters).
- [StarVLA](https://github.com/starVLA/starVLA): the PI-style layerwise action
  expert and the `condition_pathway: starvla` option follow StarVLA's design;
  the StarVLA side of this comparison lives in the sibling repository.
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) and
  [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), from which
  the Wan backbone code under `openwam/model/video_backbone/wan/` is derived.
- [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) (NVIDIA),
  used as a git submodule by the Cosmos adapter.
- [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) for the benchmark
  and demonstrations; Hugging Face `transformers` for the VLM backbones.

## License and redistribution

- Code: MIT License (`LICENSE`). Modifications made for the Fixed-16 comparison
  are released under the same terms.
- Third-party components (Apache-2.0, MIT and BSD-3-Clause parts) and the
  licenses of the model weights and datasets this code expects are listed in
  `THIRD_PARTY_NOTICES.md`; the full license texts are in `licenses/`.
- No model weights or datasets are distributed with this code; checkpoints
  fine-tuned from third-party weights inherit those weights' terms.
- When redistributing, keep `LICENSE`, `THIRD_PARTY_NOTICES.md`, `licenses/` and `openwam/model/video_backbone/wan/license/`.
