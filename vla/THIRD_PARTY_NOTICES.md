# Third-party notices (GroundingPI)

The repository is MIT-licensed (see `LICENSE`). The two sub-projects list their third-party components below; paths are relative to each sub-project. Full license texts: `openwam/licenses/`, `openwam/openwam/model/video_backbone/wan/license/`, `starvla/licenses/`.

---

# openwam/ (derived from OpenWAM)


This repository is released under the MIT License (see `LICENSE`). It contains
or depends on the third-party components listed below, which keep their own
licenses. Full license texts are in `licenses/` and in the locations named here.

## Code included in this repository

| Component | Where | License | Source |
|---|---|---|---|
| OpenWAM | whole repository (trainer, deploy server, RoboTwin client, backbone adapters) | MIT, Copyright (c) OpenWAM Team | https://github.com/OpenWAM-Official/OpenWAM |
| DiffSynth-Studio / Wan2.2 model code | `openwam/model/video_backbone/wan/` (DiT, VAE, T5 text encoder, attention, loaders, converters) | Apache License 2.0 — text and attribution in `openwam/model/video_backbone/wan/license/` | https://github.com/modelscope/DiffSynth-Studio, https://github.com/Wan-Video/Wan2.2 |
| CameraCtrl | helper functions in `openwam/model/video_backbone/wan/camera_controller.py` (marked "Copied from") | Apache License 2.0 | https://github.com/hehao13/CameraCtrl |

## Code referenced but not vendored

| Component | How it is used | License | Source |
|---|---|---|---|
| Cosmos-Predict2.5 (`cosmos_predict2`) | git submodule at `third_party/cosmos-predict2.5` (tag v1.5.2), imported by the Cosmos adapter | Apache License 2.0 | https://github.com/nvidia-cosmos/cosmos-predict2.5 |
| Hugging Face `transformers` | loads every VLM backbone (Qwen3-VL, Qwen2.5-VL, PaliGemma, LocateAnything remote code, RynnBrain) | Apache License 2.0 | https://github.com/huggingface/transformers |
| RoboTwin 2.0 | `benchmarks/robotwin/` drives RoboTwin's own `eval_policy.py` inside the RoboTwin environment; no RoboTwin code is copied here | MIT | https://github.com/RoboTwin-Platform/RoboTwin |
| PyTorch, DeepSpeed, Accelerate, Hydra, and the other packages in `pyproject.toml` | runtime dependencies | their respective licenses | PyPI |

## Model weights and data (not distributed here)

The code downloads or expects the following assets, each subject to its own
license or terms of use. Fine-tuned checkpoints inherit those terms.

| Asset | Terms |
|---|---|
| Wan2.2-TI2V-5B | Apache License 2.0 (Wan-AI) |
| Cosmos-Predict2.5-2B, Cosmos-Reason1-7B | NVIDIA Open Model License Agreement |
| Qwen3-VL-4B-Instruct | Apache License 2.0 (Qwen Team) |
| PaliGemma-3B | Gemma Terms of Use (Google) |
| Rex-Omni-3B, LocateAnything-3B, RynnBrain-2B | see the license stated on each model card |
| RoboTwin 2.0 demonstrations | RoboTwin 2.0 dataset terms |

## Redistribution

Keep `LICENSE`, this file, `licenses/` and
`openwam/model/video_backbone/wan/license/` when redistributing the code, in
whole or in part.

---

# starvla/ (derived from StarVLA)


This repository is released under the MIT License (see `LICENSE`). It contains
or depends on the third-party components listed below, which keep their own
licenses. Full license texts are in `licenses/`.

## Code included in this repository

| Component | Where | License | Source |
|---|---|---|---|
| StarVLA | whole repository (trainer, framework base classes, Qwen3-VL interface, PI frameworks, model server) | MIT, Copyright (c) StarVLA Team; file headers read "Copyright 2025 starVLA community" | https://github.com/starVLA/starVLA |
| NVIDIA Isaac GR00T N1.5 | `starVLA/dataloader/gr00t_lerobot/` (LeRobot dataset, transforms, schema, embodiment tags), `starVLA/model/modules/action_model/flow_matching_head/`, and `LayerwiseFM_ActionHeader.py` (NVIDIA code modified by StarVLA) | Apache License 2.0, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES — SPDX headers kept in each file | https://github.com/NVIDIA/Isaac-GR00T |
| Hugging Face `transformers` — Qwen3-VL modeling | `starVLA/model/modules/vlm/modeling_qwen3_vl.py` (vendored copy) | Apache License 2.0, Copyright 2025 The Qwen Team and The HuggingFace Inc. team | https://github.com/huggingface/transformers |
| Diffusion Policy | rotation-representation helper in `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` (marked "Adapted from") | MIT, Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab | https://github.com/real-stanford/diffusion_policy |
| OpenVLA / Prismatic | `starVLA/training/trainer_utils/overwatch.py` (logging utility) | MIT | https://github.com/openvla/openvla |
| msgpack-numpy | `deployment/model_server/tools/msgpack_numpy.py` (marked "adapted from") | BSD 3-Clause, Copyright (c) Lev E. Givon | https://github.com/lebedov/msgpack-numpy |

## Code referenced but not vendored

| Component | How it is used | License | Source |
|---|---|---|---|
| Cosmos-Predict2.5 (`cosmos_predict2`) | imported by `starVLA/model/modules/world_model/CosmosPredict25.py`; point `COSMOS_SOURCE` at a checkout of tag v1.5.2 | Apache License 2.0 | https://github.com/nvidia-cosmos/cosmos-predict2.5 |
| Diffusers (Wan2.2 pipeline components) | `starVLA/model/modules/world_model/Wan2.py` loads `WanTransformer3DModel` / `AutoencoderKLWan` | Apache License 2.0 | https://github.com/huggingface/diffusers |
| Hugging Face `transformers` | loads every VLM backbone | Apache License 2.0 | https://github.com/huggingface/transformers |
| PyTorch, DeepSpeed, Accelerate, and the other packages in `requirements.txt` | runtime dependencies | their respective licenses | PyPI |

## Model weights and data (not distributed here)

| Asset | Terms |
|---|---|
| Wan2.2-TI2V-5B (Diffusers layout) | Apache License 2.0 (Wan-AI) |
| Cosmos-Predict2.5-2B, Cosmos-Reason1-7B | NVIDIA Open Model License Agreement |
| Qwen3-VL-4B-Instruct | Apache License 2.0 (Qwen Team) |
| PaliGemma-3B | Gemma Terms of Use (Google) |
| Rex-Omni-3B, LocateAnything-3B, RynnBrain-2B | see the license stated on each model card |
| nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim (RoboCasa GR1) | dataset license stated on the Hugging Face dataset card |

## Redistribution

Keep `LICENSE`, this file, `licenses/` and the SPDX / copyright headers inside
the source files when redistributing the code, in whole or in part.

## Attribution and anonymous review

Third-party attribution for anonymous review

Names of upstream authors, companies and institutions, contact addresses,
repository namespaces, public model/dataset identifiers and example URLs in
third-party source and license notices identify their original sources. They
are retained for attribution and interoperability, not as a declaration of
this submission's authorship or affiliation. Such attribution alone does not
disclose submission authors. Original licenses and copyright notices remain
unchanged. Local adaptations are recorded separately; an unverified local
remark is not assigned to upstream merely because it is in a dependency.
