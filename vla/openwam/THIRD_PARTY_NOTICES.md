# Third-party notices

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
