# StarVLA backbone environments

Three Python environments were used, because the backbones pin incompatible
`transformers` versions. The launch script selects one per backbone
(`VENV_DIR`, `COSMOS_VENV_DIR`, `RYNN_VENV_DIR`).

## Main environment (`.venv`)

Python 3.10, PyTorch 2.6.0+cu124, Transformers 4.57.1, Accelerate 1.5.2,
DeepSpeed 0.16.9, flash-attn 2.7.1.post4. Build recipe: README section 1.

Backbones: Qwen3-VL-4B (QwenPI), Wan2.2-TI2V-5B (WanPI), Rex-Omni-3B,
PaliGemma-3B, LocateAnything-3B (VLMBackbonePI).

Notes:

- PaliGemma needs `sentencepiece`, and must run with `attn_implementation=sdpa`
  (its prefix-LM mask is dropped on the flash-attention-2 path).
- LocateAnything ships its own modeling code (`trust_remote_code`); the adapter
  gives every rank its own Hugging Face dynamic-module cache.
- Rex-Omni's tokenizer is validated against its checkpoint at load time; keep
  Transformers 4.57.x for it.
- `accelerate` 1.5.2 + DeepSpeed ZeRO-2 rejects `gradient_accumulation_steps>1`
  (`no_sync` incompatible with ZeRO stage 2); use enough GPUs for accumulation 1.

## RynnBrain environment (`.venv-rynn`)

Python 3.12, PyTorch 2.7.1+cu128, Transformers 5.12.1, Accelerate 1.14.0,
DeepSpeed 0.18.9. RynnBrain-2B (a Qwen3-VL derivative) was trained here.

## Cosmos environment (`/path/to/envs/cosmos`)

Python 3.12, PyTorch 2.7.1+cu128, Transformers 4.51.3 (pinned by cosmos-oss),
Accelerate 1.14.0, DeepSpeed 0.18.9, transformer-engine.

Cosmos additionally needs the official `cosmos_predict2` source (pinned to
cosmos-predict2.5 v1.5.2) and the environment's cuDNN library ahead of the
system cuDNN; the launch script sets the validated cuDNN variables. Point the
source at:

```bash
COSMOS_SOURCE=/path/to/third_party/cosmos-predict2.5 BACKBONE=cosmos \
  bash scripts/run_scripts_vlm_weight/run_robocasa_pi_backbone.sh
```

## Weights

Weights are not included. Defaults expect:

- `/path/to/backbones/VLM/{Qwen3-VL-4B, Rex-Omni-3B, Paligemma-3B, LocateAnything-3B, RynnBrain-2B}`
- `/path/to/backbones/WAM/{Wan2.2-TI2V-5B-Diffusers, Cosmos-Predict2.5-2B, Cosmos-Reason1-7B}`

Override with `BASE_MODEL` (and `COSMOS_TEXT_ENCODER` for Reason1).
