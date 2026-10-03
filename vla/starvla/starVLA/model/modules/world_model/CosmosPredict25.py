"""StarVLA-native interface for NVIDIA Cosmos-Predict2.5-2B.

This module calls the official ``cosmos_predict2`` implementation directly.
It intentionally has no dependency on OpenWAM model or training classes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn


_SYSTEM_PROMPT = "You are a helpful assistant who will provide prompts to an image generator."
_NET_KWARGS = dict(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    crossattn_emb_channels=1024,
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="transformer_engine",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False,
)


def _tokenize_reason1(tokenizer, prompt: str, length: int = 512) -> list[int]:
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": str(prompt)}]},
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    ids = ids[:length]
    return ids + [int(tokenizer.pad_token_id)] * (length - len(ids))


class CosmosPredict25Interface(nn.Module):
    """Reason1 + Wan tokenizer + Cosmos DiT feature extractor."""

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        wm_cfg = config.framework.world_model
        model_path = Path(str(wm_cfg.base_wm))
        reason_path = Path(str(wm_cfg.text_encoder))
        variant = str(wm_cfg.get("model_variant", "base/post-trained"))
        upstream = str(wm_cfg.get("cosmos_source", ""))
        if upstream and upstream not in os.sys.path:
            os.sys.path.insert(0, upstream)

        from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import CheckpointMode, SACConfig

        self.transformer = MinimalV1LVGDiT(
            sac_config=SACConfig(mode=CheckpointMode.NONE), **_NET_KWARGS
        )
        matches = sorted((model_path / variant).glob("*_ema_bf16.pt"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one Cosmos *_ema_bf16.pt under {model_path / variant}, got {matches}"
            )
        raw = torch.load(matches[0], map_location="cpu", weights_only=False)
        state = {key.removeprefix("net."): value for key, value in raw.items() if key.startswith("net.")}
        missing, unexpected = self.transformer.load_state_dict(state, strict=False)
        real_missing = [key for key in missing if not key.endswith("._extra_state")]
        if real_missing or unexpected:
            raise RuntimeError(
                f"Cosmos checkpoint mismatch: {real_missing} missing, {unexpected} unexpected"
            )
        self.transformer.to(dtype=torch.bfloat16)

        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        self.reason_tokenizer = AutoTokenizer.from_pretrained(str(reason_path), trust_remote_code=True)
        if self.reason_tokenizer.pad_token_id is None:
            self.reason_tokenizer.pad_token_id = self.reason_tokenizer.eos_token_id
        self.reason1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(reason_path), torch_dtype=torch.bfloat16
        ).eval()
        self.reason1.requires_grad_(False)

        from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        vae_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self._vae_iface = Wan2pt1VAEInterface(
            vae_pth=str(model_path / "tokenizer.pth"),
            s3_credential_path="",
            temporal_window=4,
            is_parallel=False,
            load_mean_std=False,
        )
        inner = self._vae_iface.model.model
        inner.to(device=vae_device, dtype=torch.bfloat16)
        inner.requires_grad_(False)
        self.vae = inner
        self._sync_vae(vae_device)

    @property
    def hidden_size(self) -> int:
        return 2048

    @property
    def num_layers(self) -> int:
        return 28

    def _sync_vae(self, device: torch.device) -> None:
        outer = self._vae_iface.model
        outer.device = device
        outer.dtype = torch.bfloat16
        for name in ("mean", "std", "img_mean", "img_std", "video_mean", "video_std"):
            value = getattr(outer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(outer, name, value.to(device=device, dtype=torch.bfloat16))
        if hasattr(outer, "mean") and hasattr(outer, "std"):
            outer.scale = [outer.mean, 1.0 / outer.std]

    def encode_images(self, images) -> torch.Tensor:
        from PIL import Image

        clips = []
        for sample in images:
            sample = sample if isinstance(sample, (list, tuple)) else [sample]
            frames = []
            for image in sample:
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(np.asarray(image))
                image = image.convert("RGB").resize((320, 384))
                frames.append(np.asarray(image, dtype=np.uint8))
            clips.append(np.stack(frames))
        video = torch.from_numpy(np.stack(clips)).float().div_(127.5).sub_(1.0)
        video = video.permute(0, 4, 1, 2, 3).contiguous()
        device = next(self.vae.parameters()).device
        self._sync_vae(device)
        with torch.no_grad():
            return self._vae_iface.encode(video.to(device=device, dtype=torch.bfloat16))

    def encode_text(self, prompts: Iterable[str]) -> torch.Tensor:
        prompts = list(prompts)
        device = next(self.reason1.parameters()).device
        ids = torch.tensor(
            [_tokenize_reason1(self.reason_tokenizer, prompt) for prompt in prompts],
            dtype=torch.long,
            device=device,
        )
        mask = ids.ne(int(self.reason_tokenizer.pad_token_id))
        with torch.no_grad():
            outputs = self.reason1(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = []
        for value in outputs.hidden_states[1:]:
            value = (value - value.mean(dim=-1, keepdim=True)) / (
                value.std(dim=-1, keepdim=True) + 1e-8
            )
            hidden.append(value)
        context = torch.cat(hidden, dim=-1)
        if context.shape[-1] != 100352:
            raise RuntimeError(f"Reason1 context width must be 100352, got {context.shape[-1]}")
        return context.to(dtype=torch.bfloat16)

    def extract_features(self, images, instructions, tap_indices: list[int]):
        latents = self.encode_images(images)
        context = self.encode_text(instructions).to(device=latents.device)
        batch, _, frames, height, width = latents.shape
        condition_mask = torch.zeros(
            batch, 1, frames, height, width, device=latents.device, dtype=latents.dtype
        )
        padding_mask = torch.zeros(
            batch, 1, height, width, device=latents.device, dtype=latents.dtype
        )
        timestep = torch.zeros(batch, device=latents.device, dtype=latents.dtype)
        _, taps = self.transformer(
            x_B_C_T_H_W=latents,
            timesteps_B_T=timestep,
            crossattn_emb=context,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            padding_mask=padding_mask,
            intermediate_feature_ids=tap_indices,
        )
        return taps


__all__ = ["CosmosPredict25Interface"]
