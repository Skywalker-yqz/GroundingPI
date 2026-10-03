"""Wan2.2-TI2V clean-feature interface, adapted from upstream StarVLA WM4A."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class Wan2Interface(nn.Module):
    """UMT5 + Wan VAE + Wan DiT interface for a single clean observation."""

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        wm_cfg = config.framework.world_model
        model_path = wm_cfg.base_wm

        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from diffusers.video_processor import VideoProcessor
        from transformers import T5TokenizerFast, UMT5EncoderModel

        self.tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=int(self.vae.config.scale_factor_spatial)
        )

        # These encoders are preprocessing modules in the StarVLA WM4A path.
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return int(
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )

    @property
    def num_layers(self) -> int:
        return len(self.transformer.blocks)

    def encode_text(self, instructions: list[str], max_length: int = 512) -> torch.Tensor:
        device = next(self.text_encoder.parameters()).device
        tokens = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)
        lengths = tokens.attention_mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            encoded = self.text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            ).last_hidden_state
        valid = [features[:length] for features, length in zip(encoded, lengths)]
        return torch.stack(
            [
                torch.cat(
                    [features, features.new_zeros(max_length - features.shape[0], features.shape[1])]
                )
                for features in valid
            ]
        ).to(dtype=torch.bfloat16)

    def encode_images(self, images) -> torch.Tensor:
        device = next(self.vae.parameters()).device
        videos = []
        for sample_images in images:
            if not isinstance(sample_images, (list, tuple)):
                sample_images = [sample_images]
            # Use one square observation resolution for the VLA comparison.
            video = self.video_processor.preprocess_video(
                sample_images, height=384, width=320
            ).to(device=device, dtype=self.vae.dtype)
            videos.append(video.squeeze(0))
        video = torch.stack(videos)
        with torch.no_grad():
            latents = self.vae.encode(video).latent_dist.sample()
        mean = torch.as_tensor(
            self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        inv_std = torch.as_tensor(
            self.vae.config.latents_std, device=latents.device, dtype=latents.dtype
        ).reciprocal().view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latents - mean) * inv_std

    def build_inputs(self, images, instructions) -> dict[str, torch.Tensor]:
        text = self.encode_text(instructions)
        latents = self.encode_images(images)
        patch_t, patch_h, patch_w = self.transformer.config.patch_size
        _, _, frames, height, width = latents.shape
        sequence = (frames // patch_t) * (height // patch_h) * (width // patch_w)
        if sequence > 1024:
            raise ValueError(f"Wan token sequence {sequence} exceeds RoPE limit 1024")
        timestep = torch.zeros(
            latents.shape[0], sequence, device=latents.device, dtype=torch.long
        )
        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text,
        }

    def forward(self, **inputs):
        return self.transformer(**inputs)


__all__ = ["Wan2Interface"]
