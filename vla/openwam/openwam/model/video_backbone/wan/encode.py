"""Wan VAE / text-encoder IO with native-VAE ↔ external-encoder routing.

Free functions — ``vae`` / ``encoder`` / ``tokenizer`` / ``text_encoder`` passed
explicitly so the backbone class holds no IO logic. ``encoder is not None``
selects the external-encoder path; otherwise the native Wan VAE is used.
"""

from __future__ import annotations

from typing import Tuple

from torch import Tensor

from openwam.model.video_backbone.wan.preprocess import preprocess_video as _preprocess_video_native


def encode_text(prompts: list, *, tokenizer, text_encoder, device) -> Tuple[Tensor, Tensor]:
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[i, v:] = 0
    return context, seq_lens


def preprocess_video(frames, *, encoder=None, dtype, device) -> Tensor:
    if encoder is not None:
        return encoder.preprocess_video(frames)
    return _preprocess_video_native(frames, dtype=dtype, device=device)


def encode_video(video_tensor: Tensor, *, vae, encoder=None) -> Tensor:
    if encoder is not None:
        return encoder.batch_encode(video_tensor)
    return vae.batch_encode(video_tensor, device=video_tensor.device)


