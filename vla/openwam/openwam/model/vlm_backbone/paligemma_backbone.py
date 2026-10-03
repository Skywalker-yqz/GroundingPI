"""PaliGemma backbone (SigLIP-400M vision tower + Gemma language stack).

Two things differ from the Qwen family and both are handled here:

- **No chat template.** PaliGemma's processor takes the raw instruction and
  prepends the image tokens itself, so :meth:`format_prompt` must not wrap the
  prompt in a conversation — doing so would feed literal template markup to a
  model that never saw any.
- **``token_type_ids``.** The processor emits it to separate image tokens from
  text tokens, and the model's forward consumes it. It is token-aligned, so the
  base's collate pads it alongside ``input_ids`` rather than plain-concatenating.

PaliGemma-3B is 18 blocks / hidden 2048 — the shallowest backbone in the
comparison, so its 8 depth taps land 2-3 blocks apart where a 36-block VLM's land
5 apart. That is exactly what normalized-depth sampling is for, but it is worth
stating in the writeup rather than leaving implicit.
"""

from __future__ import annotations

import logging

import torch

from openwam.model.vlm_backbone.hf_vlm_base import HFVlmBackbone

logger = logging.getLogger(__name__)


class PaliGemmaBackbone(HFVlmBackbone):
    """PaliGemma through the OpenWAM VLM contract."""

    EXTRA_TENSOR_KEYS = HFVlmBackbone.EXTRA_TENSOR_KEYS + ("token_type_ids",)

    def load_backbone(self, checkpoint_path: str, *, dtype: torch.dtype, load_pretrained: bool):
        try:
            from transformers import AutoConfig, AutoProcessor, PaliGemmaForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "PaliGemma requires transformers with PaliGemmaForConditionalGeneration. "
                "Install via `pip install -U transformers`."
            ) from e

        processor = None
        if checkpoint_path:
            try:
                processor = AutoProcessor.from_pretrained(checkpoint_path)
            except Exception:
                if load_pretrained:
                    raise
        # No ``trust_remote_code``: the class comes from upstream transformers.
        # google/paligemma-* is a gated repo — accept the license on the Hub and
        # authenticate before the first load, or from_pretrained will 401.
        if load_pretrained:
            model = PaliGemmaForConditionalGeneration.from_pretrained(checkpoint_path, dtype=dtype)
        else:
            cfg = AutoConfig.from_pretrained(checkpoint_path)
            model = PaliGemmaForConditionalGeneration(cfg).to(dtype=dtype)
        return model, processor

    def format_prompt(self, prompt: str) -> str:
        """PaliGemma takes the bare instruction; the processor adds image tokens."""
        return prompt


__all__ = ["PaliGemmaBackbone"]
