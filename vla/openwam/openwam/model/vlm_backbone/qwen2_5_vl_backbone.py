"""Qwen2.5-VL backbone — and any checkpoint that keeps its architecture.

The reason this exists separately from ``qwen3_vl`` is only the HF class name;
everything downstream is shared. It is what makes grounding-specialised
derivatives usable as evaluation backbones without new code:

- **Rex-Omni-3B** (``IDEA-Research/Rex-Omni``) reports
  ``architectures: ["Qwen2_5_VLForConditionalGeneration"]`` and ships no remote
  code — it re-purposes the final 1000 vocabulary entries as quantised coordinate
  tokens, which changes weights, not structure. Point ``checkpoint_path`` at it.

Note Qwen2.5-VL keeps ``num_hidden_layers`` / ``hidden_size`` at the top level of
its config rather than under ``text_config`` on some releases;
:meth:`HFVlmBackbone.text_config` falls back to the top level, so the depth taps
place themselves either way (3B: 36 blocks / 2048).
"""

from __future__ import annotations

import logging

import torch

from openwam.model.vlm_backbone.hf_vlm_base import HFVlmBackbone

logger = logging.getLogger(__name__)


class Qwen2_5VLBackbone(HFVlmBackbone):
    """Qwen2.5-VL (and same-architecture derivatives) through the OpenWAM contract."""

    def load_backbone(self, checkpoint_path: str, *, dtype: torch.dtype, load_pretrained: bool):
        try:
            from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Qwen2.5-VL requires transformers with Qwen2_5_VLForConditionalGeneration. "
                "Install via `pip install -U transformers`."
            ) from e

        processor = None
        if checkpoint_path:
            try:
                processor = AutoProcessor.from_pretrained(checkpoint_path)
            except Exception:
                if load_pretrained:
                    raise
        # No ``trust_remote_code``: the class comes from upstream transformers,
        # so a checkpoint directory cannot execute arbitrary Python at load time.
        if load_pretrained:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint_path, dtype=dtype)
        else:
            cfg = AutoConfig.from_pretrained(checkpoint_path)
            model = Qwen2_5_VLForConditionalGeneration(cfg).to(dtype=dtype)
        return model, processor


__all__ = ["Qwen2_5VLBackbone"]
