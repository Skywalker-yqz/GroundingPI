"""Qwen3-VL backbone (2B / 4B / …) — differs only by ``checkpoint_path``.

Everything except checkpoint loading lives in
:class:`~openwam.model.vlm_backbone.hf_vlm_base.HFVlmBackbone`, so this family is
driven byte-identically to every other VLM under evaluation.
"""

from __future__ import annotations

import logging

import torch

from openwam.model.vlm_backbone.hf_vlm_base import HFVlmBackbone

logger = logging.getLogger(__name__)


class Qwen3VLBackbone(HFVlmBackbone):
    """Qwen3-VL through the OpenWAM VLM contract.

    Geometry is read from the checkpoint, so 2B (28 blocks / 2048) and 4B
    (36 blocks / 2560) need no code change — only a different
    ``checkpoint_path``. The Fixed-16 π-style depth taps place themselves from
    ``num_layers``.
    """

    def load_backbone(self, checkpoint_path: str, *, dtype: torch.dtype, load_pretrained: bool):
        try:
            from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Qwen3VL requires transformers>=4.50 with Qwen3VLForConditionalGeneration. "
                "Install via `pip install -U transformers`."
            ) from e

        processor = None
        if checkpoint_path:
            try:
                processor = AutoProcessor.from_pretrained(checkpoint_path)
            except Exception:
                if load_pretrained:
                    raise
        # ``trust_remote_code`` is intentionally NOT set: ``Qwen3VLForConditionalGeneration``
        # is imported directly from upstream ``transformers``, so no remote
        # ``modeling_*.py`` is needed. Leaving ``trust_remote_code=True`` would
        # let a config/model directory under ``checkpoint_path`` execute
        # arbitrary Python at load time — an unnecessary code-execution
        # surface in training and deployment.
        if load_pretrained:
            model = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint_path, dtype=dtype)
        else:
            # ``_from_config`` is a transformers private API whose signature
            # has shifted across 4.5x versions (some accept ``torch_dtype``,
            # earlier ones do not). Use the public PreTrainedModel
            # constructor + ``.to(dtype=...)`` instead — equivalent
            # semantically (random init from cfg, no weights downloaded)
            # and stable across releases.
            cfg = AutoConfig.from_pretrained(checkpoint_path)
            model = Qwen3VLForConditionalGeneration(cfg).to(dtype=dtype)
        return model, processor


__all__ = ["Qwen3VLBackbone"]
