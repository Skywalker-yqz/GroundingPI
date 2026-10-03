"""Wan backbone variant strategy.

The Wan2.2-TI2V family conditions on a clean first frame; a checkpoint without
that flag runs as a plain video model. The choice is resolved once at
construction via :func:`detect` and every variant-specific decision is delegated
to ``self._variant``, so ``WanBase`` carries no per-variant branch in its hot paths.
"""

from __future__ import annotations

from openwam.model.video_backbone.wan.variants.base import WanVariant
from openwam.model.video_backbone.wan.variants.ti2v import TI2VVariant


def detect(dit) -> WanVariant:
    """Resolve the Wan variant from the loaded DiT (not from config names)."""
    if bool(getattr(dit, "fuse_vae_embedding_in_latents", False)):
        return TI2VVariant()
    return WanVariant()


__all__ = ["WanVariant", "TI2VVariant", "detect"]
