"""Contract between the WAM architecture and a concrete video backbone.

Layering: only the architecture talks to the backbone through this contract;
train/deploy go through the architecture, never the backbone object directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class BlockLoopState:
    """Mutable state flowing through prepare → run_block → finalize.

    Architecture may read/write ``hidden_states`` / ``time_mod`` / ``rope_freqs``
    / ``context``; ``grid_*`` and ``extras`` are backbone-owned (read-only to it).
    """

    # Architecture may read/write
    hidden_states: Tensor  # (B, L, dim)
    time_mod: Tensor  # (B, 6, dim) or (B, L, 6, dim) per-token
    rope_freqs: Tensor
    context: Tensor  # text cross-attention embedding
    context_mask: Optional[Tensor] = None  # (B, L_context) bool, True = attend

    # Patch grid for unpatchify
    grid_frames: int = 0
    grid_height: int = 0
    grid_width: int = 0

    # Loop config threaded from prepare() into run_block()
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False

    # Backbone-private escape hatch (Wan stashes dit/time_embed here)
    extras: dict = field(default_factory=dict)


class VideoBackbone(ABC, nn.Module):
    """Architecture ↔ video backbone contract — the architecture's private helper.

    Inherits ``nn.Module`` so named children (``self.dit`` / ``self.vae`` / ...)
    are moved by :meth:`set_dtype_device` and serialized into the state_dict.
    """

    # ================================================================
    # Required: structural metadata
    # ================================================================

    @property
    @abstractmethod
    def dim(self) -> int:
        """Hidden dim of the video DiT."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of DiT blocks."""

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Attention heads per block."""

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Per-head attention dim."""

    @property
    @abstractmethod
    def scheduler(self):
        """Flow-matching scheduler; must support set_timesteps / timesteps / sigmas."""

    @property
    def dit_patch_size(self) -> Tuple[int, int, int]:
        """DiT ``(T, H, W)`` patch size; set ``self._dit_patch_size`` in ``__init__``."""
        return self._dit_patch_size

    @property
    def temporal_compression(self) -> int:
        """``T_pixel / T_lat``; set ``self._temporal_compression`` in ``__init__``."""
        return self._temporal_compression

    # ================================================================
    # Required: construction + training preprocessing
    # ================================================================

    @classmethod
    @abstractmethod
    def from_pretrained(cls, source, **kw) -> "VideoBackbone":
        """Build from pretrained weights. ``source``: path / config / specs / pipe object."""

    @abstractmethod
    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        """Raw training data → tensor dict with at least ``input_latents`` /
        ``context`` / ``seq_lens``. Unconsumed kwargs are dropped via ``**kw``."""

    # ================================================================
    # Required: three-step execution
    # ================================================================

    @abstractmethod
    def prepare(self, **pipeline_inputs) -> BlockLoopState:
        """Pre-loop: patchify / freqs / time_mod. Returns a BlockLoopState."""

    @abstractmethod
    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        """Run a single DiT block. Gradient checkpointing is transparent to the architecture."""

    @abstractmethod
    def finalize(self, state: BlockLoopState) -> Tensor:
        """Post-loop: head + unpatchify. Returns ``(B, C, T, H, W)``."""

    # ================================================================
    # Optional metadata (defaults)
    # ================================================================

    @property
    def device(self) -> torch.device:
        return getattr(self, "_device", torch.device("cuda"))

    @property
    def dtype(self) -> torch.dtype:
        return getattr(self, "_dtype", torch.bfloat16)

    @property
    def shift_video(self) -> Optional[float]:
        """α-shift for the video scheduler (single source of truth for train +
        inference). ``None`` falls back to the scheduler default (Wan = 5.0)."""
        return getattr(self, "_shift_video", None)

    @property
    def external_encoder(self):
        """The swapped-in external :class:`VideoEncoder`, or ``None`` for the
        native VAE path."""
        return getattr(self, "video_encoder", None)

    def reinit_for_from_scratch(self, *, external_encoder=None, source=None) -> None:
        """Re-init the DiT for a ``from_scratch`` run, owning the dit/patch-size
        details internally so the architecture stays backbone-agnostic.

        Two paths, distinguished by ``source``:
          - ``source is None`` (training): random-reinit the DiT weights, after
            reshaping I/O to ``external_encoder``'s latent dim when one is swapped in.
          - ``source is not None`` (deploy): reshape I/O to ``external_encoder``'s
            latent dim WITHOUT resetting, so the subsequent strict checkpoint load
            populates the reshaped tensors. No-op when ``external_encoder is None``.

        Default raises — only backbones with a re-initializable DiT (Wan) support it."""
        raise NotImplementedError(f"{type(self).__name__} does not support from_scratch DiT re-initialization.")

    @property
    def text_dim(self) -> Optional[int]:
        """Per-token raw text/context embedding dim. ``None`` keeps the 4096 fallback."""
        return None

    @property
    def causal_temporal(self) -> bool:
        """Whether the first frame is encoded into its own standalone latent token."""
        return getattr(self, "_causal_temporal", True)

    @property
    def needs_first_frame_skip(self) -> bool:
        """Whether ``latent[0]`` is unconditionally a conditioning frame (Wan TI2V)."""
        return False










    # ================================================================
    # Lifecycle: device/dtype (default moves all registered children)
    # ================================================================

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move everything to ``(dtype, device)``. Overrides call ``super()`` first.
        Must NOT ``.eval()`` — trainable submodules stay in train mode."""
        self._dtype = dtype
        self._device = device
        self.to(dtype=dtype, device=device)

    # ================================================================
    # Optional deploy-asset hook (orchestrated by the architecture)
    # ================================================================

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Make this backbone's slice of the checkpoint self-contained.

        Both halves of self-containment in one place: (1) write the backbone's
        component/tokenizer reconstruction specs into ``cfg`` (so deploy rebuilds
        the module skeletons from ``config.yaml`` without the training-time
        ``model_path``), and (2) copy its non-weight artifact files (tokenizer /
        processor / external-encoder side files) into ``output_dir``. Runs before
        the architecture writes ``config.yaml``. Default no-op."""


__all__ = [
    "BlockLoopState",
    "VideoBackbone",
]
