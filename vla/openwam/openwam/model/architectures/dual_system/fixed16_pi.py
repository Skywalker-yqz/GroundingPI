"""Fixed-16 π-style Layerwise Action DiT over a video-generation backbone.

A representation-comparison harness, not a world-action generator. The video
backbone is used as a **pinned feature extractor**: it runs once per observation
at a fixed diffusion timestep, 8 normalized-depth hidden states are tapped,
projected and resampled to a fixed ``8 × (B, 64, 1024)`` condition set, and the
shared 16-block Action Expert consumes them.

Everything backbone-agnostic — the Action Expert, the flow schedule, the loss,
the cached-condition inference loop — lives in
:class:`~openwam.model.architectures.fixed16_pi_base.Fixed16PiArchitectureBase`,
so the VLM sibling gets a byte-identical Action Expert. This file only supplies
the two backbone hooks.

Deliberate departures from ``dual_system_cross_attn``:

- **No video loss.** Full video denoising trajectories belong to a separate
  joint world-action experiment; mixing them in would make the backbone
  comparison depend on how well each backbone is being fine-tuned as a
  generator. ``vb.finalize`` is therefore skipped too.
- **No raw text/proprio cross-attention.** Instruction reaches control only
  through the evaluated backbone's hidden states; proprioceptive state enters as
  a state token inside the action sequence, never through the video backbone's
  text context.
- **One backbone forward per observation**, with the conditions cached across
  the action denoising steps.
- **Its own flow schedule.** Beta(1.5, 1.0) timestep sampling and ``v* = a - ε``,
  owned by the Action Expert, rather than a shifted-sigmoid
  σ grid and ``ε - a``.

Registered as ``framework=dual_system, variant=fixed16_pi_layerwise``. The
system baseline this protocol is meant to be compared against.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from openwam.model.action_backbone.fixed16_pi_action_dit import NUM_ACTION_BLOCKS
from openwam.model.architectures.fixed16_pi_base import (
    Fixed16PiArchitectureBase,
    fixed16_pi_options,
)
from openwam.model.architectures.registry import _cfg_get, register_architecture

logger = logging.getLogger(__name__)


@register_architecture(
    "dual_system_fixed16_pi",
    # ``supported`` because nothing in the repo ever passes
    # ``build_architecture(..., allow_experimental=True)``, so an experimental
    # entry could not be trained at all. It is a backbone-representation
    # comparison harness, not a video-action generation system.
    status="supported",
    note=(
        "Fixed-16 π-style Layerwise Action DiT: video backbone pinned as a feature extractor, "
        "8 normalized-depth taps → fixed 16-block Action Expert. Action loss only, no video loss."
    ),
    framework="dual_system",
    variant="fixed16_pi_layerwise",
    options_from_cfg=fixed16_pi_options,
)
class DualSystemFixed16PiArchitecture(Fixed16PiArchitectureBase):
    """Video backbone as a pinned feature extractor + the unified Action Expert."""

    _evaluated_backbone_name = "video_backbone"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._compiled_action_forward = None
        if cfg is None:
            return

        # Feature-extraction state must be pinned, not sampled: the comparison is
        # only meaningful if one observation always maps to one condition set.
        self._feature_timestep = float(_cfg_get(cfg, "feature_extraction_timestep", 0.0))
        self._feature_sigma = float(_cfg_get(cfg, "feature_extraction_sigma", 0.0))
        if not 0.0 <= self._feature_sigma <= 1.0:
            raise ValueError(f"feature_extraction_sigma must be in [0, 1], got {self._feature_sigma}")
        self._feature_seed = _cfg_get(cfg, "feature_extraction_seed", 0)
        # Default is co-training: action gradients flow into the video DiT.
        # Set True for the frozen-backbone protocol, which stops gradients at the
        # taps and makes the 8 retained hidden states cheap — but then the video
        # DiT must also be added to the model config's `freeze:` list, or it stays
        # nominally trainable (counted in the parameter report, activations kept).
        self._detach_backbone_features = bool(_cfg_get(cfg, "detach_backbone_features", False))

        # Prefer the loaded backbone's real geometry; fall back to cfg so the
        # architecture can be built before a backbone is attached (the pattern
        # tests use, since real video backbones cannot be loaded on CPU).
        video_dim = self._resolve_video_dim(cfg)
        num_backbone_blocks = (
            self.video_backbone.num_layers
            if self.video_backbone is not None
            else int(_cfg_get(cfg, "num_dit_layers", 0) or 0)
        )
        if num_backbone_blocks <= 0:
            raise ValueError(
                "num_dit_layers must be specified in config or inferred from video_backbone "
                "so the 8 normalized-depth taps can be placed."
            )
        self.setup_fixed16_action_stack(cfg, backbone_hidden_dim=video_dim, num_backbone_blocks=num_backbone_blocks)

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """Batch as usual, but hand the VAE only the observation frame(s).

        Without this the video backbone would encode the whole clip (9 frames
        under the shipped RoboTwin config, so 3 latent frames) while the VLM
        sibling sees a single image — the two backbones would not be looking at
        the same observation and the comparison would be meaningless.

        Multi-camera views are already composed into one canvas per timestep by
        the dataloader, so a single frame still carries every view — the same
        canvas ``extract_first_image`` hands the VLM.
        """
        if isinstance(batch, dict):
            batch = [batch]
        return super().prepare_inputs(self._truncate_to_observation(batch))

    # ------------------------------------------------------------------
    # Backbone hooks
    # ------------------------------------------------------------------

    def _prepare_forward_inputs(self, forward_inputs: dict, *, batch_size: int) -> dict:
        """Pin the backbone's input state so features are reproducible.

        ``feature_extraction_sigma=0`` (the default) feeds the clean VAE latent;
        a non-zero value injects noise from a fixed seed instead. Either way the
        result is a deterministic function of the observation, and the video
        diffusion timestep is a constant rather than a sampled one.
        """
        latents = forward_inputs["input_latents"]
        if self._feature_sigma > 0.0:
            generator = torch.Generator(device=latents.device).manual_seed(int(self._feature_seed))
            noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            latents = (1 - self._feature_sigma) * latents + self._feature_sigma * noise
        if forward_inputs.get("first_frame_latents") is not None:
            ref = forward_inputs["first_frame_latents"]
            latents = latents.clone()
            latents[:, :, : ref.shape[2]] = ref
        forward_inputs["latents"] = latents
        # Assigned, not setdefault: the timestep is part of the pinned feature
        # extraction state (report §9.3), so a caller that happens to carry a
        # `timestep` in its pipeline inputs must not silently move the backbone
        # off the clean t=0 point every other backbone is read at.
        forward_inputs["timestep"] = torch.full(
            (batch_size,), self._feature_timestep, dtype=self.dtype, device=self.device
        )
        return forward_inputs

    def _infer_batch_size(self, pipeline_inputs: dict) -> int:
        latents = pipeline_inputs.get("input_latents")
        if latents is None:
            latents = pipeline_inputs.get("latents")
        if latents is None:
            raise ValueError("Cannot infer batch size: neither input_latents nor latents is present.")
        return int(latents.shape[0])

    def encode_conditions(
        self,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> list[Tensor]:
        """Run the video backbone once and return the 8 cached conditions."""
        vb = self.video_backbone
        if vb is None:
            raise RuntimeError("video_backbone is None — cannot extract conditions.")

        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        taps: dict[int, Tensor] = {}
        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
            if block_id in self._required_blocks:
                hidden = vstate.hidden_states
                taps[block_id] = hidden.detach() if self._detach_backbone_features else hidden
        # ``vb.finalize`` is intentionally not called: this architecture never
        # predicts video, so the DiT head and unpatchify would be wasted compute.
        return self.backbone_conditioner(taps)


__all__ = ["DualSystemFixed16PiArchitecture", "NUM_ACTION_BLOCKS"]
