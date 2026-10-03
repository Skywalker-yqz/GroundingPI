"""Shared Wan :class:`VideoBackbone` base plus the concrete Wan subclasses.

:class:`WanBase` holds every piece of behavior common to the Wan family (DiT
forward, conditioning, deploy). Concrete backbones add only their construction
+ encoder specifics:
  - :class:`Wan22Ti2v` — Wan2.2-TI2V-5B; supports swapping the native VAE for
    an external :class:`VideoEncoder`.

Lives outside ``wan/`` to keep that package Wan-internal. Owns the Wan
modules (DiT/VAE/text encoder/tokenizer) directly — modules as named
children, scheduler/tokenizer/division factors as plain attributes; external
code reaches them only through the ABC methods.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.wan import dit_forward as wan_dit_forward
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan import loader
from openwam.model.video_backbone.wan.models.dit import modulate, rope_apply
from openwam.model.video_backbone.wan.preprocess import (
    check_resize_height_width,
)
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import gradient_checkpoint_forward

logger = logging.getLogger(__name__)


class WanBase(VideoBackbone):
    """Exposes the Wan modules through the VideoBackbone interface.

    Modules registered as named children (clean ``dit.*`` / ``vae.*`` keys);
    scheduler / tokenizer / division factors are plain attributes. Concrete
    subclasses construct via their own ``from_pretrained(source)``.
    """

    # ================================================================
    # Construction
    # ================================================================

    def __init__(self, holder, *, external_encoder=None, shift_video=None, text_dim: Optional[int] = None):
        """Internal constructor. Use a subclass ``from_pretrained()`` instead.

        ``external_encoder`` is ``None`` on the native VAE path so ``state_dict()``
        carries only ``vae.*`` keys; the external-encoder subclass passes one to
        activate VAE-IO routing and register it as a named child.

        ``shift_video`` is the optional Esser α-shift on the video scheduler,
        stored as the single source of truth behind the ABC property. ``None``
        keeps the scheduler template default (Wan = 5.0).
        """
        super().__init__()
        # ``holder`` is a transient carrier: drain its sub-modules + non-Module
        # state into self, then let it go out of scope. Nothing reads it after.
        # Set after nn.Module.__init__ (super) so an nn.Module encoder registers
        # as a named child; shared methods reference ``self.video_encoder``.
        self.video_encoder = external_encoder
        # Optional sub-modules declared up front so the attribute always exists
        # (the loop below only setattr's the ones the holder actually carries).
        self.vae = None
        # Promote sub-modules to named children so state_dict uses clean prefixes.
        for _name in ("dit", "dit2", "vae", "text_encoder"):
            _mod = getattr(holder, _name, None)
            if _mod is not None:
                # nn.Module → named child; non-Module (test mocks) → plain attr,
                # same ``self.<name>`` access resolves on both.
                setattr(self, _name, _mod)
        # Backbone-owned non-Module state. from_pretrained sets the external
        # division factors / latent_spec before ``cls(holder, ...)``.
        self._scheduler = getattr(holder, "scheduler", None)
        self._tokenizer = getattr(holder, "tokenizer", None)
        self._height_division_factor = getattr(holder, "height_division_factor", None)
        self._width_division_factor = getattr(holder, "width_division_factor", None)
        self._time_division_factor = getattr(holder, "time_division_factor", None)
        self._time_division_remainder = getattr(holder, "time_division_remainder", None)
        self._latent_spec = getattr(holder, "latent_spec", None)
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16
        self._shift_video = None if shift_video is None else float(shift_video)
        actual_text_dim = loader.infer_text_dim(getattr(holder, "dit", None))
        self._text_dim = actual_text_dim if text_dim is None else int(text_dim)
        # Resolve the Wan variant (TI2V/plain) ONCE; first-frame conditioning
        # delegates to it so hot paths carry no per-variant branch.
        from openwam.model.video_backbone.wan import variants as _variants

        self._variant = _variants.detect(getattr(holder, "dit", None))
        # Wan native contract, invariant across all Wan2.x variants: DiT
        # patch (1,2,2); VAE 4× temporal compression + causal first-frame token.
        # The external-encoder subclass overrides these from its encoder spec.
        self._dit_patch_size = (1, 2, 2)
        self._temporal_compression, self._causal_temporal = 4, True

    # ================================================================
    # Internal properties
    # ================================================================

    @property
    def _dit(self):
        return self.dit

    @property
    def _uses_external_encoder(self) -> bool:
        """True when routing VAE IO through an external encoder, not the native VAE."""
        return self.video_encoder is not None


    @property
    def _is_ti2v(self) -> bool:
        return bool(getattr(self._dit, "fuse_vae_embedding_in_latents", False))

    @property
    def needs_first_frame_skip(self) -> bool:
        """``True`` iff ``latent[0]`` is a clean conditioning frame excluded from
        the diffusion loss. Only TI2V (per-token t=0 on frame-0 tokens) skips.

        """
        return self._variant.needs_first_frame_skip

    @property
    def _freq_dim(self) -> int:
        return int(self._dit.freq_dim)

    # ================================================================
    # ABC: Properties (6) — device/dtype inherited from VideoBackbone
    # ================================================================

    @property
    def dim(self) -> int:
        return int(self._dit.dim)

    @property
    def num_layers(self) -> int:
        return len(self._dit.blocks)

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def num_heads(self) -> int:
        return int(self._dit.blocks[0].num_heads)

    @property
    def head_dim(self) -> int:
        return int(self._dit.dim) // self.num_heads

    @property
    def text_dim(self) -> Optional[int]:
        return getattr(self, "_text_dim", None)


    def reinit_for_from_scratch(self, *, external_encoder=None, source=None) -> None:
        """from_scratch DiT re-init, owning the Wan dit/patch-size internally.

        Training (``source is None``): random-reinit the DiT, reshaping I/O to
        ``external_encoder`` first when one is swapped in. Deploy
        (``source is not None``): reshape-only (no reset) so the strict checkpoint
        load populates the reshaped tensors; skipped when no external encoder.
        Both transparently no-op when the backbone carries no dit (logged inside
        the reinit helpers)."""
        if source is None:
            from openwam.model.video_backbone.wan.reinit import reinit_dit_from_scratch

            reinit_dit_from_scratch(
                self,
                external_encoder=external_encoder,
                dit_patch_size=self.dit_patch_size,
            )
        elif external_encoder is not None:
            from openwam.model.video_backbone.wan.reinit import adapt_dit_to_external_encoder

            adapt_dit_to_external_encoder(self, external_encoder, self.dit_patch_size)


    # ================================================================
    # ABC: Three-step execution (3)
    # ================================================================

    def prepare(self, **kw) -> BlockLoopState:
        dit = self.dit
        latents = kw["latents"]
        timestep = kw["timestep"]
        context = kw["context"]
        context_mask = kw.get("context_mask")
        seq_lens = kw.get("seq_lens")
        fuse_vae_embedding_in_latents = kw.get("fuse_vae_embedding_in_latents", False)
        num_clean_prefix_frames = kw.get("num_clean_prefix_frames", 0)
        use_gradient_checkpointing = kw.get("use_gradient_checkpointing", False)
        use_gradient_checkpointing_offload = kw.get("use_gradient_checkpointing_offload", False)
        force_per_token_t_mod = bool(kw.get("force_per_token_t_mod", False))

        time_embed, time_modulation = wan_dit_forward.build_time_modulation(
            dit,
            timestep,
            latents,
            patch_size=self._dit_patch_size,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            force_per_token_t_mod=force_per_token_t_mod,
            num_clean_prefix_frames=num_clean_prefix_frames,
            zero_clean_prefix_t_mod=bool(kw.get("zero_clean_prefix_t_mod", False)),
            has_first_frame_latents=kw.get("first_frame_latents") is not None,
        )

        context = dit.text_embedding(context)
        if context_mask is None:
            if seq_lens is not None:
                seq_lens = seq_lens.to(device=context.device)
                positions = torch.arange(context.shape[1], device=context.device).unsqueeze(0)
                context_mask = positions < seq_lens.unsqueeze(1)
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)
            if context_mask.ndim != 2:
                raise ValueError(f"context_mask must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"context_mask shape must match context [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        hidden_states = latents
        if hidden_states.shape[0] != context.shape[0]:
            hidden_states = torch.concat([hidden_states] * context.shape[0], dim=0)
        if timestep.shape[0] != context.shape[0]:
            timestep = torch.concat([timestep] * context.shape[0], dim=0)

        hidden_states = dit.patchify(hidden_states)

        grid_frames, grid_height, grid_width = hidden_states.shape[2:]
        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c").contiguous()
        freqs = (
            torch.cat(
                [
                    dit.freqs[0][:grid_frames]
                    .view(grid_frames, 1, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[1][:grid_height]
                    .view(1, grid_height, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[2][:grid_width]
                    .view(1, 1, grid_width, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                ],
                dim=-1,
            )
            .reshape(grid_frames * grid_height * grid_width, 1, -1)
            .to(hidden_states.device)
        )

        extras = {
            "dit": dit,
            "time_embed": time_embed,  # Wan head time embedding; consumed in finalize()
        }

        return BlockLoopState(
            hidden_states=hidden_states,
            time_mod=time_modulation,
            rope_freqs=freqs,
            context=context,
            context_mask=context_mask,
            grid_frames=grid_frames,
            grid_height=grid_height,
            grid_width=grid_width,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            extras=extras,
        )


    def _run_wan_block(
        self,
        block_id: int,
        block: nn.Module,
        state: BlockLoopState,
        block_context_mask: Optional[Tensor],
    ) -> Tensor:
        return gradient_checkpoint_forward(
            block,
            state.use_gradient_checkpointing,
            state.use_gradient_checkpointing_offload,
            state.hidden_states,
            state.context,
            state.time_mod,
            state.rope_freqs,
            block_context_mask,
        )

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        dit = state.extras["dit"]
        block = dit.blocks[block_id]
        context_mask = state.context_mask
        block_context_mask = (
            context_mask.unsqueeze(1).expand(-1, state.hidden_states.shape[1], -1) if context_mask is not None else None
        )
        state.hidden_states = self._run_wan_block(block_id, block, state, block_context_mask)
        return state

    def finalize(self, state: BlockLoopState):
        """Wan DiT head + unpatchify. Returns ``(B, z_dim, F, H, W)``."""
        dit = state.extras["dit"]
        head = dit.head
        time_embed = state.extras["time_embed"]
        head_time_embed = time_embed if time_embed.dim() == 3 else time_embed.unsqueeze(1)

        hidden_states = head(state.hidden_states, head_time_embed)

        hidden_states = dit.unpatchify(hidden_states, (state.grid_frames, state.grid_height, state.grid_width))
        return hidden_states





    # ================================================================
    # ABC: Unified preprocessing (1)
    # ================================================================

    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        """Unified train preprocessing: raw data (``frames``/``text``, optional
        ``ref_images`` in kw) → the denoising-loop input dict.
        """
        device = self.device
        dtype = self.dtype

        height, width, num_frames = check_resize_height_width(
            frames[0][0].size[1],
            frames[0][0].size[0],
            len(frames[0]),
            height_division_factor=self._height_division_factor,
            width_division_factor=self._width_division_factor,
            time_division_factor=self._time_division_factor,
            time_division_remainder=self._time_division_remainder,
        )

        batch_size = len(frames)
        context, seq_lens = wan_encode.encode_text(
            text, tokenizer=self._tokenizer, text_encoder=self.text_encoder, device=self.device
        )

        all_input_videos = []
        for clip_frames in frames:
            all_input_videos.append(
                wan_encode.preprocess_video(
                    clip_frames, encoder=self.video_encoder, dtype=self.dtype, device=self.device
                )
            )
        stacked_inputs = torch.cat(all_input_videos, dim=0)
        input_latents = wan_encode.encode_video(stacked_inputs, vae=self.vae, encoder=self.video_encoder)
        input_latents = input_latents.to(dtype=dtype, device=device)

        # Variant-specific first-frame / control conditioning lives in
        # ``self._variant``, so this method carries no per-variant ``if``.
        cond = self._variant.build_train_conditioning(
            self,
            input_latents=input_latents,
            frames=frames,
            ref_images=kw.get("ref_images"),
            stacked_inputs=stacked_inputs,
            B=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
            first_frame_image=kw.get("first_frame_image"),
        )

        return {
            "input_latents": input_latents,
            "context": context,
            "seq_lens": seq_lens,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fuse_vae_embedding_in_latents": cond.get("fuse_vae_embedding_in_latents", False),
            "num_clean_prefix_frames": cond.get("num_clean_prefix_frames", 0),
            "first_frame_latents": cond.get("first_frame_latents"),
        }

    # ================================================================
    # ABC: Sub-module access (1)
    # ================================================================

    def get_submodule(self, name: str) -> nn.Module | None:
        # Non-Module names (tokenizer / scheduler) resolve to None.
        mod = getattr(self, name, None)
        return mod if isinstance(mod, nn.Module) else None



    # ================================================================
    # Self-contained checkpoint: specs into config + artifacts into dir
    # ================================================================

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Make this backbone's checkpoint slice self-contained in one pass:
        merge the Wan component + tokenizer specs into ``cfg`` and copy the Wan
        tokenizer into ``output_dir`` (single tokenizer-layout source, no
        duplication). The external-encoder subclass forwards to the encoder's
        own deploy-artifact hook so its side files land alongside.
        """
        from openwam.model.video_backbone.wan.component_specs import save_video_backbone_deploy_assets

        save_video_backbone_deploy_assets(output_dir, cfg)



class Wan22Ti2v(WanBase):
    """Wan2.2-TI2V-5B backbone with optional external-encoder VAE-IO routing.

    ``external_encoder`` is ``None`` on the default path so ``state_dict()``
    carries only ``vae.*`` keys; setting it activates external-encoder VAE-IO
    routing and aliases the encoder under ``"vae"``.
    """

    def __init__(self, holder, *, external_encoder=None, shift_video=None, text_dim: Optional[int] = None):
        """Internal constructor. Use ``from_pretrained()`` instead."""
        # Base sets self.video_encoder after nn.Module.__init__ (an nn.Module
        # encoder cannot be assigned before that), activating VAE-IO routing.
        super().__init__(holder, external_encoder=external_encoder, shift_video=shift_video, text_dim=text_dim)
        if external_encoder is not None:
            # Override the native (1,2,2)/4×/causal contract with the encoder's;
            # callers consult these attrs and never branch on the encoder.
            self._dit_patch_size = external_encoder.properties.dit_patch_size
            self._temporal_compression = int(external_encoder.properties.temporal_compression)
            self._causal_temporal = bool(external_encoder.properties.causal_temporal)

    @classmethod
    def from_pretrained(cls, source, *, external_encoder=None, text_dim: Optional[int] = None, **kw) -> "Wan22Ti2v":
        """Build a Wan22Ti2v from a source.

        Sources: ``DictConfig`` (full Hydra cfg → loader), ``str`` dir path /
        ``dict`` with ``model_path`` (lightweight build), else an already-built
        component holder. Construction returns a transient holder that
        ``__init__`` drains into the backbone.

        With ``external_encoder``: derive division factors from the encoder
        spec, release the native VAE, expose latent-shape metadata. See the
        inline comments.
        """
        from omegaconf import DictConfig

        # Skip materializing the native VAE (avoid ~1.5GB waste / a duplicate
        # VAE slot deploy has no weights for) on training-with-irreversible and
        # on deploy-with-ANY external encoder. Reversible-on-training keeps it,
        # needed for the step-(2) spec cross-check against ``v.z_dim`` etc.
        is_deploy = not isinstance(source, DictConfig)
        skip_native_vae = bool(
            external_encoder is not None and (is_deploy or not external_encoder.properties.pixel_decode)
        )

        holder = loader.build_holder(source, skip_native_vae=skip_native_vae, **kw)

        if external_encoder is not None:
            # (3) Division factors from the encoder spec, not a hardcoded ``* 2`` / Wan-VAE grid, else
            # ``check_resize_height_width`` rounds encoder-legal sizes to Wan's grid. Remainder is 1 iff causal.
            patch_size = external_encoder.properties.dit_patch_size
            holder.height_division_factor = external_encoder.properties.spatial_compression * patch_size[1]
            holder.width_division_factor = external_encoder.properties.spatial_compression * patch_size[2]
            holder.time_division_factor = external_encoder.properties.temporal_compression * patch_size[0]
            holder.time_division_remainder = 1 if external_encoder.properties.causal_temporal else 0

            # (4) Release the native VAE so state_dict keys don't double-count with the external encoder. print (not
            # logger.info) because arch init runs before the logger is wired up; rank-0 gated.
            holder.vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if int(os.environ.get("RANK", 0)) == 0:
                print(
                    f"[Wan22Ti2v] native VAE released; "
                    f"external_encoder={type(external_encoder).__name__} "
                    f"(z_dim={external_encoder.properties.z_dim}, "
                    f"pixel_decode={external_encoder.properties.pixel_decode}, "
                    f"dit_patch_size={external_encoder.properties.dit_patch_size})",
                    flush=True,
                )

            # (5) Expose latent-shape metadata so deploy noise init reads it without the native VAE (now None).
            holder.latent_spec = external_encoder.properties

        # Resolve optional cfg-side ``shift_video`` here (not in __init__)
        # because the cfg shape depends on the ``source`` type.
        shift_video_cfg = loader.resolve_cfg_shift_video(source)

        return cls(holder, external_encoder=external_encoder, shift_video=shift_video_cfg, text_dim=text_dim)

    # ================================================================
    # External-encoder-aware overrides
    # ================================================================

    def get_submodule(self, name: str) -> nn.Module | None:
        if name == "vae" and self._uses_external_encoder:
            return self.video_encoder
        return super().get_submodule(name)


    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Wan deploy assets, then forward to the external encoder's own
        deploy-artifact hook so its side files (e.g. V-JEPA ``manifest.json``)
        land alongside.
        """
        super().save_deploy_assets(output_dir, cfg)
        if self.video_encoder is not None:
            self.video_encoder.save_deploy_assets(output_dir, cfg)


__all__ = ["WanBase", "Wan22Ti2v"]
