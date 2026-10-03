"""Cosmos-Predict2.5 video backbone.

Implements the :class:`VideoBackbone` contract (``base.py``) directly on the
upstream Cosmos DiT, VAE, and Reason1 text encoder — **flat named children**,
mirroring the Wan backbone (no wrapper indirection). The heavy DiT block-loop
orchestration lives as stateless helpers in ``cosmos_predict25/dit_forward.py`` (the
analogue of ``wan/dit_forward.py``); this class delegates ``prepare`` /
``run_block`` / ``finalize`` to them, reading ``self.dit``.

The components are built lazily inside :meth:`from_pretrained` (which imports
``cosmos_predict2`` only there), so importing this module is CPU-only-CI safe.

Public surface: **only** the methods/properties already declared on
:class:`VideoBackbone`. Everything else is an auxiliary helper (``_``-prefixed).

Plain-object reality (upstream-imposed): the VAE (``Wan2pt1VAEInterface``) and
Reason1 encoder (``Reason1LiveTextEncoder``) are plain Python objects, not
``nn.Module``. The callable facade is kept as a plain attribute
(``self._vae_iface`` / ``self.text_encoder``) for encode/decode + dtype/device
tracking, while the inner ``nn.Module`` is registered under the clean child
name (``self.vae`` / ``self.reason1``) so its weights enter the unified
state_dict (``vae.*`` / ``reason1.*``). Identity is preserved, so the facade's
``iface.model.model`` still resolves to the same tensors. The inner modules are
moved explicitly in :meth:`set_dtype_device` via ``cosmos_predict25/_vae_utils.py``.

Scope: the video side of the Fixed-16 comparison (``dual_system_fixed16``), which
reads tapped hidden states off ``prepare`` + ``run_block``. Freeze policy is owned
by the training-strategy / model freeze list, reached via native ``nn.Module.get_submodule`` dotted paths
(``dit`` / ``vae`` / ``reason1``). The ``freeze`` kwarg here is retained for
tests / direct programmatic use and defaults to ``False``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.cosmos_predict25 import dit_forward
from openwam.model.video_backbone.cosmos_predict25._vae_utils import (
    _move_cosmos_reason1,
    _move_cosmos_vae,
    _pil_video_to_tensor,
    _vae_device,
    _vae_inner_module,
    _video_tensor_to_pil,
)
from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter

logger = logging.getLogger(__name__)

# Cosmos-Predict2.5 native geometry, invariant across the 2B/14B size family.
# Mirrors the MiniTrainDIT config (``patch_temporal=1`` / ``patch_spatial=2``)
# and the Wan2pt1 VAE temporal contract (causal first frame + 4-frame tail).
_COSMOS25_DIT_PATCH_SIZE: Tuple[int, int, int] = (1, 2, 2)
_COSMOS25_TEMPORAL_COMPRESSION: int = 4
_COSMOS25_CAUSAL_TEMPORAL: bool = True


class CosmosPredict25VideoBackbone(VideoBackbone):
    """Wrap a Cosmos-Predict2.5 DiT/VAE/text-encoder behind the VideoBackbone ABC."""

    def __init__(
        self,
        *,
        net: nn.Module,
        vae: Optional[Any] = None,
        text_encoder: Optional[Any] = None,
        dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        scheduler: Optional[CosmosFlowSchedulerAdapter] = None,
        shift_video: float = 5.0,
        text_dropout_p: float = 0.0,
        text_dropout_seed: Optional[int] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        # --- Flat named children (Wan-shape) ---
        self.dit = net
        # VAE / Reason1 are plain objects: keep the facade as a plain attr (for
        # encode/decode + dtype/device tracking) but register the inner nn.Module
        # so its weights ride the unified state_dict. Identity is preserved, so
        # the facade's `iface.model.model` still resolves to the same tensors.
        self._vae_iface = vae
        inner_vae = _vae_inner_module(vae)
        if inner_vae is not None:
            self.vae = inner_vae
        self.text_encoder = text_encoder
        if text_encoder is not None:
            te_inner = getattr(text_encoder, "model", None)
            if isinstance(te_inner, nn.Module):
                self.reason1 = te_inner

        # --- Geometry + scheduler ---
        self._dim = int(dim)
        self._num_layers = int(num_layers)
        self._num_heads = int(num_heads)
        self._head_dim = int(head_dim)
        # CosmosPredict25's per-token text/context embedding dim (1024 for 2B, vs Wan's
        # 4096). Exposed through the base ``text_dim`` property.
        self._context_dim = int(context_dim)
        self._scheduler = scheduler if scheduler is not None else CosmosFlowSchedulerAdapter()
        self._shift_video = float(shift_video)

        # --- §14.7 CFG dropout (live-encoder path) ---
        if not 0.0 <= float(text_dropout_p) <= 1.0:
            raise ValueError(f"text_dropout_p must be in [0, 1]; got {text_dropout_p!r}.")
        self.text_dropout_p = float(text_dropout_p)
        self._text_dropout_rng = random.Random(text_dropout_seed)

        # Native patch size + temporal contract feed the base properties.
        self._dit_patch_size = _COSMOS25_DIT_PATCH_SIZE
        self._temporal_compression = _COSMOS25_TEMPORAL_COMPRESSION
        self._causal_temporal = _COSMOS25_CAUSAL_TEMPORAL

        self._freeze = bool(freeze)
        if self._freeze:
            for p in self.parameters():  # dit + vae + reason1
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, source: Any, *, device=None, ckpt_dir=None, **kw) -> "CosmosPredict25VideoBackbone":
        """Build a backbone from a config / model dir.

        Defers to :func:`cosmos_predict25.pipeline_builder.build_cosmos_predict25_pipeline`, which
        lazily imports ``cosmos_predict2`` and returns a lightweight holder
        (net + vae + text_encoder + geometry + shift_video). Drains it into flat
        children (Wan holder-drain parity).
        """
        from openwam.model.video_backbone.cosmos_predict25.pipeline_builder import build_cosmos_predict25_pipeline

        cfg_for_loader = _video_backbone_cfg(source)
        shift_video = float(_cfg_get(cfg_for_loader, "shift_video", 5.0))

        holder = build_cosmos_predict25_pipeline(source, device=device, ckpt_dir=ckpt_dir, **kw)
        dim, num_layers, num_heads, head_dim, context_dim = _probe_pipeline_geometry(holder)
        return cls(
            net=holder.net,
            vae=getattr(holder, "vae", None),
            text_encoder=getattr(holder, "text_encoder", None),
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            scheduler=CosmosFlowSchedulerAdapter(shift_video=shift_video),
            shift_video=float(getattr(holder, "shift_video", shift_video)),
            text_dropout_p=float(getattr(holder, "text_dropout_p", 0.0)),
            text_dropout_seed=getattr(holder, "text_dropout_seed", None),
        )

    # ------------------------------------------------------------------
    # Required VideoBackbone properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def scheduler(self) -> CosmosFlowSchedulerAdapter:
        return self._scheduler

    @property
    def text_dim(self) -> Optional[int]:
        """Per-token text/context embedding dim (1024 for CosmosPredict25-2B)."""
        return self._context_dim



    # ------------------------------------------------------------------
    # Three-step block loop — delegates to cosmos_predict25/dit_forward.py
    # ------------------------------------------------------------------

    def prepare(self, **pipeline_inputs) -> BlockLoopState:
        """Patchify / position-embed / pack a :class:`BlockLoopState`."""
        return dit_forward.prepare_block_loop(self.dit, **pipeline_inputs)

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        return dit_forward.run_block(self.dit, block_id, state)

    def finalize(self, state: BlockLoopState) -> Tensor:
        return dit_forward.finalize_block_loop(self.dit, state)







    # ------------------------------------------------------------------
    # Preprocessing & decoding
    # ------------------------------------------------------------------

    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        return self._preprocess_input(frames=frames, text=text, **kw)

    def _preprocess_input(
        self,
        *,
        frames: Any = None,
        text: Any = None,
        input_latents: Optional[Tensor] = None,
        ref_images: Any = None,
        **kw: Any,
    ) -> dict:
        """VAE-encode frames (or accept latents) + encode text → training/inference dict.

        Reads ``self.dit`` (for ``crossattn_proj``), ``self._vae_iface``,
        ``self.text_encoder``, and the live-path CFG-dropout state
        (``self.training`` / ``self.text_dropout_p`` / ``self._text_dropout_rng``).
        Relocated verbatim from the former pipeline wrapper.
        """
        if input_latents is None:
            if frames is None:
                raise ValueError(
                    "CosmosPredict25VideoBackbone._preprocess_input requires either `input_latents` or `frames`."
                )
            if self._vae_iface is None:
                raise RuntimeError(
                    "CosmosPredict25 VAE is not configured. Set `video_backbone.vae: wan2pt1` "
                    "(default; loads `<model_path>/tokenizer.pth`)."
                )
            input_latents = self._encode_frames(frames)

        if text is None:
            raise ValueError("CosmosPredict25VideoBackbone._preprocess_input requires `text`.")
        if self.text_encoder is None:
            raise ValueError("`text=` requires a configured text encoder (Reason1LiveTextEncoder).")
        # §14.7 — CFG dropout: substitute selected prompts with `""` so the
        # encoder produces the canonical empty embedding.
        if self.training and self.text_dropout_p > 0.0:
            text_list = [text] if isinstance(text, str) else list(text)
            text = [t if self._text_dropout_rng.random() >= self.text_dropout_p else "" for t in text_list]
        # Inference-only memoization: the deploy engine passes its bounded
        # server-lifetime `prompt_embed_cache`; training never does (CFG
        # dropout must re-encode per step).
        cache = kw.get("prompt_embed_cache")
        if cache is not None and isinstance(text, str) and text in cache:
            context = cache[text].to(device=input_latents.device, dtype=input_latents.dtype)
        else:
            context = self._encode_text_context(text, device=input_latents.device, dtype=input_latents.dtype)
            if cache is not None and isinstance(text, str):
                cache[text] = context

        B = input_latents.shape[0]
        seq_lens = torch.full((B,), context.shape[1], dtype=torch.long, device=context.device)
        out: dict = {
            "input_latents": input_latents,
            "context": context,
            "context_mask": kw.get("context_mask"),
            "seq_lens": seq_lens,
            "num_frames": input_latents.shape[2],
            "height": input_latents.shape[3],
            "width": input_latents.shape[4],
        }

        # TI2V first-frame conditioning (activated by `ref_images`): VAE-encode
        # one reference frame per sample → `first_frame_latents` + LVG mask.
        ref_active = (
            ref_images is not None
            and isinstance(ref_images, (list, tuple))
            and len(ref_images) > 0
            and all(r is not None for r in ref_images)
        )
        if ref_active:
            if self._vae_iface is None:
                raise RuntimeError(
                    "CosmosPredict25VideoBackbone._preprocess_input received `ref_images` but no VAE is "
                    "configured. Set `video_backbone.vae: wan2pt1` to enable TI2V."
                )
            ref_clips = [r if isinstance(r, (list, tuple)) else [r] for r in ref_images]
            first_frame_latents = self._encode_frames(ref_clips).to(
                device=input_latents.device, dtype=input_latents.dtype
            )
            T_lat = input_latents.shape[2]
            H_lat = input_latents.shape[3]
            W_lat = input_latents.shape[4]
            condition_mask = torch.zeros(
                (B, 1, T_lat, H_lat, W_lat), dtype=input_latents.dtype, device=input_latents.device
            )
            condition_mask[:, :, 0] = 1.0
            out["first_frame_latents"] = first_frame_latents
            out["condition_mask"] = condition_mask
            out["num_clean_prefix_frames"] = 1

        return out

    def _encode_text_context(self, text: Any, *, device, dtype) -> Tensor:
        """Encode text into the post-projection context shared by cond and uncond CFG paths.

        Live encoder returns pre-projection ``(B, 512, 100352) bf16``; the
        DiT-owned ``crossattn_proj`` is applied HERE (not in prepare) so the
        architecture's proprio-token concat sees 1024-d context.
        """
        context = self.text_encoder(text)
        context = context.to(device=device, dtype=dtype)
        net = self.dit
        if getattr(net, "use_crossattn_projection", False) and context.shape[-1] == int(
            getattr(net, "crossattn_proj_in_channels", -1)
        ):
            context = net.crossattn_proj(context)
        return context

    def _encode_frames(self, frames: Any) -> Tensor:
        """PIL frames → bf16 ``(B, 16, T_lat, H/8, W/8)`` Wan2pt1 latents."""
        if self._vae_iface is None:
            raise RuntimeError("CosmosPredict25VideoBackbone._encode_frames called without a configured VAE.")
        video = _pil_video_to_tensor(frames)
        video = video.to(device=_vae_device(self._vae_iface), dtype=torch.bfloat16)
        return self._vae_iface.encode(video)


    # ------------------------------------------------------------------
    # Device / dtype
    # ------------------------------------------------------------------

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        self._dtype = dtype
        self._device = device
        # nn.Module.to(...) walks the flat children (dit / vae / reason1).
        self.to(dtype=dtype, device=device)
        # The plain-object VAE / Reason1 facades are not nn.Modules, so move them
        # explicitly (incl. the wan2pt1 mean/std + scale-list stale-device fix).
        if self._vae_iface is not None:
            _move_cosmos_vae(self._vae_iface, dtype=dtype, device=device)
        if self.text_encoder is not None and not isinstance(self.text_encoder, nn.Module):
            _move_cosmos_reason1(self.text_encoder, dtype=dtype, device=device)


    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Make this backbone's slice of the checkpoint self-contained.

        (1) Merge component reconstruction specs into
        ``cfg.model.video_backbone.components`` (only when absent). (2) Copy the
        Reason1 structural JSONs into ``<output_dir>/reason1/`` — ONLY when a live
        Reason1 encoder is part of this checkpoint (``self.text_encoder`` set, so
        its weights ride the safetensors via ``reason1``). The VAE
        component is emitted only when the ``vae`` child is registered (a VAE was
        configured); under ``vae: none`` no ``vae`` child exists, so emitting the
        spec would leave the saved config internally inconsistent. No-op when
        ``model_path`` is unreadable.
        """
        from omegaconf import DictConfig, OmegaConf, open_dict

        from openwam.model.video_backbone.cosmos_predict25.component_specs import (
            copy_cosmos_predict25_artifacts,
            generate_cosmos_predict25_component_specs,
        )

        is_plain = not isinstance(cfg, DictConfig)
        oc = OmegaConf.create(cfg) if is_plain else cfg

        model_path = OmegaConf.select(oc, "model.video_backbone.model_path", default=None)
        specs = generate_cosmos_predict25_component_specs(str(model_path) if model_path is not None else "")
        if specs is None:
            logger.info(
                "[cosmos_predict25] video_backbone.model_path not readable (%s); skipping deploy-asset save.",
                model_path,
            )
            return

        has_reason1 = getattr(self, "text_encoder", None) is not None
        has_vae = getattr(self, "vae", None) is not None

        def _keep_component(c) -> bool:
            attr = c.get("attr")
            if attr == "text_encoder":
                return has_reason1
            if attr == "vae":
                return has_vae
            return True

        components = [c for c in specs["components"] if _keep_component(c)]

        if "components" not in oc.model.video_backbone:
            with open_dict(oc):
                OmegaConf.update(oc, "model.video_backbone.components", components)
            if is_plain:
                cfg["model"]["video_backbone"]["components"] = components

        if has_reason1:
            copy_cosmos_predict25_artifacts(output_dir, oc)


# ----------------------------------------------------------------------
# Helpers (private)
# ----------------------------------------------------------------------


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _video_backbone_cfg(source: Any) -> Any:
    """Extract the ``video_backbone`` sub-config from a Hydra cfg / path / dict.

    Kept in lockstep with ``pipeline_builder._video_backbone_cfg``.
    """
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        return {"model_path": str(source)}
    if isinstance(source, dict):
        model = source.get("model") if "model" in source else source
        vb = model.get("video_backbone") if isinstance(model, dict) else None
        return vb if vb is not None else source
    model = getattr(source, "model", source)
    vb = getattr(model, "video_backbone", None)
    return vb if vb is not None else source


def _probe_pipeline_geometry(holder: Any) -> Tuple[int, int, int, int, int]:
    """Read (dim, num_layers, num_heads, head_dim, context_dim) off the builder holder."""
    try:
        return (
            int(holder.dim),
            int(holder.num_layers),
            int(holder.num_heads),
            int(holder.head_dim),
            int(holder.context_dim),
        )
    except AttributeError as exc:
        raise AttributeError(
            "Cosmos holder is missing one of {dim, num_layers, num_heads, head_dim, context_dim}. "
            "Attach these in `pipeline_builder.build_cosmos_predict25_pipeline`."
        ) from exc


__all__ = ["CosmosPredict25VideoBackbone"]
