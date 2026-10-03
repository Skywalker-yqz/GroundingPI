"""Abstract base class for the Fixed-16 comparison architectures.

``BaseWAMArchitecture`` owns the pieces both sides of the comparison share:
building the optional video backbone from config, freezing / device placement,
checkpoint save + load, batching dataset samples into forward inputs, and the
deploy-time normalizer hooks. The concrete architectures
(``dual_system_fixed16_pi`` / ``vlm_system_fixed16_pi``) implement ``forward``,
``compute_loss`` and ``generate`` on top of it.
"""

import functools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn



def _wrap_single_forward(module: nn.Module) -> None:
    """Wrap a single module's ``forward`` in ``torch.no_grad``. Idempotent."""
    if getattr(module, "_openwam_no_grad_wrapped", False):
        return
    original_forward = module.forward

    @functools.wraps(original_forward)
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    module.forward = wrapped
    module._openwam_no_grad_wrapped = True


def _wrap_forward_in_no_grad(module: nn.Module) -> None:
    """Wrap ``forward`` of ``module`` AND every submodule in its subtree in ``torch.no_grad``.

    Recursion matters because callers commonly bypass the root forward and call a
    nested submodule directly. The canonical case in OpenWAM is
    ``Qwen3VLBackbone.extract_features`` which calls ``self.vlm_model.model(...)``
    (the inner ``Qwen3VLModel``, skipping the LM head) — wrapping only
    ``vlm_model.forward`` would leave that path grad-tracking. Recursively wrapping
    every descendant makes the semantic complete: any entry point into the frozen
    subtree is in ``no_grad``.

    Idempotent — a marker attribute on each module prevents double-wrapping if
    ``freeze_modules`` runs more than once. ``nn.Module.modules()`` deduplicates
    via its internal memo, so cyclic registrations (e.g. test fakes with
    ``self.model = self``) are visited once.

    Safe to apply to any module: if the caller already wraps the call in a
    ``no_grad`` context (e.g. ``prepare_inputs``), the inner ``no_grad`` is a
    no-op; if the caller is inside a grad-tracking forward (a co-trained VLM
    case this actually saves memory in), it short-circuits activation saving.

    **Subtree-level semantic, not per-parameter**: a trainable child under a frozen
    parent will NOT receive gradients, because every descendant ``forward`` is
    wrapped in ``no_grad``. For partial-freeze setups (e.g. LoRA on a frozen base,
    or training only the LM head of an otherwise frozen VLM), do NOT pass the
    parent's dotted path to ``freeze_modules``; pass the specific leaves you want
    frozen instead. The current freeze list in ``configs/model/*.yaml``
    only names complete subtrees, so this limitation does not bite today.
    """
    for sub in module.modules():
        _wrap_single_forward(sub)


logger = logging.getLogger(__name__)

# Prefix for VLM backbone parameters in the architecture state_dict.
# VLM weights are saved as a separate checkpoint directory (not in safetensors)
# to avoid tied-weight deduplication complexity.
VLM_STATE_DICT_PREFIX = "vlm_backbone."


def _exclude_vlm_from_state_dict(state_dict: dict[str, "Tensor"]) -> dict[str, "Tensor"]:
    """Filter out VLM backbone parameters from a state dict.

    Applied only when the VLM is **frozen** (see
    :meth:`BaseWAMArchitecture.save_checkpoint`) — a co-trained VLM must be
    saved, or its training is silently discarded.

    Note: this exclusion is prefix-based (``vlm_backbone.*``).  Future
    trainable modules on the VLM (e.g. LoRA adapters) must be registered at
    the architecture top level (as siblings of ``vlm_backbone``), NOT as
    children under ``vlm_backbone``, otherwise they will be silently excluded
    from the checkpoint whenever the VLM is frozen.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith(VLM_STATE_DICT_PREFIX)}


def _dedupe_shared_tensors(state_dict: dict[str, "Tensor"]) -> dict[str, "Tensor"]:
    """Drop entries whose storage is already covered by an earlier key.

    ``safetensors.save_file`` refuses a state dict containing tensors that share
    memory, and Qwen ties ``lm_head.weight`` to ``model.embed_tokens.weight``.
    Excluding the whole VLM used to side-step this; now that a co-trained VLM has
    to be saved, the tie is resolved here instead.

    Dropping is safe because the tie is re-established when the model is
    constructed, and ``load_checkpoint`` tolerates missing ``vlm_backbone.*``
    keys. Dropped keys are logged rather than silently swallowed.
    """
    seen: dict[tuple, str] = {}
    deduped: dict[str, "Tensor"] = {}
    dropped: list[str] = []
    for key, value in state_dict.items():
        storage = getattr(value, "untyped_storage", None)
        if storage is None or value.device.type == "meta":
            deduped[key] = value
            continue
        ident = (value.device, storage().data_ptr(), value.storage_offset(), tuple(value.shape), value.stride())
        owner = seen.get(ident)
        if owner is None:
            seen[ident] = key
            deduped[key] = value
        else:
            dropped.append(f"{key} (shares storage with {owner})")
    if dropped:
        logger.info("Checkpoint: dropped %d tied tensor(s): %s", len(dropped), ", ".join(dropped))
    return deduped


if TYPE_CHECKING:
    from openwam.model.action_backbone.base import ActionDiTBackbone
    from openwam.model.video_backbone.base import VideoBackbone

    AnyActionBackbone = Union["ActionDiTBackbone"]


class BaseWAMArchitecture(ABC, nn.Module):
    """Base class for WAM architecture variants.

    Composes a ``video_backbone`` and an ``action_backbone`` plus optional
    extra backbones. Subclasses instantiate the appropriate action backbone
    subclass in ``__init__`` and own the complete ``forward()`` control flow.

    Args:
        cfg: Architecture-specific configuration (OmegaConf DictConfig or dict).
    """

    # --- Construction & config resolution ---

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.video_backbone: Optional["VideoBackbone"] = None
        self.action_backbone: Optional["AnyActionBackbone"] = None
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        # Forward-time training runtime flags. Trainer calls
        # ``set_training_runtime`` once during construction so ``prepare_inputs``
        # can read these without the trainer having to thread them through.
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0

        # Optional action normalizer for deployment. ``generate`` uses it to
        # return real-scale actions; deploy-side proprio preprocessing uses it
        # to normalize raw robot state into the model's training space.
        self.normalizer = None

        if cfg is not None:
            self._init_video_backbone(cfg)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _init_video_backbone(self, cfg):
        """Build video backbone from config.

        Supports two source types in ``cfg.video_backbone``:
        - ``_source``: direct model directory or dict with components → deploy-time path
        - ``name``: registry key → training-time path

        Both paths flow through the public :func:`build_video_backbone`.

        Optional ``video_backbone.encoder`` block (yaml-whitelisted to
        ``{name, model_path}``) swaps the backbone's native VAE for an
        external :class:`VideoEncoder`. The encoder block is **only** read
        when ``video_backbone.from_scratch=true`` — the DiT must be
        reinitialized when its latent space changes. When the block is set
        but ``from_scratch=false`` we silently route through the native
        ``pipe.vae`` (with an INFO log explaining what happened) so that the
        default yaml's documentation-friendly ``encoder:`` block doesn't
        break the default training command.
        """
        from openwam.model.video_backbone import build_video_backbone

        vb_cfg = cfg.get("video_backbone", {}) if isinstance(cfg, dict) else getattr(cfg, "video_backbone", None)
        if vb_cfg is None:
            return

        source = vb_cfg.get("_source") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_source", None)
        vb_name = vb_cfg.get("name") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "name", None)
        from_scratch = bool(self._cfg_get(vb_cfg, "from_scratch", False))

        # ------------------------------------------------------------------
        # External encoder gate. Four cases, only one of which builds an
        # encoder:
        #   - encoder set + from_scratch=true  → build external encoder
        #   - encoder set + from_scratch=false → INFO log + skip (silent
        #     ignore is the right UX since the default yaml ships an
        #     encoder: block for documentation discoverability, and we
        #     don't want the default training command to fail)
        #   - encoder unset + from_scratch=true → reset DiT weights only
        #   - encoder unset + from_scratch=false → no-op (default path)
        # ------------------------------------------------------------------
        enc_cfg = vb_cfg.get("encoder") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "encoder", None)
        external_encoder = None
        # Single gate, identical for training and deploy: encoder block is
        # honored ONLY when ``from_scratch=true``. The framework yamls ship
        # an inline ``encoder:`` block for discoverability even at default
        # ``from_scratch=false`` (see commit 6588044) — that block must be
        # silently ignored on both paths so default training and deploy of
        # ``from_scratch=false`` checkpoints (state_dict topology
        # ``_pipe.vae.*``) keep working bit-exactly.
        if enc_cfg is not None and from_scratch:
            if source is None:
                # Training: build the encoder from yaml + model_path.
                from openwam.model.video_backbone.encoder import build_video_encoder

                external_encoder = build_video_encoder(enc_cfg)
            else:
                # Deploy: reconstruct the encoder skeleton from the saved
                # components entry; weights filled in by the architecture's
                # subsequent ``load_checkpoint`` strict load. ``source`` is
                # the dict produced by deploy/model_loader.py. ``_ckpt_dir``
                # is plumbed onto ``vb_cfg`` by model_loader and forwarded
                # to :meth:`VideoEncoder.from_skeleton` so each encoder can
                # consult the checkpoint-local artifacts that its
                # :meth:`VideoEncoder.save_deploy_assets` wrote at save
                # time. For example, V-JEPA 2.1 prefers
                # ``<ckpt_dir>/manifest.json`` with a fallback to
                # ``encoder.model_path``. The user-side weight directory does
                # not need to be reachable on the deploy host.
                ckpt_dir_for_encoder = (
                    vb_cfg.get("_ckpt_dir") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_ckpt_dir", None)
                )
                external_encoder = self._build_external_encoder_skeleton(enc_cfg, source, ckpt_dir=ckpt_dir_for_encoder)
        elif enc_cfg is not None and source is None:
            # Training with encoder block set but from_scratch=false. Two
            # sub-cases:
            #   (a) ``encoder.name == "wan_vae"`` (the default-yaml template
            #       value) — stay silent (INFO only). Native ``pipe.vae``
            #       and the wan_vae external encoder are bit-identical, so
            #       nothing is lost; the default yaml ships the
            #       ``encoder:`` block as a discoverable hint and
            #       fail-fast here would break every default config.
            #   (b) ``encoder.name`` is anything else — that is an explicit
            #       choice that *cannot* take effect under
            #       ``from_scratch=false``: the pre-trained DiT's first
            #       conv channels are bound to the native Wan VAE's
            #       ``z_dim`` and there is no way to wire a different
            #       encoder's latent space through without re-initializing
            #       the DiT. Silently INFO-logging would produce a Wan-VAE
            #       run that *looks* like a V-JEPA run from the yaml, so
            #       fail-fast.
            enc_name = ""
            if isinstance(enc_cfg, dict):
                enc_name = str(enc_cfg.get("name", ""))
            else:
                enc_name = str(getattr(enc_cfg, "name", ""))
            if enc_name and enc_name != "wan_vae":
                raise ValueError(
                    f"video_backbone.encoder.name='{enc_name}' is incompatible "
                    "with from_scratch=false: the pre-trained DiT's first conv "
                    "channels are bound to native Wan VAE's z_dim and cannot "
                    "consume a different encoder's latent space. Set "
                    "from_scratch=true to activate the encoder swap (and re-init "
                    "the DiT), or remove the encoder block to keep the native "
                    "Wan VAE path. See docs/external_video_encoder.md §1."
                )
            logger.info(
                "video_backbone.encoder is set but from_scratch=false; "
                "encoder block IGNORED, using native pipe.vae. Set from_scratch=true "
                "to activate the encoder swap. See docs/external_video_encoder.md."
            )

        text_dim = self._cfg_get(cfg, "text_dim", None)
        text_dim = None if text_dim in (None, 0) else int(text_dim)
        if source is not None:
            ckpt_dir = vb_cfg.get("_ckpt_dir") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_ckpt_dir", None)
            self.video_backbone = build_video_backbone(
                vb_name,
                cfg,
                source=source,
                device="cpu",
                ckpt_dir=ckpt_dir,
                external_encoder=external_encoder,
                text_dim=text_dim,
            )
        elif vb_name is not None:
            self.video_backbone = build_video_backbone(
                vb_name, cfg, external_encoder=external_encoder, text_dim=text_dim
            )

        # Cross-check: yaml-declared temporal contract must match what the
        # backbone actually exposes (sourced from external encoder spec on the
        # external path, native VAE defaults otherwise). Drift here would let
        # the dataloader enforce the wrong divisibility rule and let the
        # mask-downsampler produce a wrong-length tail, so we fail-fast at
        # backbone init. We read from the backbone (not directly from the
        # encoder spec) so the contract has a single owner — see A1's
        # dit_patch_size ABC-property design.
        if self.video_backbone is not None:
            declared_tc = self._cfg_get(vb_cfg, "temporal_compression", 4)
            declared_causal = self._cfg_get(vb_cfg, "causal_temporal", True)
            actual_tc = self.video_backbone.temporal_compression
            actual_causal = self.video_backbone.causal_temporal
            if external_encoder is not None:
                encoder_src = f"external encoder {type(external_encoder).__name__}"
            else:
                encoder_src = "native VAE"
            if (declared_tc, declared_causal) != (actual_tc, actual_causal):
                raise ValueError(
                    f"video_backbone.temporal_compression / causal_temporal yaml "
                    f"({declared_tc}, {declared_causal}) does not match {encoder_src} "
                    f"({actual_tc}, {actual_causal}). Update the yaml fields to match."
                )

        # Optional from-scratch DiT: keep the Wan video backbone structure
        # but discard the loaded DiT weights and re-randomize them in place.
        # VAE and the text encoder stay pretrained and are frozen by the
        # training strategy yaml. Reproducibility comes from
        # ``cfg.project.seed`` which ``OpenWAMTrainer`` applies before
        # architecture construction. Applies uniformly to every architecture
        # that builds its video backbone via this method.
        #
        # IMPORTANT: gated on ``source is None`` (training path only). On
        # deploy, ``cfg.video_backbone.from_scratch`` is True because the
        # config was saved from a from-scratch training run, but DiT weights
        # come from the checkpoint, NOT from a re-initialization. Calling
        # reinit here would silently wipe the trained DiT weights and the
        # subsequent ``load_checkpoint`` would overwrite them again — wasted
        # work in the best case, but if the checkpoint had any missing keys
        # the strict load would surface them against zeroed weights instead
        # of the random init, masking the diagnostic.
        # Both the training reset (source is None) and the deploy reshape-only
        # path (source set + external encoder) are owned by the backbone via the
        # ``reinit_for_from_scratch`` contract — the architecture no longer
        # reaches into ``vb.dit`` / ``wan.reinit``. Non-Wan backbones raise
        # NotImplementedError, so this stays gated on from_scratch.
        if self.video_backbone is not None and from_scratch:
            self.video_backbone.reinit_for_from_scratch(
                external_encoder=external_encoder,
                source=source,
            )

    @staticmethod
    def _build_external_encoder_skeleton(enc_cfg, source, *, ckpt_dir=None):
        """Deploy-time external encoder constructor.

        Reads the encoder ``name`` and reaches into the saved ``source`` dict
        for the ``components`` list to find the ``attr == "vae"`` entry. That
        entry's ``model_class`` / ``extra_kwargs`` is handed to the
        encoder class's :meth:`VideoEncoder.from_skeleton` classmethod,
        which instantiates the underlying module with zero weights. The
        architecture's :meth:`load_checkpoint` strict load fills in the
        weights immediately after.

        ``ckpt_dir`` is forwarded to ``from_skeleton`` so encoders that
        depend on side files can read them from the checkpoint dir itself,
        not from the user-side weight directory. Two patterns coexist:
        V-JEPA 2.1 prefers ``<ckpt_dir>/manifest.json`` and falls back to
        ``encoder.model_path`` for older checkpoints.

        Refuses to silently fall back to the native VAE path here: if the
        cfg has an encoder block but the components list is missing a vae
        entry (e.g. corrupted save), raise so the operator sees the
        mismatch up front.
        """
        from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY

        enc_name = enc_cfg["name"] if isinstance(enc_cfg, dict) else enc_cfg.name
        if enc_name not in _VIDEO_ENCODER_REGISTRY:
            available = ", ".join(sorted(_VIDEO_ENCODER_REGISTRY)) or "(none)"
            raise KeyError(f"Unknown video encoder '{enc_name}'. Available: {available}")

        components = (source or {}).get("components") if isinstance(source, dict) else None
        if not components:
            raise RuntimeError(
                "Deploy with encoder block but saved config has no "
                "video_backbone.components — cannot reconstruct encoder skeleton. "
                "Re-save the checkpoint with the current code, or strip the "
                "encoder block from config.yaml to fall back to native VAE."
            )
        vae_entry = next((e for e in components if e.get("attr") == "vae"), None)
        if vae_entry is None:
            raise RuntimeError(
                "Deploy with encoder block but components list has no attr=vae "
                "entry to construct the encoder skeleton from."
            )
        encoder_cls = _VIDEO_ENCODER_REGISTRY[enc_name]
        return encoder_cls.from_skeleton(vae_entry, encoder_cfg=enc_cfg, ckpt_dir=ckpt_dir)

    def _resolve_video_dim(self, cfg) -> int:
        """Resolve video_dim from config or video_backbone; raise if neither provides it."""
        dim = int(cfg.get("video_dim", 0)) if isinstance(cfg, dict) else int(getattr(cfg, "video_dim", 0))
        if dim == 0 and self.video_backbone is not None:
            dim = self.video_backbone.dim
        if not dim:
            raise ValueError("video_dim must be specified in config or inferred from video_backbone")
        return dim

    # --- Backbone composition ---

    @property
    def backbones(self) -> dict[str, nn.Module]:
        """All backbone modules owned by this architecture.

        Subclasses with additional backbones (e.g. TriSystem with a VLM
        backbone) should override this to include them. The returned dict
        is used by ``init_training_schedulers``, ``set_dtype_device``,
        ``move_frozen_to_device``, and ``save_assets_for_deployment`` to iterate
        over all backbones generically.
        """
        result = {}
        if self.video_backbone is not None:
            result["video_backbone"] = self.video_backbone
        if self.action_backbone is not None:
            result["action_backbone"] = self.action_backbone
        return result

    # --- Action-side properties (delegate to action_backbone) ---

    #: Whether deploy must build a joint video/action denoising schedule before
    #: calling :meth:`generate`. True for architectures that denoise both streams
    #: in lockstep. An architecture that denoises actions alone sets this False:
    #: the engine then skips ``make_schedule``, which would otherwise reach for
    #: ``video_scheduler`` — and that raises when there is no video backbone.


    @property
    def external_encoder(self):
        """The video backbone's external encoder, or ``None`` (native VAE / no video
        backbone). Train/deploy read this instead of reaching into video_backbone
        internals (the layering boundary: only the architecture talks to backbones)."""
        vb = self.video_backbone
        return vb.external_encoder if vb is not None else None

    @property
    def action_dim(self) -> int:
        return self.action_backbone.action_dim if self.action_backbone is not None else 0

    @property
    def bridge_layers(self) -> tuple:
        return self.action_backbone.bridge_layers if self.action_backbone is not None else ()

    @property
    def uses_proprioception(self) -> bool:
        return bool(getattr(self, "_use_proprioception_context", False)) or (
            self.action_backbone is not None and self.action_backbone.uses_proprioception
        )



    # --- Device / dtype (top-level authority) ---

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Dispatch to each backbone — they own their own dtype/device handling."""
        self._dtype = dtype
        self._device = device
        proprio_encoder = getattr(self, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.to(dtype=dtype, device=device)
        for bb in self.backbones.values():
            bb.set_dtype_device(dtype, device)

    # --- Normalizer (deployment) ---

    def attach_normalizer(self, normalizer) -> None:
        """Attach (or clear) an action normalizer used by ``generate``.

        Deployment paths build the same normalizer used by training from
        ``normalization_stats.npy``. ``generate`` uses it to return real-scale actions,
        while server-side proprio preprocessing uses it to normalize raw robot
        state into the model's training space. Pass ``None`` to clear.
        """
        self.normalizer = normalizer

    def normalize_deploy_proprio(self, proprio):
        """Normalize raw deploy proprio (array-like) into a float32 tensor; ``None`` passes through.

        The denoising loop re-casts to the model device/dtype, so a CPU tensor is fine.
        """
        if proprio is None:
            return None

        import numpy as np
        import torch

        arr = np.asarray(proprio, dtype=np.float32)
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            arr = normalizer.normalize(arr)
        return torch.from_numpy(arr)

    # --- Checkpoint save / load ---

    def vlm_is_trainable(self) -> bool:
        """Whether any VLM parameter can be updated by the optimizer."""
        vlm = getattr(self, "vlm_backbone", None)
        return vlm is not None and any(p.requires_grad for p in vlm.parameters())

    def save_checkpoint(self, path: str, *, state_dict: dict | None = None) -> None:
        """Save architecture state to safetensors.

        A **frozen** VLM is excluded — its weights are unchanged from the
        pretrained directory that ``save_deploy_assets`` copies next to the
        checkpoint, so storing them again would just bloat every checkpoint.
        A **co-trained** VLM is included: excluding it would silently throw away
        the training. Tied tensors are resolved by :func:`_dedupe_shared_tensors`
        because safetensors rejects shared storage.

        ``state_dict`` defaults to ``self.state_dict()`` (deploy export); the
        trainer passes a gathered state_dict (ZeRO/DDP all-gather) instead.
        """
        from safetensors.torch import save_file

        if state_dict is None:
            state_dict = self.state_dict()
        if self.vlm_is_trainable():
            logger.info("VLM is trainable — including vlm_backbone.* in the checkpoint.")
            state_dict = _dedupe_shared_tensors(state_dict)
        else:
            state_dict = _exclude_vlm_from_state_dict(state_dict)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state_dict, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        """Load architecture state from a safetensors checkpoint.

        VLM backbone weights are not stored in the safetensors file (they
        are saved as a separate directory). When a VLM backbone is present,
        missing ``vlm_backbone.*`` keys are tolerated; unexpected or missing
        non-VLM keys still raise under ``strict=True``.

        Meta-device sub-modules (self-contained deploy empty shells built
        via ``from_empty`` / ``init_empty_weights``) need
        ``load_state_dict(..., assign=True)`` — the default in-place copy is
        a silent no-op against meta tensors and leaves the shells
        unpopulated. ``assign=True`` rebinds the parameter slot to the
        safetensors tensor instead. We only flip the flag when meta params
        actually exist so the training-resume path (real-device params,
        in-place copy preserves identity) is unchanged.
        """
        from safetensors.torch import load_file

        state_dict = load_file(path)
        has_vlm = getattr(self, "vlm_backbone", None) is not None
        has_meta = any(p.device.type == "meta" for p in self.parameters())
        missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=has_meta)
        if strict and not has_vlm:
            if missing or unexpected:
                raise RuntimeError(f"Strict load failed: missing={missing}, unexpected={unexpected}")
        elif strict and has_vlm:
            non_vlm_missing = [k for k in missing if not k.startswith(VLM_STATE_DICT_PREFIX)]
            if non_vlm_missing or unexpected:
                raise RuntimeError(
                    f"Strict load failed (VLM keys excluded): missing={non_vlm_missing}, unexpected={unexpected}"
                )

    # --- Training: module management ---

    def init_training_schedulers(self, num_timesteps: int = 1000) -> None:
        """Initialize all backbone schedulers for training.

        Single source of truth for each stream's α-shift:
        ``video_backbone.shift_video`` and ``action_backbone.shift_action``.
        The same properties are read by the deploy schedule at inference time,
        so the discrete training sigma buffer and the inference denoising
        trajectory are sampled from the same shifted schedule — train/inference
        cannot drift, and the shift is owned by the checkpoint config (not a
        separate deploy-time knob).

        ``shift is None`` (the default for configs without an explicit override)
        falls back to each scheduler's template default (Wan/action = 5.0),
        i.e. bit-identical pre-shift behavior.
        """
        # ``getattr`` (rather than direct attribute access) so test doubles
        # / mocks that don't carry the shift property still work — they fall
        # back to the scheduler's template default, matching the production
        # no-override path.
        for name, bb in self.backbones.items():
            if not hasattr(bb, "scheduler"):
                continue
            kwargs = {"training": True}
            shift = getattr(bb, "shift_video" if name == "video_backbone" else "shift_action", None)
            if shift is not None:
                kwargs["shift"] = float(shift)
            bb.scheduler.set_timesteps(num_timesteps, **kwargs)

    def freeze_modules(self, names: list[str]) -> list[str]:
        """Freeze named sub-modules by dotted path. Returns actually frozen names.

        Single-point freeze API. Two effects per frozen submodule:

        1. ``module.requires_grad_(False)`` — optimizer cannot update its params.
        2. ``module.forward`` is wrapped in ``torch.no_grad`` so the frozen
           subtree never saves activations for backward. This is the full
           semantic of "freeze" — neither the trainer nor any backbone needs to
           inspect freeze status separately.

        For text_encoder / vae, which are already called under the
        ``@torch.no_grad()`` ``prepare_inputs`` decorator, the wrapper is a
        no-op (nested ``no_grad``). For modules called inside the training
        forward graph (e.g. a frozen VLM backbone), the
        wrapper is what actually saves activation memory.

        Uses ``nn.Module.get_submodule()`` so dotted paths like
        ``video_backbone.text_encoder`` work naturally; unknown names
        are silently skipped, so a freeze list mentioning modules absent on a
        given architecture (e.g. ``vlm_backbone.vlm_model`` on dual_system)
        is harmless.
        """
        frozen = []
        for name in names:
            try:
                module = self.get_submodule(name)
            except (AttributeError, KeyError):
                module = None
            if module is not None:
                module.requires_grad_(False)
                # Set eval mode on the frozen subtree. Use modules() instead
                # of .eval() to avoid infinite recursion when a submodule has
                # self-referential aliases (e.g. HF model.model = self).
                for sub in module.modules():
                    sub.training = False
                _wrap_forward_in_no_grad(module)
                frozen.append(name)
        return frozen

    def get_trainable_modules(self, freeze_list: list[str] = ()) -> dict[str, nn.Module]:
        """Return top-level trainable sub-modules.

        Walks ``self.named_children()`` and returns modules that have at
        least one parameter with ``requires_grad=True``, excluding those
        in *freeze_list*. Used by ``optimizer_groups.build_trainable_parameters``
        to source the param groups for the optimizer.
        """
        result = {}
        freeze_set = set(freeze_list)
        for name, mod in self.named_children():
            if name in freeze_set:
                continue
            if any(p.requires_grad for p in mod.parameters()):
                result[name] = mod
        return result

    def move_frozen_to_device(self, device: torch.device, names: tuple[str, ...] = ("text_encoder", "vae")) -> None:
        """Move named frozen modules to device.

        Searches via ``get_submodule`` on self first, then on each backbone.
        """
        for name in names:
            mod = None
            try:
                mod = self.get_submodule(name)
            except (AttributeError, KeyError):
                pass
            if mod is None:
                for bb in self.backbones.values():
                    found = None
                    try:
                        found = bb.get_submodule(name)
                    except (AttributeError, KeyError):
                        found = None
                    if found is not None:
                        mod = found
                        break
            if mod is not None:
                mod.to(device=device)

    def save_assets_for_deployment(self, output_dir: str, cfg) -> None:
        """Make the checkpoint directory self-contained for deploy.

        One entry the trainer calls once per checkpoint save, BEFORE
        ``save_config`` writes ``config.yaml``. Each backbone's
        :meth:`VideoBackbone.save_deploy_assets` merges its component/tokenizer
        reconstruction specs into ``cfg`` (so deploy rebuilds the module
        skeletons from ``config.yaml`` without the training-time ``model_path``)
        and copies its artifact files (tokenizer / processor) into ``output_dir``.
        Every backbone base declares the hook (default no-op), so no probing here.
        """
        for bb in self.backbones.values():
            bb.save_deploy_assets(output_dir, cfg)

    # --- Training: preprocessing ---

    @torch.no_grad()
    def preprocess(self, **kwargs) -> dict:
        """Encode raw frames/text into latents + context for training.

        Delegates to ``video_backbone.preprocess_input_for_train()``. External code
        (trainer) should call this instead of touching video_backbone directly.
        """
        return self.video_backbone.preprocess_input_for_train(**kwargs)

    def set_training_runtime(
        self,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
    ) -> None:
        """Set forward-time training flags consumed by ``prepare_inputs``.

        Trainers call this once during construction. Keeping these on the
        architecture keeps ``prepare_inputs(batch)`` self-contained — the
        trainer no longer needs to thread these flags through every loss call.
        """
        self._use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self._max_timestep_boundary = float(max_timestep_boundary)
        self._min_timestep_boundary = float(min_timestep_boundary)

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """Aggregate a list of dataset samples into a batched inputs dict.

        Absorbs the per-sample field collection that previously lived in
        ``OpenWAMTrainer._forward_batch``. The returned dict is designed to be
        unpacked directly into ``compute_loss`` via ``**inputs``.

        Args:
            batch: List of dataset samples (each a dict). A single dict is
                accepted as well and treated as a one-sample batch.

        Returns:
            Dict with all preprocessed video latents, text embeddings, action
            tensors, masks, and forward-time flags ready for ``compute_loss``.
        """
        from openwam.dataloader.transforms.pipeline import FirstFrameConditioningTransform

        if isinstance(batch, dict):
            batch = [batch]

        if not hasattr(self, "_pipeline_transform_instance"):
            self._pipeline_transform_instance = FirstFrameConditioningTransform()
        samples = [self._pipeline_transform_instance.apply(s) for s in batch]

        _dtype = self.dtype
        _device = self.device

        all_frames: list = []
        all_prompts: list = []
        all_ref_images: list = []
        all_video_masks: list = []

        for sample in samples:
            all_frames.append(sample["video"])
            all_prompts.append(sample["prompt"])
            all_ref_images.append(sample.get("first_frame_image"))
            vmask = sample.get("video_mask", None)
            if isinstance(vmask, np.ndarray):
                vmask = torch.from_numpy(vmask)
            all_video_masks.append(vmask)

        # Action / proprio / action-mask collection is backbone-independent and
        # lives in its own helper so architectures without a video stream reuse
        # it verbatim instead of reimplementing it.
        sample_tensors = self._collect_sample_tensors(samples)

        ref_flags = [r is not None for r in all_ref_images]
        if any(ref_flags) and not all(ref_flags):
            raise ValueError("Mixed reference images in batch: all samples must be consistent.")

        preprocessed = self.preprocess(
            frames=all_frames,
            text=all_prompts,
            ref_images=all_ref_images if ref_flags[0] else None,
        )

        inputs = {
            **preprocessed,
            "latents": None,
            "cfg_scale": 1,
            "cfg_merge": False,
            "tiled": False,
            "use_gradient_checkpointing": self._use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self._use_gradient_checkpointing_offload,
            "max_timestep_boundary": self._max_timestep_boundary,
            "min_timestep_boundary": self._min_timestep_boundary,
            "actions": sample_tensors["actions"],
        }
        # proprio / proprio_mask / action_is_pad are present only when the batch
        # actually carried them, matching the previous inline behaviour.
        for key in ("proprio", "proprio_mask", "action_is_pad"):
            if key in sample_tensors:
                inputs[key] = sample_tensors[key]

        if all_video_masks[0] is not None:
            inputs["video_is_pad"] = self._video_masks_to_latent_pad(all_video_masks, inputs)

        return inputs

    @torch.no_grad()
    def _collect_sample_tensors(self, samples: list[dict]) -> dict:
        """Batch the backbone-independent per-sample fields.

        Returns ``actions`` (possibly ``None``) plus ``proprio`` /
        ``proprio_mask`` / ``action_is_pad`` **only when the batch carries them**,
        so callers can splat the result straight into their inputs dict.

        Split out of :meth:`prepare_inputs` so a VLM-only architecture — which
        cannot call the video ``preprocess`` path at all — reuses this exact
        logic. The 2-D action-mask handling and the mixed-rank ``proprio_mask``
        reconciliation below are pinned by
        ``tests/test_prepare_inputs_proprio_mask.py`` and
        ``tests/test_proprio_mask_mixture_collate.py``; duplicating them would
        drift.
        """
        _dtype = self.dtype
        _device = self.device

        all_actions: list = []
        all_proprios: list = []
        all_proprio_masks: list = []
        all_action_masks: list = []

        for sample in samples:
            action = sample.get("action")
            if action is not None:
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action)
                action = action.to(dtype=_dtype, device=_device).unsqueeze(0)
            all_actions.append(action)

            # Carry proprio whenever the sample provides it — the main-stream
            # proprio-context path consumes it downstream, so the bridge into
            # ``inputs`` is not gated on a single consumer's flag.
            proprio = sample.get("proprio")
            if proprio is not None:
                if isinstance(proprio, np.ndarray):
                    proprio = torch.from_numpy(proprio)
                proprio = proprio.to(dtype=_dtype, device=_device)
                if proprio.ndim == 1:
                    pass
                elif proprio.ndim == 2 and proprio.shape[0] == 1:
                    proprio = proprio[0]
                else:
                    raise ValueError(f"sample['proprio'] must be [D] or [1, D], got shape {tuple(proprio.shape)}")
            all_proprios.append(proprio)

            # Collect per-sample proprio_mask. Two accepted shapes:
            #   * 1D ``(1,) bool`` — legacy "is the proprio token enabled".
            #   * 2D ``(1, D) bool`` — per-dim mask; sample-level enable is
            #     ``pmask.any(dim=-1)``. Used by the dataset readers
            #     after the 2D mask migration.
            # Default for readers that don't emit the field: all True (1,).
            # (Mixed 1-D / 2-D ranks across a batch are reconciled just before
            # the stack below, so the bare (1,) default is safe.)
            pmask = sample.get("proprio_mask")
            if pmask is None:
                pmask = torch.ones(1, dtype=torch.bool)
            else:
                if isinstance(pmask, np.ndarray):
                    pmask = torch.from_numpy(pmask)
                pmask = pmask.to(dtype=torch.bool)
                if pmask.ndim == 0:
                    pmask = pmask.unsqueeze(0)
            all_proprio_masks.append(pmask)

            amask = sample.get("action_mask", None)
            if isinstance(amask, np.ndarray):
                amask = torch.from_numpy(amask)
            all_action_masks.append(amask)

        inputs: dict = {"actions": torch.cat(all_actions, dim=0) if all_actions[0] is not None else None}

        # Bridge proprio into inputs whenever the batch carries it (not gated on
        # uses_proprioception): the main-stream proprio-context path reads
        # inputs["proprio"]. Consumers that don't need it simply ignore it.
        if all_proprios[0] is not None:
            inputs["proprio"] = torch.stack(all_proprios, dim=0).contiguous()
            # Reconcile mixed 1-D (1,) / 2-D (1, D) proprio_masks before stacking:
            # promote any 1-D enable-flag to (1, D) (broadcasts the sample-level
            # flag across all dims) so a batch mixing a 2-D reader mask with a
            # 1-D default/external mask doesn't raise on rank mismatch. All-1-D
            # and all-2-D batches are left untouched.
            if len({m.ndim for m in all_proprio_masks}) > 1:
                pdim = max((m.shape[-1] for m in all_proprio_masks if m.ndim == 2), default=1)
                all_proprio_masks = [
                    m if m.ndim == 2 else m.reshape(m.shape[0], 1).expand(m.shape[0], pdim) for m in all_proprio_masks
                ]
            inputs["proprio_mask"] = torch.stack(all_proprio_masks, dim=0).contiguous()

        if all_action_masks[0] is not None:
            inputs["action_is_pad"] = torch.stack([~m for m in all_action_masks], dim=0).to(device=_device)

        return inputs

    def _video_masks_to_latent_pad(self, all_video_masks: list, inputs: dict):
        """Downsample per-frame video masks onto the latent grid. Video stream only.

        Split out of :meth:`prepare_inputs` alongside
        :meth:`_collect_sample_tensors`: this is the one part of the batching
        that genuinely needs a video backbone.
        """
        from openwam.model.architectures.utils.common import downsample_video_mask_to_latent

        # ``latent[0]`` is a clean conditioning frame (and must be excluded
        # from the loss mask) when either:
        #   (a) the input batch carries ``first_frame_latents`` (Wan TI2V
        #       / cosmos_predict25 TI2V — per-batch signal), in which case
        #       ``base.compute_loss`` will clean-replace ``latents[:, :, 0:1]``
        #       on every step; or
        #   (b) the backbone's *configuration* always reserves ``latent[0]``
        #       for conditioning (only TI2V via the
        #       ``fuse_vae_embedding_in_latents`` / per-token-t=0 path
        #       today).
        # CosmosPredict25 T2V: no first-frame conditioning at all — both
        # signals off.
        skip_first = inputs.get("first_frame_latents") is not None or self.video_backbone.needs_first_frame_skip
        # Pass the backbone's temporal_compression so the tail-grouping
        # divisor matches the actual latent-T produced by the encoder.
        # The default 4 in ``downsample_video_mask_to_latent`` is the Wan
        # VAE legacy; for V-JEPA / other encoders it would silently emit
        # a wrong-length mask. See VideoBackbone.temporal_compression for
        # the source-of-truth contract.
        temporal_factor = int(self.video_backbone.temporal_compression)
        latent_masks = [
            downsample_video_mask_to_latent(~m, temporal_factor=temporal_factor, skip_first=skip_first)
            for m in all_video_masks
        ]
        return torch.stack(latent_masks, dim=0).to(device=self.device)




    def _resolve_inactive_action_dims(
        self, active_action_mask: Optional[Tensor], device: torch.device
    ) -> Optional[Tensor]:
        """Resolve which unified-action dims must ride the analytic noise path.

        An explicit ``active_action_mask`` wins; otherwise the active indices
        are inferred from the attached unify normalizer's scatter map (its
        absence, or a width mismatch with ``action_dim``, disables pinning).
        Returns a bool ``(action_dim,)`` mask of INACTIVE dims, or ``None``
        when every dim is active.
        """
        if active_action_mask is None:
            normalizer = getattr(self, "normalizer", None)
            active_action_indices = getattr(normalizer, "_dst_index", None)
            unified_action_dim = getattr(normalizer, "_unify_dim", None)
            if (
                active_action_indices is not None
                and unified_action_dim is not None
                and int(unified_action_dim) == self.action_dim
            ):
                active_action_indices = torch.as_tensor(active_action_indices, device=device, dtype=torch.long)
                if active_action_indices.numel() and (
                    int(active_action_indices.min()) < 0 or int(active_action_indices.max()) >= self.action_dim
                ):
                    raise ValueError(
                        f"Unified action indices must be within [0, {self.action_dim}); "
                        f"got {active_action_indices.tolist()}."
                    )
                active_action_mask = torch.zeros(self.action_dim, device=device, dtype=torch.bool)
                active_action_mask[active_action_indices] = True

        if active_action_mask is None:
            return None
        active_action_mask = torch.as_tensor(active_action_mask, device=device, dtype=torch.bool)
        if active_action_mask.shape != (self.action_dim,):
            raise ValueError(
                f"active_action_mask must have shape ({self.action_dim},); got {tuple(active_action_mask.shape)}."
            )
        inactive_action_dims = ~active_action_mask
        if not bool(inactive_action_dims.any()):
            return None
        return inactive_action_dims


    # --- §15: Classifier-Free Guidance helpers (inference-time) ---




    @abstractmethod
    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the backbone + Action Expert forward.

        Each concrete architecture implements its own forward end-to-end and
        returns ``(video_noise_pred, action_noise_pred)``; the Fixed-16 family
        has no video stream, so the first element is ``None``.
        """
        ...




# Keys in ``inputs_shared`` that carry a leading batch axis and therefore
# need duplication when stacking ``[uncond, cond]`` for cfg_merge=True.
_CFG_BATCH_AXIS_KEYS: tuple = (
    "latents",
    "input_latents",
    "proprio",
    "first_frame_latents",
    "seq_lens",
    "context_mask",
    # cosmos_predict25 TI2V emits ``condition_mask`` of shape (B, 1, T_lat, H_lat, W_lat)
    # in ``_finalize_ti2v_inputs`` and the wrapper cats it to ``x_in`` along
    # dim=1; cfg_merge=True must double B here or that cat shape-mismatches.
    "condition_mask",
)


