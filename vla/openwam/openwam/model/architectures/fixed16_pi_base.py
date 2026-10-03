"""Shared base for Fixed-16 π-style Layerwise Action DiT architectures.

Every backbone under evaluation — video-generation or vision-language — is
coupled to a **byte-identical** Action Expert: 16 atomic transformer blocks
running ``Cross → Self`` alternately (8 layer-wise cross-attention blocks, 8
action self-attention blocks, 16 FFNs), 1024 residual width, 16 heads × 64, FFN
4096. Only the backbone-specific width projector differs between runs, and it is
reported separately.

This base owns everything that must not vary across backbones:

- construction of the Action Expert and the :class:`BackboneConditioner`,
- the flow-matching training step and the action-only loss,
- the state-token path,
- the cached-condition inference loop and its latency breakdown.

A subclass supplies exactly two things:

- :meth:`encode_conditions` — run its backbone once and return the 8 conditions,
- :meth:`_prepare_forward_inputs` — any backbone-specific massaging of the
  pipeline inputs (pinning a video timestep, passing VLM processor tensors, …).

Lives at the ``architectures`` top level rather than inside ``dual_system/`` so
sibling packages can import it without a circular dependency.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.backbone_conditioner import BackboneConditioner
from openwam.model.action_backbone.fixed16_pi_action_dit import Fixed16PiActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import _cfg_get

logger = logging.getLogger(__name__)


def fixed16_pi_options(cfg) -> dict:
    """``options_from_cfg`` shared by every Fixed-16 π-style registry entry."""
    return {
        "condition_sampling": str(_cfg_get(cfg, "condition_sampling", "normalized_depth")),
        "detach_backbone_features": bool(_cfg_get(cfg, "detach_backbone_features", False)),
    }


#: Geometry report §9.1 fixes across every backbone under comparison. Present in
#: yaml so the config surface matches §12, but a value differing from the pinned
#: one is refused unless ``allow_geometry_override: true`` — changing any of these
#: silently voids the comparison without producing a single error.
PINNED_GEOMETRY: dict = {
    "num_blocks": 16,
    "hidden_dim": 1024,
    "num_attention_heads": 16,
    "attention_head_dim": 64,
    "ffn_dim": 4096,
    "num_depth_taps": 8,
    "condition_hidden_dim": 1024,
    "tokens_per_tap": 64,
    "num_planning_tokens": 32,
    "max_seq_len": 1024,
}

#: How backbone hidden states reach the Action Expert. One knob rather than five,
#: because the two pathways are not meant to be mixed and every intermediate
#: combination is a configuration nobody has run.
#:
#: ``report`` — the design document's §2.1/§3/§8 pathway, and what every
#: checkpoint so far was trained with: ``LN(H)·W_B + e_j`` then a shared 64-query
#: resampler, position embeddings over the whole ``[state; planning; action]``
#: stack.
#:
#: ``starvla`` — starVLA's WanPI pathway: one shared ``Linear(D_B, 1024)`` and
#: nothing else, conditions keeping the backbone's own sequence (its padding mask
#: travels with them into the cross-attention), position embeddings on the action
#: tokens alone.
#:
#: The two produce different parameter sets — ``report`` has the resampler's
#: 4,265,984 parameters and 8,192 depth embeddings, ``starvla`` has neither — so
#: their checkpoints are not interchangeable. Use the SAME pathway for every
#: backbone in a comparison.
CONDITION_PATHWAYS: dict = {
    "report": {
        "use_resampler": True,
        "add_depth_embedding": True,
        "depth_norm_mode": "parameter_free",
        "position_embedding_scope": "sequence",
    },
    "starvla": {
        "use_resampler": False,
        "add_depth_embedding": False,
        "depth_norm_mode": "none",
        "position_embedding_scope": "action",
    },
}

#: Structural properties the implementation has no switch for. They appear in
#: §12 as yaml keys, so they are read and checked — but ``allow_geometry_override``
#: does **not** unlock them: there is no alternative implementation to select, so
#: accepting a different value would be a lie rather than a configuration.
STRUCTURAL_INVARIANTS: dict = {
    "interleave_self_attention": True,
    "first_block": "cross",
    "attention_bias": True,
    "norm_type": "ada_norm",
    "norm_elementwise_affine": False,
    "internal_dit_output_head": False,
    "share_projector_across_depth": True,
    "share_resampler_across_depth": True,
    "raw_context_cross_attention": False,
    "decode_action_positions_only": True,
}


def resolve_fixed16_geometry(cfg) -> dict:
    """Read §12's geometry keys, refusing any deviation that is not opted into.

    Returns the resolved :data:`PINNED_GEOMETRY` values. Absent keys take the
    pinned value, so an untouched config always yields the protocol geometry.
    """
    for key, canonical in STRUCTURAL_INVARIANTS.items():
        value = _cfg_get(cfg, key, canonical)
        if value != canonical:
            raise ValueError(
                f"{key}={value!r} is not implementable: the Fixed-16 π-style stack hard-codes "
                f"{key}={canonical!r} and offers no alternative path. `allow_geometry_override` does "
                "not unlock this — accepting the value would misreport what actually runs."
            )

    allow_override = bool(_cfg_get(cfg, "allow_geometry_override", False))
    resolved, changed = {}, []
    for key, pinned in PINNED_GEOMETRY.items():
        if key == "condition_hidden_dim":
            continue  # resolved below, defaulting to whatever hidden_dim became
        value = type(pinned)(_cfg_get(cfg, key, pinned))
        if value != pinned:
            if not allow_override:
                raise ValueError(
                    f"{key}={value} differs from the protocol's pinned {key}={pinned}. The Action "
                    "Expert must be byte-identical across every backbone being compared, so this is "
                    "refused by default. Set `allow_geometry_override: true` if you are deliberately "
                    "running an off-protocol ablation."
                )
            changed.append(f"{key}: {pinned} -> {value}")
        resolved[key] = value

    # Follows hidden_dim rather than its own pinned value, so an override of the
    # residual width does not also require remembering to move this one. The two
    # have to agree anyway (checked next).
    resolved["condition_hidden_dim"] = int(_cfg_get(cfg, "condition_hidden_dim", resolved["hidden_dim"]))
    if resolved["condition_hidden_dim"] != PINNED_GEOMETRY["condition_hidden_dim"]:
        changed.append(
            f"condition_hidden_dim: {PINNED_GEOMETRY['condition_hidden_dim']} -> {resolved['condition_hidden_dim']}"
        )
    if resolved["condition_hidden_dim"] != resolved["hidden_dim"]:
        raise ValueError(
            f"condition_hidden_dim={resolved['condition_hidden_dim']} must equal "
            f"hidden_dim={resolved['hidden_dim']}: the conditioner projects every tap to the Action "
            "residual width so Cross and Self blocks have identical parameter counts (report §7.1)."
        )
    if changed:
        logger.warning(
            "[Fixed-16 π-style] allow_geometry_override=true — OFF-PROTOCOL geometry: %s. "
            "The Action Expert is no longer byte-identical to on-protocol runs; do not pool the results.",
            "; ".join(changed),
        )
    return resolved


class Fixed16PiArchitectureBase(BaseWAMArchitecture):
    """Backbone-agnostic half of a Fixed-16 π-style architecture."""

    #: Key used for the evaluated backbone in :meth:`parameter_report`. Subclasses
    #: override so the report names the backbone that is actually under test.
    _evaluated_backbone_name: str = "backbone"

    #: No joint denoising here — the backbone runs once and only the action flow
    #: is integrated, so deploy must not build a video/action schedule. On the VLM

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def setup_fixed16_action_stack(self, cfg, *, backbone_hidden_dim: int, num_backbone_blocks: int) -> None:
        """Build the Action Expert and the backbone conditioner.

        Call from the subclass ``__init__`` once the backbone geometry
        (``D_B``, ``N_B``) is known.
        """
        use_proprio = bool(_cfg_get(cfg, "use_proprioception", False))
        state_dim = int(_cfg_get(cfg, "state_dim", 0) or 0) if use_proprio else 0
        if use_proprio and state_dim <= 0:
            raise ValueError("use_proprioception=true requires state_dim > 0 for the action state token.")

        # How many leading frames of the clip count as "the observation". Every
        # backbone must see the same pixels, and the VLM path can only take one
        # image, so 1 is the only value that keeps the comparison honest.
        self.num_observation_frames = int(_cfg_get(cfg, "num_observation_frames", 1))
        if self.num_observation_frames < 1:
            raise ValueError(f"num_observation_frames must be >= 1, got {self.num_observation_frames}")
        if self.num_observation_frames % 4 != 1:
            # Wan's check_resize_height_width silently rounds num_frames up to the
            # next 4k+1, so 2/3/4 would quietly become 5 (T_lat 1 -> 2) and the
            # two backbones would stop seeing the same thing without any error.
            raise ValueError(
                f"num_observation_frames must satisfy n % 4 == 1 (1, 5, 9, ...), got "
                f"{self.num_observation_frames}. Wan rounds other values up silently, which would "
                "change the latent time dimension behind your back."
            )

        # Emitted once per process, not per step.
        self._logged_action_clip = False

        geometry = resolve_fixed16_geometry(cfg)

        pathway_name = str(_cfg_get(cfg, "condition_pathway", "report"))
        if pathway_name not in CONDITION_PATHWAYS:
            raise ValueError(
                f"Unknown condition_pathway {pathway_name!r}. Choose from: {', '.join(CONDITION_PATHWAYS)}"
            )
        pathway = dict(CONDITION_PATHWAYS[pathway_name])
        self.condition_pathway = pathway_name

        self.action_backbone = Fixed16PiActionDiT(
            action_dim=int(_cfg_get(cfg, "action_dim", 80)),
            state_dim=state_dim,
            # The protocol only requires this to be fixed within a benchmark and
            # identical across backbones; the value itself is a config choice.
            action_horizon=int(_cfg_get(cfg, "action_horizon", 32)),
            num_inference_steps=int(_cfg_get(cfg, "num_inference_steps", 10)),
            noise_beta_alpha=float(_cfg_get(cfg, "noise_beta_alpha", 1.5)),
            noise_beta_beta=float(_cfg_get(cfg, "noise_beta_beta", 1.0)),
            noise_s=float(_cfg_get(cfg, "noise_s", 0.999)),
            num_timestep_buckets=int(_cfg_get(cfg, "num_timestep_buckets", 1000)),
            num_planning_tokens=geometry["num_planning_tokens"],
            max_seq_len=geometry["max_seq_len"],
            dim=geometry["hidden_dim"],
            ffn_dim=geometry["ffn_dim"],
            num_heads=geometry["num_attention_heads"],
            head_dim=geometry["attention_head_dim"],
            num_blocks=geometry["num_blocks"],
            compute_fp32=bool(_cfg_get(cfg, "action_expert_fp32", False)),
            position_embedding_scope=pathway["position_embedding_scope"],
        )

        # Optional warm start from a pretrained video DiT. Off unless the config
        # names a source; see interpolated_init for why the source must be the
        # same for every backbone in a comparison.
        from openwam.model.action_backbone.interpolated_init import maybe_interpolate_init

        maybe_interpolate_init(self.action_backbone, cfg, _cfg_get)

        # How many flow timesteps to draw per sample off one backbone forward.
        # 1 = the previous behaviour. starVLA's LayerwiseFM configs use 2-8.
        self.repeated_diffusion_steps = int(_cfg_get(cfg, "repeated_diffusion_steps", 1) or 1)
        if self.repeated_diffusion_steps < 1:
            raise ValueError(f"repeated_diffusion_steps must be >= 1, got {self.repeated_diffusion_steps}")

        if num_backbone_blocks <= 0:
            raise ValueError("The backbone block count must be known so the 8 normalized-depth taps can be placed.")
        self.backbone_conditioner = BackboneConditioner(
            backbone_hidden_dim=backbone_hidden_dim,
            num_backbone_blocks=num_backbone_blocks,
            dim=geometry["condition_hidden_dim"],
            num_depth_taps=geometry["num_depth_taps"],
            tokens_per_tap=geometry["tokens_per_tap"],
            sampling=str(_cfg_get(cfg, "condition_sampling", "normalized_depth")),
            add_depth_embedding=pathway["add_depth_embedding"],
            depth_norm_mode=pathway["depth_norm_mode"],
            use_resampler=pathway["use_resampler"],
        )
        self._required_blocks = frozenset(self.backbone_conditioner.required_block_indices)

        logger.info(
            "[Fixed-16 π-style] %s | backbone N_B=%d D_B=%d | sampling=%s taps=%s | "
            "pathway=%s (%s)",
            self.action_backbone.topology_summary(),
            num_backbone_blocks,
            backbone_hidden_dim,
            self.backbone_conditioner.sampling,
            self.backbone_conditioner.tap_indices,
            pathway_name,
            f"{self.backbone_conditioner.tokens_per_tap} tokens/tap"
            if pathway["use_resampler"]
            else "conditions keep the backbone sequence",
        )

    @property
    def _condition_padding_mask(self):
        """Padding mask for the conditions the conditioner produced last.

        ``None`` on the ``report`` pathway — the resampler already absorbed it and
        the Action Expert sees 64 valid tokens. On ``starvla`` the conditions carry
        the backbone's sequence with its padded tail, so the mask has to reach the
        cross-attention. Read off the conditioner rather than threaded through
        every signature, and only valid right after ``encode_conditions``.
        """
        conditioner = getattr(self, "backbone_conditioner", None)
        return None if conditioner is None else conditioner.last_key_padding_mask

    def _check_adaptation_protocol(self) -> None:
        """Refuse a half-set adaptation protocol, once, at the first forward.

        Protocol A needs two switches thrown together: ``detach_backbone_features``
        here and the backbone's entry in the model config's ``freeze:`` list. Set
        only the first and the backbone is nominally trainable but receives no
        gradient — the optimizer carries states for parameters that never move and
        the parameter report claims an adaptation that is not happening. Set only
        the second and the detach is redundant but harmless.

        Neither shows up as an error anywhere, and both change what the numbers in
        the comparison table mean, so this is checked rather than documented. The
        check runs at forward because freezing happens after construction.
        """
        if getattr(self, "_adaptation_protocol_checked", False):
            return
        self._adaptation_protocol_checked = True

        backbone = self.evaluated_backbone
        if backbone is None:
            return
        detached = bool(getattr(self, "_detach_backbone_features", False))
        trainable = any(p.requires_grad for p in backbone.parameters())
        if detached and trainable:
            raise ValueError(
                f"detach_backbone_features=true but {self._evaluated_backbone_name} still has trainable "
                "parameters. Protocol A needs both halves: add the backbone to the model config's "
                "`freeze:` list (vlm_system.yaml freezes `vlm_backbone.vlm_model`, "
                "dual_system_fixed16.yaml freezes `video_backbone.dit`). As it stands the backbone "
                "would be carried by the optimizer while receiving no gradient."
            )
        if not detached and not trainable:
            raise ValueError(
                f"detach_backbone_features=false but {self._evaluated_backbone_name} is fully frozen, so "
                "the evaluated backbone receives no gradient. The comparison is about how each "
                "representation adapts under the action loss, and a run configured like this silently "
                "measures something else. Drop the backbone from the model config's `freeze:` list to "
                "co-train it, or set detach_backbone_features=true to state that the probe is intended."
            )

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @property
    def evaluated_backbone(self):
        """The backbone whose representation is under test. Subclasses override."""
        return self.video_backbone

    @property
    def backbones(self) -> dict:
        """Include the conditioner so dtype/device moves and deploy saves reach it.

        It is listed separately from ``action_backbone`` on purpose: its width
        projector is backbone-specific, so its parameters must never be folded
        into the Action Expert's reported 242.7M.
        """
        result = super().backbones
        conditioner = getattr(self, "backbone_conditioner", None)
        if conditioner is not None:
            result["backbone_conditioner"] = conditioner
        return result

    def parameter_report(self) -> dict:
        """Per-group trainable parameter counts, as the checklist asks them to be reported."""

        def count(module) -> int:
            return 0 if module is None else sum(p.numel() for p in module.parameters())

        conditioner = getattr(self, "backbone_conditioner", None)
        report = {
            "action_head": count(getattr(self, "action_backbone", None)),
            "backbone_projector": 0 if conditioner is None else count(conditioner.projector),
            "token_resampler": 0 if conditioner is None else count(conditioner.resampler),
            "depth_embeddings": 0
            if conditioner is None or conditioner.depth_embedding is None
            else conditioner.depth_embedding.numel(),
            self._evaluated_backbone_name: count(self.evaluated_backbone),
        }
        report["total_trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report

    # ------------------------------------------------------------------
    # Backbone hooks — the only things a subclass must supply
    # ------------------------------------------------------------------

    def encode_conditions(self, **inputs) -> list[Tensor]:
        """Run the backbone once and return ``num_depth_taps`` conditions of ``(B, 64, 1024)``."""
        raise NotImplementedError(f"{type(self).__name__} must implement encode_conditions().")

    def _truncate_to_observation(self, batch: list[dict]) -> list[dict]:
        """Keep only the leading ``num_observation_frames`` of each sample's clip.

        The protocol compares representations of **one observation**, so every
        backbone must be handed the same pixels. Truncating here — before the VAE
        — rather than slicing latents afterwards is deliberate: the VAE is
        temporally causal with a factor of 4, so ``encode(clip)[:, :, 0]`` is a
        blend of the first four frames, not the encoding of frame 0.

        ``video_mask`` is truncated alongside ``video``; leaving it at full length
        makes ``downsample_video_mask_to_latent`` emit a mask that no longer
        matches the latent grid.

        Returns shallow copies — the caller's sample dicts are not mutated, since
        ``prepare_inputs`` further down mutates what it is given.
        """
        keep = self.num_observation_frames
        truncated = []
        for sample in batch:
            video = sample.get("video")
            if video is None:
                truncated.append(sample)
                continue
            if len(video) < keep:
                raise ValueError(f"sample['video'] has {len(video)} frames but num_observation_frames={keep}.")
            trimmed = dict(sample)
            trimmed["video"] = video[:keep]
            mask = sample.get("video_mask")
            if mask is not None:
                trimmed["video_mask"] = mask[..., :keep]
            truncated.append(trimmed)
        return truncated

    def _prepare_forward_inputs(self, forward_inputs: dict, *, batch_size: int) -> dict:
        """Backbone-specific massaging of the pipeline inputs before ``forward``.

        Default is a pass-through. The video variant uses this to pin the
        feature-extraction latents and timestep.
        """
        return forward_inputs

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def _clip_to_action_horizon(self, actions: Tensor, action_is_pad: Optional[Tensor]):
        """Make ``action_horizon`` authoritative over whatever the dataloader shipped.

        The dataloader's chunk length is ``num_frames - 1``, which is a separate
        knob; without this, training would silently use the dataloader's length
        while :meth:`Fixed16PiActionDiT.predict_action` generates
        ``action_horizon`` steps at inference, and nothing would report the
        mismatch.

        The leading steps are kept, not the trailing ones: the observation is
        frame 0, so the chunk to predict is t=1..H. (Slicing ``[-H:]`` — what a
        reference implementation whose dataloader already emits exactly H steps
        can get away with — would pair a t=0 observation with a chunk starting
        mid-window.)
        """
        horizon = self.action_backbone.action_horizon
        available = actions.shape[1]
        if available < horizon:
            raise ValueError(
                f"action_horizon={horizon} but the dataloader only supplied {available} action steps. "
                f"Raise the dataloader's num_frames to at least {horizon + 1}, or lower action_horizon."
            )
        if available == horizon:
            return actions, action_is_pad
        if not self._logged_action_clip:
            logger.info(
                "Clipping the dataloader's %d action steps to action_horizon=%d (keeping t=1..%d).",
                available,
                horizon,
                horizon,
            )
            self._logged_action_clip = True
        actions = actions[:, :horizon]
        if action_is_pad is not None and action_is_pad.shape[1] == available:
            action_is_pad = action_is_pad[:, :horizon]
        return actions, action_is_pad

    def _state_token(self, proprio: Optional[Tensor]) -> Optional[Tensor]:
        """Shape proprio into the ``(B, 1, state_dim)`` the state encoder wants."""
        if self.action_backbone.state_encoder is None:
            return None
        if proprio is None:
            raise ValueError("use_proprioception=true but no proprio was provided.")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        if proprio.ndim != 3 or proprio.shape[1] != 1:
            raise ValueError(f"proprio must be [D], [B, D] or [B, 1, D], got {tuple(proprio.shape)}")
        return proprio.to(device=self.device, dtype=self.dtype)

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Returns ``(None, velocity)`` — no architecture in this family predicts video.

        ``action_timestep`` is the discretized flow bucket ``(B,)``; a float
        tensor in ``[0, 1]`` is accepted and bucketized.
        """
        pipeline_inputs.pop("_proprio_sample_mask", None)
        # Lets compute_loss reuse one backbone forward across several flow draws
        # (repeated_diffusion_steps) without calling encode_conditions itself.
        conditions_out = pipeline_inputs.pop("_conditions_out", None)
        self._check_adaptation_protocol()
        conditions = self.encode_conditions(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        if conditions_out is not None:
            conditions_out["conditions"] = conditions
        if noisy_actions is None:
            return None, None

        ab = self.action_backbone
        if action_timestep is None:
            raise ValueError("action_timestep is required when noisy_actions is provided.")
        buckets = action_timestep
        buckets = ab.to_buckets(buckets) if torch.is_floating_point(buckets) else buckets.long()
        return None, ab.predict_velocity(
            noisy_actions,
            buckets,
            conditions,
            self._state_token(proprio),
            condition_padding_mask=self._condition_padding_mask,
        )

    def compute_loss(
        self,
        *,
        actions: Optional[Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        decoupled_sampler=None,
        **inputs,
    ) -> dict:
        """Action-only flow-matching loss on top of pinned backbone features.

        ``lambda_video`` and ``decoupled_sampler`` are accepted for interface
        compatibility and ignored: this family does not train a video stream, so
        a non-zero video weight is a configuration error rather than something to
        silently honour.
        """
        arch_name = type(self).__name__
        if lambda_video:
            logger.warning(
                "%s does not train a video stream; ignoring lambda_video=%s. "
                "Set training.lambda_video=0 to silence this.",
                arch_name,
                lambda_video,
            )
        if decoupled_sampler is not None:
            logger.warning("decoupled timestep sampling does not apply to the fixed flow schedule; ignoring.")

        ab = self.action_backbone
        if actions is None:
            actions = inputs.pop("actions", None)
        else:
            inputs.pop("actions", None)
        if actions is None or lambda_action == 0:
            raise ValueError(f"{arch_name} requires actions and lambda_action > 0.")

        inputs.pop("max_timestep_boundary", None)
        inputs.pop("min_timestep_boundary", None)

        forward_inputs = dict(inputs)
        proprio = forward_inputs.pop("proprio", None)
        forward_inputs.pop("proprio_mask", None)
        action_is_pad = forward_inputs.pop("action_is_pad", None)
        forward_inputs.pop("video_is_pad", None)
        use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
        use_grad_ckpt_offload = forward_inputs.pop("use_gradient_checkpointing_offload", False)

        actions = actions.to(dtype=self.dtype, device=self.device)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        actions, action_is_pad = self._clip_to_action_horizon(actions, action_is_pad)

        forward_inputs = self._prepare_forward_inputs(forward_inputs, batch_size=actions.shape[0])

        if self.repeated_diffusion_steps <= 1:
            noisy_actions, velocity_target, buckets = ab.sample_flow_batch(actions)
            _, predicted = self(
                noisy_actions,
                buckets,
                proprio=proprio,
                use_gradient_checkpointing=use_grad_ckpt,
                use_gradient_checkpointing_offload=use_grad_ckpt_offload,
                **forward_inputs,
            )
            loss_action = ab.masked_mse(predicted, velocity_target, action_is_pad)
        else:
            # K flow timesteps per sample off ONE backbone forward. Flow matching's
            # sample efficiency is bounded by the number of (sample, t) pairs, not
            # by the number of samples, and the backbone dominates the cost — so
            # the extra supervision is nearly free: only the 243M Action Expert
            # runs K times, the multi-billion backbone runs once.
            # First draw goes through ``self(...)`` so nn.Module.__call__ runs and
            # any architecture-level hook fires exactly as it does at K=1; the
            # conditions it computed are then reused for the remaining draws, so
            # the backbone still runs once.
            losses = []
            noisy_actions, velocity_target, buckets = ab.sample_flow_batch(actions)
            conditions_box = {}
            _, predicted = self(
                noisy_actions,
                buckets,
                proprio=proprio,
                use_gradient_checkpointing=use_grad_ckpt,
                use_gradient_checkpointing_offload=use_grad_ckpt_offload,
                _conditions_out=conditions_box,
                **forward_inputs,
            )
            losses.append(ab.masked_mse(predicted, velocity_target, action_is_pad))
            conditions = conditions_box["conditions"]
            state = self._state_token(proprio)
            # Captured once: the extra draws reuse the same backbone forward, so
            # they reuse its mask too.
            padding_mask = self._condition_padding_mask
            for _ in range(self.repeated_diffusion_steps - 1):
                noisy_actions, velocity_target, buckets = ab.sample_flow_batch(actions)
                predicted = ab.predict_velocity(
                    noisy_actions, buckets, conditions, state, condition_padding_mask=padding_mask
                )
                losses.append(ab.masked_mse(predicted, velocity_target, action_is_pad))
            # Mean, not sum: the loss scale stays comparable to K=1 so the learning
            # rate carries over unchanged.
            loss_action = torch.stack(losses).mean()
        loss = lambda_action * loss_action
        return {
            "loss": loss,
            "loss_video": torch.zeros((), device=loss.device),
            "loss_action": loss.detach(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(
        self,
        *,
        proprio: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        profile: bool = False,
        seed: Optional[int] = None,
        **pipeline_inputs,
    ) -> dict:
        """One backbone forward, then ``num_inference_steps`` action steps.

        Reports the three latencies separately, as the protocol requires: folding
        backbone feature extraction into the action-denoising number would make
        backbones with different token counts look like they have different
        action heads.

        ``seed`` fixes the flow's initial noise. Two backbones evaluated on the
        same observation otherwise start from different noise, and part of the
        difference in their actions is that draw rather than their representations.
        """
        # Restored on the way out. Leaving the model in eval would be invisible
        # from here but not harmless: the VLM path gates gradient checkpointing on
        # ``self.training``, so a mid-training evaluation would switch it off for
        # good and the run would grow its activation footprint until it OOMs.
        was_training = self.training
        self.eval()
        try:
            return self._predict_action_chunk(
                proprio=proprio, batch_size=batch_size, profile=profile, seed=seed, **pipeline_inputs
            )
        finally:
            self.train(was_training)

    def _predict_action_chunk(
        self,
        *,
        proprio: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        profile: bool = False,
        seed: Optional[int] = None,
        **pipeline_inputs,
    ) -> dict:
        pipeline_inputs = dict(pipeline_inputs)
        pipeline_inputs.pop("proprio", None)
        pipeline_inputs.pop("action_is_pad", None)
        pipeline_inputs.pop("video_is_pad", None)
        batch_size = self._infer_batch_size(pipeline_inputs) if batch_size is None else int(batch_size)
        pipeline_inputs = self._prepare_forward_inputs(pipeline_inputs, batch_size=batch_size)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        def _sync() -> None:
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()

        _sync()
        t_start = time.time()
        conditions = self.encode_conditions(**pipeline_inputs)
        _sync()
        t_features = time.time()

        # Inactive unified-action dims are masked out of the training loss, so the
        # head never learns a velocity for them. Left free they drift on every
        # Euler step and feed back into the supervised dims through the action
        # self-attention — worse the more steps there are. Hold them.
        inactive = self._resolve_inactive_action_dims(None, self.device)
        active_action_mask = None if inactive is None else ~inactive

        actions = self.action_backbone.predict_action(
            conditions,
            state=self._state_token(proprio),
            batch_size=batch_size,
            generator=generator,
            active_action_mask=active_action_mask,
            condition_padding_mask=self._condition_padding_mask,
        )
        _sync()
        t_end = time.time()

        latency = {
            "backbone_feature_extraction_s": t_features - t_start,
            "action_denoising_s": t_end - t_features,
            "end_to_end_s": t_end - t_start,
        }
        if profile:
            logger.info(
                "[WAM_PROFILE] backbone=%.4fs action=%.4fs end_to_end=%.4fs",
                latency["backbone_feature_extraction_s"],
                latency["action_denoising_s"],
                latency["end_to_end_s"],
            )

        out = actions.float().cpu().numpy()
        if out.shape[0] == 1:
            # Squeeze unconditionally, matching BaseWAMArchitecture.generate: a
            # single request returns (T, action_dim), and the deploy server must
            # not receive a different rank depending on whether a normalizer
            # happens to be attached.
            out = out.squeeze(0)
        # Outside the squeeze: nesting it inside meant a batched request came back
        # in normalized model space, with nothing to distinguish it from real
        # actions. The normalizer scales the last axis, so it applies to both
        # (T, action_dim) and (B, T, action_dim).
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            out = normalizer.unnormalize(out)
        return {"video": None, "actions": out, "latency": latency}

    def _infer_batch_size(self, pipeline_inputs: dict) -> int:
        """Subclasses that cannot derive B from the pipeline inputs must override."""
        raise NotImplementedError(f"{type(self).__name__} must implement _infer_batch_size().")

    def generate(
        self,
        schedule=None,
        prompt: str = "",
        *,
        first_frame_image=None,
        proprio=None,
        profile: bool = False,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """Deploy entry point: adapts the engine's call onto the protocol's loop.

        ``JointInferenceEngine`` is written against
        ``BaseWAMArchitecture.generate``, which re-runs the full video DiT at
        every denoising step and decodes video. This family does neither — one
        backbone forward per observation, then ``num_inference_steps`` action
        steps over the cached conditions — so the engine's video arguments
        (``schedule``, ``decode_video``, ``tiled``, ``num_frames``,
        …) are accepted and dropped rather than honoured. Returning the same
        ``{"video", "actions", "latency"}`` dict keeps the server unchanged.

        The observation is ``first_frame_image[0]``, the same frame training
        feeds both backbones, routed through this architecture's own
        ``prepare_inputs`` so deploy and training build the inputs identically.
        """
        if not first_frame_image:
            raise ValueError(
                f"{type(self).__name__}.generate requires `first_frame_image`: the observation frame is "
                "the only visual input this family reads. The deploy request must carry one."
            )

        pinned_steps = self.action_backbone.num_inference_steps
        if num_inference_steps is not None and int(num_inference_steps) != pinned_steps:
            if not getattr(self, "_warned_deploy_steps", False):
                self._warned_deploy_steps = True
                logger.warning(
                    "Ignoring deploy num_inference_steps=%s: the protocol pins the action flow at %d Euler "
                    "steps for every backbone (report §9.1), so honouring a per-request value would make "
                    "deployed backbones incomparable. Change it in the model config instead.",
                    num_inference_steps,
                    pinned_steps,
                )

        # Before prepare_inputs, not just before the denoising loop. The video
        # path's prepare_inputs runs the text encoder, whose dropout is live in
        # train mode — a freshly loaded checkpoint is in train mode, so the same
        # instruction encoded twice gave two different contexts and the same seed
        # did not reproduce an action. predict_action_chunk's own eval() came too
        # late to prevent that. Restored below, for the reason given there.
        was_training = self.training
        self.eval()

        images = list(first_frame_image)
        inputs = self.prepare_inputs([{"video": images[:1], "prompt": prompt}])
        for key in ("actions", "action_is_pad", "proprio", "proprio_mask"):
            inputs.pop(key, None)

        # Deliberately NOT normalize_deploy_proprio(proprio): JointInferenceEngine
        # already normalized at engine.py:330 before calling generate, and the
        # transform is affine, not idempotent -- applying q99 twice turned a state
        # of 0.25 into 1.0 instead of 0.5, so the state token the policy
        # conditioned on was wrong by 100% and saturated near the edges of the
        # range. BaseWAMArchitecture.generate does not normalize here either: the
        # engine owns it. What that call also did, and what is still needed, is
        # the array-to-tensor conversion for callers that hand in a raw array.
        if proprio is not None and not torch.is_tensor(proprio):
            import numpy as np

            proprio = torch.from_numpy(np.asarray(proprio, dtype=np.float32))

        try:
            return self.predict_action_chunk(
                proprio=proprio,
                batch_size=1,
                profile=profile,
                seed=seed,
                **inputs,
            )
        finally:
            self.train(was_training)


__all__ = ["Fixed16PiArchitectureBase", "fixed16_pi_options"]
