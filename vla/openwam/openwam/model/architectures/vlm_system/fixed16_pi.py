"""Fixed-16 π-style Layerwise Action DiT over a vision-language backbone.

The VLM sibling of ``dual_system_fixed16_pi``. Both couple their backbone to a
**byte-identical** Action Expert — same 16 atomic blocks, same width, same flow
schedule, same state path — so the only thing that differs between a run on
Qwen3-VL and a run on Wan2.2 is the representation being tapped. That is the
entire point of the protocol; if the two Action Experts ever diverge, the
comparison is meaningless.

There is **no video stream here at all**: no VAE, no T5, no video loss, no
`generate`. The VLM runs once per observation, 8 normalized-depth hidden states
are tapped, projected and resampled to ``8 × (B, 64, 1024)``, and the Action
Expert consumes them. The instruction reaches control only through those hidden
states (it is already inside the Qwen prompt), and proprioceptive state enters
as a state token in the action sequence — so, as on the video path, there is no
raw-context bypass.

The VLM sees one image per sample: the first frame, via the shared
:func:`~openwam.model.architectures.utils.common.extract_first_image`, which is
the same frame every VLM consumer in this codebase reads.

Registered as ``framework=vlm_system, variant=fixed16_pi_layerwise``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from openwam.model.architectures.fixed16_pi_base import (
    Fixed16PiArchitectureBase,
    fixed16_pi_options,
)
from openwam.model.architectures.registry import _cfg_get, register_architecture
from openwam.model.architectures.utils.common import extract_first_image

logger = logging.getLogger(__name__)


@register_architecture(
    "vlm_system_fixed16_pi",
    status="supported",
    note=(
        "Fixed-16 π-style Layerwise Action DiT on a VLM backbone: 8 normalized-depth taps "
        "→ the same fixed 16-block Action Expert the video path uses. No video stream."
    ),
    framework="vlm_system",
    variant="fixed16_pi_layerwise",
    options_from_cfg=fixed16_pi_options,
)
class VlmSystemFixed16PiArchitecture(Fixed16PiArchitectureBase):
    """VLM backbone as a pinned feature extractor + the unified Action Expert."""

    _evaluated_backbone_name = "vlm_backbone"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.vlm_backbone = None
        if cfg is None:
            return

        # A video backbone here would be silently loaded dead weight: nothing in
        # this architecture reads it. Gate on the built object rather than the
        # cfg key, because the deploy loader legitimately passes an empty dict.
        if self.video_backbone is not None:
            raise ValueError(
                "vlm_system_fixed16_pi must not be given a video backbone — it would load "
                "billions of parameters that never contribute. Remove the `video_backbone` "
                "entry from the model config's defaults list."
            )

        vlm_cfg = _cfg_get(cfg, "vlm_backbone", None)
        if vlm_cfg is not None:
            from omegaconf import DictConfig, OmegaConf

            from openwam.model.vlm_backbone import build_vlm_backbone

            # Forward every yaml field except `name` straight to the wrapper, so
            # family-specific options (LocateAnything's allow_remote_code, …)
            # reach it without this architecture learning about each family.
            if isinstance(vlm_cfg, DictConfig):
                vlm_kwargs = OmegaConf.to_container(vlm_cfg, resolve=True) or {}
            else:
                vlm_kwargs = dict(vlm_cfg)
            # `or` rather than a pop default: an explicit `name: null` in yaml
            # would otherwise sail past and hit the registry as None.
            vlm_name = vlm_kwargs.pop("name", None) or "qwen3_vl"
            vlm_kwargs.setdefault("load_pretrained", True)
            vlm_kwargs.setdefault("max_length", 512)
            self.vlm_backbone = build_vlm_backbone(vlm_name, dtype=self.dtype, **vlm_kwargs)

        # Prefer the loaded backbone's real geometry; fall back to cfg so the
        # architecture can be built before a backbone is attached (the pattern
        # tests use, since a real VLM cannot be loaded on CPU).
        if self.vlm_backbone is not None:
            vlm_dim = int(self.vlm_backbone.hidden_size)
            num_vlm_layers = int(self.vlm_backbone.num_layers)
        else:
            vlm_dim = int(_cfg_get(cfg, "vlm_dim", 0) or 0)
            num_vlm_layers = int(_cfg_get(cfg, "num_vlm_layers", 0) or 0)
        if vlm_dim <= 0 or num_vlm_layers <= 0:
            raise ValueError(
                "vlm_dim / num_vlm_layers must come from the loaded VLM backbone or be given "
                "in config so the 8 normalized-depth taps can be placed."
            )

        self.setup_fixed16_action_stack(cfg, backbone_hidden_dim=vlm_dim, num_backbone_blocks=num_vlm_layers)
        if self.num_observation_frames != 1:
            # Qwen3VLBackbone.format_prompt injects exactly one image block per
            # message, so this path physically cannot consume more than one frame.
            # Letting the video path use more would silently reintroduce the
            # observation-input asymmetry this knob exists to remove.
            raise ValueError(
                f"vlm_system_fixed16_pi supports num_observation_frames=1 only, got "
                f"{self.num_observation_frames}. Multi-image VLM input needs new plumbing in "
                "Qwen3VLBackbone; raising a value here on the video side alone would make the "
                "two backbones see different observations."
            )
        self._detach_backbone_features = bool(_cfg_get(cfg, "detach_backbone_features", False))

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @property
    def evaluated_backbone(self):
        return self.vlm_backbone

    @property
    def backbones(self) -> dict:
        result = super().backbones
        if getattr(self, "vlm_backbone", None) is not None:
            # Named ``vlm_backbone`` so the checkpoint machinery (the
            # ``vlm_backbone.`` state-dict prefix exclusion and the tolerant
            # strict load) and ``save_deploy_assets`` all apply unchanged.
            result["vlm_backbone"] = self.vlm_backbone
        return result

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def preprocess(self, **kwargs) -> dict:
        """Never valid here — there is no video backbone to delegate to."""
        raise NotImplementedError(
            "vlm_system_fixed16_pi has no video backbone and therefore no VAE/text preprocess "
            "step. Batching happens entirely in prepare_inputs()."
        )

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """Batch a list of dataset samples into VLM inputs + action tensors.

        Deliberately does **not** chain to ``BaseWAMArchitecture.prepare_inputs``:
        that path calls ``self.preprocess`` (base.py:837) and dereferences
        ``self.video_backbone`` for the latent mask, neither of which exists
        here. The backbone-independent half is reused via
        ``_collect_sample_tensors`` rather than copied.
        """
        from openwam.dataloader.transforms.pipeline import FirstFrameConditioningTransform

        if isinstance(batch, dict):
            batch = [batch]
        if not batch:
            raise ValueError("vlm_system_fixed16_pi.prepare_inputs: empty batch")

        if not hasattr(self, "_pipeline_transform_instance"):
            self._pipeline_transform_instance = FirstFrameConditioningTransform()
        samples = [self._pipeline_transform_instance.apply(s) for s in batch]

        inputs = self._collect_sample_tensors(samples)
        inputs.update(
            {
                "use_gradient_checkpointing": self._use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self._use_gradient_checkpointing_offload,
            }
        )

        provided = [sample.get("vlm_inputs") for sample in samples]
        if all(item is not None for item in provided):
            inputs["vlm_inputs"] = self.vlm_backbone.batch_vlm_inputs(provided)
            return inputs
        if any(item is not None for item in provided):
            raise ValueError("Mixed vlm_inputs in batch: provide vlm_inputs for all samples or none.")

        prompts = [sample["prompt"] for sample in samples]
        images = [extract_first_image(sample) for sample in samples]
        if any(image is None for image in images):
            raise ValueError("vlm_system_fixed16_pi requires sample['vlm_inputs'] or a first frame image/video[0].")
        inputs["vlm_inputs"] = self.vlm_backbone.prepare_vlm_inputs(prompts, images)
        return inputs

    # ------------------------------------------------------------------
    # Backbone hooks
    # ------------------------------------------------------------------

    def _infer_batch_size(self, pipeline_inputs: dict) -> int:
        vlm_inputs = pipeline_inputs.get("vlm_inputs")
        if vlm_inputs is None:
            raise ValueError("Cannot infer batch size: vlm_inputs is missing.")
        if isinstance(vlm_inputs, list):
            return len(vlm_inputs)
        return int(vlm_inputs["input_ids"].shape[0])

    def encode_conditions(self, **pipeline_inputs) -> list[Tensor]:
        """Run the VLM once and return the 8 cached conditions."""
        if self.vlm_backbone is None:
            raise RuntimeError("vlm_backbone is None — cannot extract conditions.")

        # The video path honours this flag inside its DiT loop; without the same
        # call here the co-trained VLM kept every layer's activations, which is
        # what capped the batch size on the VLM side of the comparison.
        setter = getattr(self.vlm_backbone, "set_gradient_checkpointing", None)
        if setter is not None:
            setter(bool(pipeline_inputs.get("use_gradient_checkpointing", False)) and self.training)

        vlm_inputs = pipeline_inputs.get("vlm_inputs")
        if vlm_inputs is None:
            raise ValueError("vlm_system_fixed16_pi.forward requires `vlm_inputs` in the pipeline inputs.")

        taps, key_padding_mask = self.vlm_backbone.extract_layerwise_features(vlm_inputs, sorted(self._required_blocks))
        if self._detach_backbone_features:
            taps = {block_id: hidden.detach() for block_id, hidden in taps.items()}
        return self.backbone_conditioner(taps, key_padding_mask=key_padding_mask)


__all__ = ["VlmSystemFixed16PiArchitecture"]
