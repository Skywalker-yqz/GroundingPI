"""Cosmos-Predict2.5 backbone with StarVLA's fixed layerwise pi action expert."""

from __future__ import annotations

from typing import Optional

import torch

from starVLA.model.framework.WanPI import Wan_PI
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model
from starVLA.model.modules.projector.PiBackboneConditioner import (
    PiBackboneConditioner,
    normalized_depth_indices,
)
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("CosmosPI")
class Cosmos_PI(Wan_PI):
    """Feed eight normalized-depth Cosmos DiT features to the native StarVLA pi head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # Deliberately do not call Wan_PI.__init__: Cosmos is an independent
        # world-model implementation and does not modify the Wan architecture.
        baseframework.__init__(self)
        self.config = config
        self.backbone = get_world_model(config)
        self.tap_indices = normalized_depth_indices(self.backbone.num_layers, 8)
        self.backbone_conditioner = PiBackboneConditioner(self.backbone.hidden_size)

        action_cfg = self.config.framework.action_model
        action_cfg.hidden_size = 1024
        action_cfg.DiTConfig = {
            "num_layers": 16,
            "input_embedding_dim": 1024,
            "attention_head_dim": 64,
            "num_attention_heads": 16,
        }
        action_cfg.diffusion_model_cfg.num_layers = 16
        action_cfg.diffusion_model_cfg.num_attention_heads = 16
        action_cfg.diffusion_model_cfg.attention_head_dim = 64
        action_cfg.diffusion_model_cfg.output_dim = 1024
        action_cfg.diffusion_model_cfg.cross_attention_dim = 1024
        action_cfg.diffusion_model_cfg.interleave_self_attention = True
        action_cfg.num_target_vision_tokens = 32
        action_cfg.repeated_diffusion_steps = int(action_cfg.get("repeated_diffusion_steps", 2))
        action_cfg.action_expert_fp32 = bool(action_cfg.get("action_expert_fp32", True))
        self.action_model = get_action_model(self.config)
        self.action_horizon = int(action_cfg.future_action_window_size) + 1

    def _extract_conditions(self, images, instructions):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            taps = self.backbone.extract_features(images, instructions, self.tap_indices)
            return self.backbone_conditioner(taps)


__all__ = ["Cosmos_PI"]
