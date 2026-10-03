"""Additional VLM backbones on the native StarVLA PI action architecture.

QwenPI and WanPI are intentionally left untouched.  This framework reuses the
validated QwenPI forward/predict/get_action methods and only replaces model
construction with an adapter that exposes the same StarVLA VLM contract.
"""

from __future__ import annotations

from typing import Optional

from starVLA.model.framework.QwenPI import Qwen_PI
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model
from starVLA.model.modules.projector.PiBackboneConditioner import (
    PiBackboneConditioner,
    normalized_depth_indices,
)
from starVLA.model.modules.vlm.comparison_backbones import get_comparison_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("VLMBackbonePI")
class VLMBackbone_PI(Qwen_PI):
    """Run an additional VLM through StarVLA's unchanged layerwise PI head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # Do not call Qwen_PI.__init__: it deliberately contains Qwen-specific
        # vision-tower discovery. Its forward/inference methods are inherited.
        baseframework.__init__(self)
        self.config = config
        self.qwen_vl_interface = get_comparison_vlm_model(config)

        visual = self.qwen_vl_interface.get_visual_module()
        if visual is None:
            raise AttributeError(
                f"{type(self.qwen_vl_interface).__name__} did not expose a visual module"
            )
        visual.requires_grad_(False)

        self.tap_indices = normalized_depth_indices(
            self.qwen_vl_interface.num_layers, 8
        )
        self.backbone_conditioner = PiBackboneConditioner(
            self.qwen_vl_interface.hidden_size
        )

        # Byte-for-byte equivalent geometry to QwenPI/WanPI.  Only the
        # perception backbone and its input adapter differ.
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
        action_cfg.repeated_diffusion_steps = 2
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = action_cfg.future_action_window_size
        self.past_action_window_size = action_cfg.past_action_window_size
        self.chunk_len = (
            self.past_action_window_size + 1 + self.future_action_window_size
        )


__all__ = ["VLMBackbone_PI"]
