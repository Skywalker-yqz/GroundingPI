"""Wan2.2-TI2V backbone with StarVLA's fixed layerwise pi action expert."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.projector.PiBackboneConditioner import (
    PiBackboneConditioner,
    normalized_depth_indices,
)
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("WanPI")
class Wan_PI(baseframework):
    """Extract eight normalized-depth Wan features for one shared pi head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
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
        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(self.config)
        self.action_horizon = int(action_cfg.future_action_window_size) + 1

        self._captured: dict[int, torch.Tensor] = {}
        self._hooks = [
            self.backbone.transformer.blocks[index].register_forward_hook(
                self._make_capture_hook(index)
            )
            for index in self.tap_indices
        ]

    def _make_capture_hook(self, index: int):
        def capture(_module, _inputs, output):
            self._captured[index] = output[0] if isinstance(output, tuple) else output

        return capture

    def _extract_conditions(self, images, instructions) -> list[torch.Tensor]:
        inputs = self.backbone.build_inputs(images, instructions)
        self._captured.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.backbone(**inputs)
            missing = [index for index in self.tap_indices if index not in self._captured]
            if missing:
                raise RuntimeError(f"Wan hooks did not capture blocks {missing}")
            taps = [self._captured[index] for index in self.tap_indices]
            return self.backbone_conditioner(taps)

    def forward(self, examples: List[dict] = None, **kwargs):
        conditions = self._extract_conditions(
            [example["image"] for example in examples],
            [example["lang"] for example in examples],
        )
        actions = torch.as_tensor(
            np.array([example["action"] for example in examples]),
            device=conditions[0].device,
            dtype=conditions[0].dtype,
        )[:, -self.action_horizon :]
        state = None
        if "state" in examples[0]:
            state = torch.as_tensor(
                np.array([example["state"] for example in examples]),
                device=conditions[0].device,
                dtype=conditions[0].dtype,
            )
        repeat = int(self.config.framework.action_model.get("repeated_diffusion_steps", 2))
        repeated_conditions = [condition.repeat(repeat, 1, 1) for condition in conditions]
        repeated_state = state.repeat(repeat, 1, 1) if state is not None else None
        loss = self.action_model(
            repeated_conditions,
            actions.repeat(repeat, 1, 1),
            repeated_state,
        )
        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images,
        instructions,
        state=None,
        **kwargs,
    ):
        conditions = self._extract_conditions(batch_images, instructions)
        if state is not None:
            state = torch.as_tensor(
                np.array(state),
                device=conditions[0].device,
                dtype=conditions[0].dtype,
            )
        actions = self.action_model.predict_action(conditions, state)
        return {"normalized_actions": actions.detach().cpu()}

    @torch.inference_mode()
    def get_action(self, data: Dict[str, Any]) -> Any:
        """Eval entry matching the native StarVLA pi-style contract."""
        from transformers.feature_extraction_utils import BatchFeature

        video_data = data["video"]
        if hasattr(video_data, "ndim") and video_data.ndim == 6:
            video_data = video_data.squeeze(1)

        batch_size = int(video_data.shape[0])
        target_size = tuple(self.config.datasets.vla_data.image_size)
        batch_images: List[List[Image.Image]] = []
        for i in range(batch_size):
            images = []
            for j in range(video_data[i].shape[0]):
                frame = video_data[i][j]
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().numpy()
                images.append(Image.fromarray(np.asarray(frame)).resize(target_size))
            batch_images.append(images)

        instructions = data.get("annotation.task_index")
        if instructions is None:
            instructions = data.get("annotation.human.coarse_action")
        if instructions is None:
            instructions = [""] * batch_size
        elif isinstance(instructions, np.ndarray):
            instructions = instructions.tolist()
        elif isinstance(instructions, torch.Tensor):
            instructions = instructions.detach().cpu().tolist()
        if isinstance(instructions, str):
            instructions = [instructions] * batch_size

        state = data.get("state")
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()

        outputs = self.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            state=state,
        )
        return BatchFeature(data=outputs)


__all__ = ["Wan_PI"]
