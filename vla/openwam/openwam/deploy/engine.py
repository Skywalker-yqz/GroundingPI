"""Deploy-side inference engine: the deployment adapter around ``architecture.generate``.

``BaseInferenceEngine`` declares the engine interface; ``JointInferenceEngine``
is the production implementation. The engine translates deploy config +
per-request conditions into ``architecture.generate(...)`` arguments. The
Fixed-16 architectures own their sampling loop: one backbone forward per
observation, then ``num_inference_steps`` Euler steps of the Action Expert.
"""

import inspect
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from openwam.model.architectures.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)


class BaseInferenceEngine(ABC):
    """Inference engine base class.

    Args:
        cfg: Hydra config.
        architecture: WAM architecture wrapping the action backbone.
        action_backbone: Optional action backbone reference for implementations that still expose one.
    """

    require_architecture = False

    def __init__(self, cfg, architecture: Optional[BaseWAMArchitecture] = None, action_backbone=None):
        if self.require_architecture and architecture is None:
            raise ValueError("architecture is required")
        self.cfg = cfg
        self.architecture = architecture
        self.action_backbone = action_backbone

    @abstractmethod
    def generate(self, conditions: dict) -> dict:
        """Generate actions from observation conditions."""
        ...


class JointInferenceEngine(BaseInferenceEngine):
    """Production engine: normalizes the request, calls ``architecture.generate``.

    Args:
        cfg: Hydra config (must contain ``cfg.inference``).
        architecture: WAM architecture wrapping the action backbone.
        action_backbone: Optional action backbone reference retained for base-class storage.
    """

    require_architecture = True

    def __init__(
        self,
        cfg,
        architecture: Optional[BaseWAMArchitecture] = None,
        action_backbone=None,
    ):
        super().__init__(cfg, architecture=architecture, action_backbone=action_backbone)
        self._architecture_generate_accepts_extra_kwargs: Optional[bool] = None
        self._architecture_generate_kwarg_names: Optional[set[str]] = None
        self._architecture_generate_warned_dropped_kwargs: set[tuple[str, tuple[str, ...]]] = set()
        optimization = getattr(cfg, "optimization", None)
        self._decode_video = bool(getattr(optimization, "decode_video", False)) if optimization else False
        self._profile = os.environ.get("WAM_PROFILE", "0") == "1"

    def _filter_architecture_generate_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop deploy-only kwargs the architecture's generate() cannot consume."""

        if getattr(self, "_architecture_generate_kwarg_names", None) is None:
            params = inspect.signature(self.architecture.generate).parameters
            self._architecture_generate_accepts_extra_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
            )
            self._architecture_generate_kwarg_names = {
                name
                for name, param in params.items()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
        if self._architecture_generate_accepts_extra_kwargs:
            return kwargs
        accepted = self._architecture_generate_kwarg_names
        dropped = {name: value for name, value in kwargs.items() if name not in accepted}
        meaningful_dropped = tuple(
            sorted(name for name, value in dropped.items() if not self._is_noop_dropped_generate_kwarg(name, value))
        )
        if meaningful_dropped:
            warned = getattr(self, "_architecture_generate_warned_dropped_kwargs", set())
            architecture_name = type(self.architecture).__name__
            warning_key = (architecture_name, meaningful_dropped)
            if warning_key not in warned:
                warned.add(warning_key)
                self._architecture_generate_warned_dropped_kwargs = warned
                logger.warning(
                    "%s.generate does not accept deploy kwarg(s) %s; dropping them for this request.",
                    architecture_name,
                    ", ".join(meaningful_dropped),
                )
        return {name: value for name, value in kwargs.items() if name in accepted}

    @staticmethod
    def _is_noop_dropped_generate_kwarg(name: str, value: Any) -> bool:
        """Return whether dropping an unsupported deploy kwarg preserves default behavior."""

        if name == "decode_video":
            return value in (True, False)
        if name == "profile":
            return value is False
        return value is None

    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        """Generate actions from conditions.

        Args:
            conditions: dict with keys:
                - prompt (str): text prompt
                - first_frame_image (list[PIL.Image]): the observation frame(s)
                - num_frames (int, optional): raw state/action window length;
                  generated action chunk length is ``num_frames - 1``
                - video_num_frames (int, optional): observation clip length after any
                  training-time video_stride sub-sampling; defaults from cfg, then
                  falls back to ``num_frames``
                - height / width (int, optional): defaults from cfg
                - seed (int, optional): random seed, default 42
                - denoise_steps (int, optional): override the number of action denoising steps
                - proprio / observation.state: raw proprioception, normalized here

        Returns:
            dict with ``video`` (always ``None`` for this family) and ``actions`` (numpy array).
        """
        inf_cfg = self.cfg.inference
        denoise_steps = conditions.get("denoise_steps", inf_cfg.denoise_steps)

        # Deploy proprio: array-like in, normalized model-space tensor out.
        proprio = conditions.get("proprio")
        if proprio is None:
            observation = conditions.get("observation") or {}
            proprio = observation.get("state") if isinstance(observation, dict) else None
        if proprio is not None:
            proprio = self.architecture.normalize_deploy_proprio(proprio)

        action_num_frames = int(conditions.get("num_frames", getattr(inf_cfg, "num_frames", 49)))
        video_num_frames = int(
            conditions.get(
                "video_num_frames",
                getattr(inf_cfg, "video_num_frames", action_num_frames),
            )
        )

        generate_kwargs = self._filter_architecture_generate_kwargs(
            {
                "prompt": conditions.get("prompt", ""),
                "first_frame_image": conditions.get("first_frame_image", None),
                "num_frames": video_num_frames,
                "action_num_frames": action_num_frames,
                "height": conditions.get("height", getattr(inf_cfg, "height", 384)),
                "width": conditions.get("width", getattr(inf_cfg, "width", 320)),
                "seed": conditions.get("seed", 42),
                "tiled": conditions.get("tiled", True),
                "input_video_latents": conditions.get("input_video_latents", None),
                "num_inference_steps": denoise_steps,
                "decode_video": self._decode_video,
                "profile": self._profile,
                "proprio": proprio,
            }
        )
        return self.architecture.generate(**generate_kwargs)
