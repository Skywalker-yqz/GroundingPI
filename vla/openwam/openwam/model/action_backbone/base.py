"""Action-stream backbone ABC.

:class:`ActionDiTBackbone` is the root of the standalone action transformer the
Fixed-16 comparison attaches to every backbone. Concrete subclasses inherit
``nn.Module, ABC`` directly so the state_dict lives at ``action_backbone.<param>``
with no wrapping prefix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ActionDiTBackbone(nn.Module, ABC):
    """ABC for the standalone action transformer (ActionDiT)."""

    def __init__(self):
        super().__init__()

    @property
    def bridge_layers(self) -> tuple[int, ...]:
        """Backbone blocks whose hidden states the action stream reads. Default none."""
        return getattr(self, "_bridge_layers", ())

    def set_dtype_device(self, dtype, device) -> None:
        """Move action backbone params/buffers to (dtype, device)."""
        self.to(dtype=dtype, device=device)

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Default no-op: action weights are fully captured by the safetensors
        checkpoint, no external artifacts to copy. Part of the architecture's
        deploy-asset hook contract."""

    @property
    def uses_proprioception(self) -> bool:
        """Whether this backbone consumes a ``proprio`` input. Default False."""
        return False

    @property
    @abstractmethod
    def num_heads(self) -> int: ...

    @property
    @abstractmethod
    def head_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Standalone action prediction from the cached backbone conditions."""
        raise NotImplementedError


__all__ = ["ActionDiTBackbone"]
