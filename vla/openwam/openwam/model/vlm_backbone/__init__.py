"""VLM backbone package: the VlmBackbone ABC + its Qwen3-VL implementation.

    from openwam.model.vlm_backbone import build_vlm_backbone
    backbone = build_vlm_backbone("qwen3_vl", checkpoint_path=..., dtype=...)

Adding a VLM backbone: subclass :class:`VlmBackbone`, then register it at the
bottom of this file via ``register_vlm_backbone("name")(YourClass)`` and set
``vlm_backbone.name: your_name`` in the model config yaml.
"""

from openwam.model.vlm_backbone.base import VlmBackbone
from openwam.model.vlm_backbone.registry import (
    _VLM_BACKBONE_REGISTRY,
    build_vlm_backbone,
    register_vlm_backbone,
)

__all__ = [
    "VlmBackbone",
    "HFVlmBackbone",
    "LocateAnythingBackbone",
    "PaliGemmaBackbone",
    "Qwen2_5VLBackbone",
    "Qwen3VLBackbone",
    "RynnBrainBackbone",
    "build_vlm_backbone",
    "register_vlm_backbone",
    "_VLM_BACKBONE_REGISTRY",
]

# Built-in registration (at the bottom so the implementations import only from
# base/hf_vlm_base/registry, with no circular dependency).
from openwam.model.vlm_backbone.hf_vlm_base import HFVlmBackbone  # noqa: E402
from openwam.model.vlm_backbone.locate_anything_backbone import LocateAnythingBackbone  # noqa: E402
from openwam.model.vlm_backbone.paligemma_backbone import PaliGemmaBackbone  # noqa: E402
from openwam.model.vlm_backbone.qwen2_5_vl_backbone import Qwen2_5VLBackbone  # noqa: E402
from openwam.model.vlm_backbone.qwen3_vl_backbone import Qwen3VLBackbone  # noqa: E402
from openwam.model.vlm_backbone.rynnbrain_backbone import RynnBrainBackbone  # noqa: E402

register_vlm_backbone("qwen3_vl")(Qwen3VLBackbone)
# Also covers same-architecture derivatives such as Rex-Omni-3B.
register_vlm_backbone("qwen2_5_vl")(Qwen2_5VLBackbone)
register_vlm_backbone("paligemma")(PaliGemmaBackbone)
register_vlm_backbone("locate_anything")(LocateAnythingBackbone)
# RynnBrain-2B is a Qwen3-VL derivative; the named subclass preserves model-family
# identity in configs/logs while sharing the standard Qwen3-VL implementation.
register_vlm_backbone("rynnbrain")(RynnBrainBackbone)
