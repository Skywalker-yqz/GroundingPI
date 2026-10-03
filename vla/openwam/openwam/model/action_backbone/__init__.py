"""Action backbone package: the action-stream ABCs + the Fixed-16 π-style head.

    from openwam.model.action_backbone import Fixed16PiActionDiT, BackboneConditioner

Unlike video/vlm there is no registry — the architecture constructs its action
backbone directly — so this file only re-exports the public classes.

``ActionDiT`` / ``SharedMoEActionBackbone`` / ``SharedVanillaActionBackbone`` were
removed with the joint-denoising architectures; recover them from git history if
a baseline ever needs them.
"""

from openwam.model.action_backbone.backbone_conditioner import BackboneConditioner
from openwam.model.action_backbone.base import ActionDiTBackbone
from openwam.model.action_backbone.fixed16_pi_action_dit import Fixed16PiActionDiT

__all__ = [
    "ActionDiTBackbone",
    "BackboneConditioner",
    "Fixed16PiActionDiT",
]
