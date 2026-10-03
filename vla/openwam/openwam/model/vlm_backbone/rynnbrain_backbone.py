"""RynnBrain-2B through the OpenWAM VLM contract.

The released RynnBrain-2B checkpoint is a Qwen3-VL-2B-Instruct derivative and
declares ``Qwen3VLForConditionalGeneration``. Keep a named wrapper so configs,
logs, and result directories say ``rynnbrain`` while reusing the exact loader and
feature-extraction path already exercised by Qwen3-VL.
"""

from openwam.model.vlm_backbone.qwen3_vl_backbone import Qwen3VLBackbone


class RynnBrainBackbone(Qwen3VLBackbone):
    """Named Qwen3-VL adapter for Alibaba DAMO Academy's RynnBrain-2B.

    Geometry comes from the checkpoint (28 blocks, hidden size 2048), producing
    normalized-depth taps ``[0, 4, 8, 12, 15, 19, 23, 27]``.
    """


__all__ = ["RynnBrainBackbone"]
