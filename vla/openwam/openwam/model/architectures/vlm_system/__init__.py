"""VlmSystem architecture family.

A vision-language backbone as the sole perception stream — no video DiT, no VAE,
no video loss. Exists so a VLM representation can be compared against a
video-generation representation through a byte-identical Action Expert.

- :class:`VlmSystemFixed16PiArchitecture` — Fixed-16 π-style layerwise plan, the
  VLM sibling of ``dual_system_fixed16_pi``.
"""

from openwam.model.architectures.vlm_system.fixed16_pi import VlmSystemFixed16PiArchitecture

__all__ = ["VlmSystemFixed16PiArchitecture"]
