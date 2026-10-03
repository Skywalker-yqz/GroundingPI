"""DualSystem architecture family.

- :class:`DualSystemFixed16PiArchitecture` — Fixed-16 π-style layerwise plan: the
  video DiT is a pinned feature extractor and a fixed 16-block Action Expert
  reads 8 normalized-depth taps. Representation comparison, no video loss.

Only the Fixed-16 variant ships here, because this codebase exists to run the backbone
comparison. Recover them from git history if a joint-denoising baseline is ever
needed again.
"""

from openwam.model.architectures.dual_system.fixed16_pi import DualSystemFixed16PiArchitecture

__all__ = ["DualSystemFixed16PiArchitecture"]
