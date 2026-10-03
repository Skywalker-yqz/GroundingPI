"""Architecture package exports and side-effect registration."""

from openwam.model.architectures import dual_system, vlm_system  # noqa: F401
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system import DualSystemFixed16PiArchitecture
from openwam.model.architectures.registry import (
    ARCHITECTURE_METADATA,
    ARCHITECTURE_REGISTRY,
    ARCHITECTURE_SUPPORT,
    CanonicalArchitectureSpec,
    build_architecture,
    get_architecture_support,
    list_supported_architectures,
    normalize_architecture_spec,
    register_architecture,
    resolve_architecture_config,
)
from openwam.model.architectures.vlm_system import VlmSystemFixed16PiArchitecture

__all__ = [
    "BaseWAMArchitecture",
    "ARCHITECTURE_METADATA",
    "ARCHITECTURE_REGISTRY",
    "ARCHITECTURE_SUPPORT",
    "CanonicalArchitectureSpec",
    "build_architecture",
    "get_architecture_support",
    "list_supported_architectures",
    "normalize_architecture_spec",
    "register_architecture",
    "resolve_architecture_config",
    "DualSystemFixed16PiArchitecture",
    "VlmSystemFixed16PiArchitecture",
]
