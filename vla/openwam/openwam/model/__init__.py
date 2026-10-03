# Import architecture packages to trigger @register_architecture decorators
from openwam.model import architectures  # noqa: F401
from openwam.model.architectures import (
    ARCHITECTURE_METADATA,
    ARCHITECTURE_REGISTRY,
    ARCHITECTURE_SUPPORT,
    BaseWAMArchitecture,
    CanonicalArchitectureSpec,
    build_architecture,
    get_architecture_support,
    list_supported_architectures,
    normalize_architecture_spec,
    resolve_architecture_config,
)

__all__ = [
    "BaseWAMArchitecture",
    "ARCHITECTURE_METADATA",
    "ARCHITECTURE_REGISTRY",
    "ARCHITECTURE_SUPPORT",
    "build_architecture",
    "get_architecture_support",
    "list_supported_architectures",
    "normalize_architecture_spec",
    "resolve_architecture_config",
    "CanonicalArchitectureSpec",
]
