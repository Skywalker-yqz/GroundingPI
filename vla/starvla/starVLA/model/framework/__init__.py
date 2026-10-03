"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., QwenPI.py, WanPI.py) should register itself:
    from starVLA.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("QwenPI")
    def build_model_framework(config):
        return Qwen_PI(config=config)
"""

import pkgutil
import importlib
from starVLA.model.tools import FRAMEWORK_REGISTRY


try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# 自动导入所有 framework 子模块 import，触发注册
if pkg_path is not None:
    try:
        for _, module_name, _ in pkgutil.iter_modules(pkg_path):
            importlib.import_module(f"{__name__}.{module_name}")
    except Exception as e:
        print(f"Warning: Failed to auto-import framework submodules: {e}")
        
def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "QwenPI")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"): 
        cfg.framework.name = cfg.framework.framework_py # 兼容旧配置yaml

    if cfg.framework.name == "QwenPI":
        from starVLA.model.framework.QwenPI import Qwen_PI
        return Qwen_PI(cfg)
    elif cfg.framework.name == "WanPI":
        from starVLA.model.framework.WanPI import Wan_PI
        return Wan_PI(cfg)
    elif cfg.framework.name == "CosmosPI":
        from starVLA.model.framework.CosmosPI import Cosmos_PI
        return Cosmos_PI(cfg)
    elif cfg.framework.name == "VLMBackbonePI":
        from starVLA.model.framework.VLMBackbonePI import VLMBackbone_PI
        return VLMBackbone_PI(cfg)

    # auto detect from registry
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented.")
    
    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
