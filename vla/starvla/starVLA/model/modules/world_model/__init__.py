"""World-model backbone interfaces used by WM4A frameworks."""


def get_world_model(config):
    wm_cfg = config.framework.get("world_model", {})
    model_name = str(wm_cfg.get("base_wm", ""))
    if "cosmos-predict2.5" in model_name.lower() or str(wm_cfg.get("type", "")).lower() == "cosmos":
        from .CosmosPredict25 import CosmosPredict25Interface

        return CosmosPredict25Interface(config)
    if "wan2" in model_name.lower() or "ti2v" in model_name.lower():
        from .Wan2 import Wan2Interface

        return Wan2Interface(config)
    raise NotImplementedError(f"World model {model_name} is not implemented")


__all__ = ["get_world_model"]
