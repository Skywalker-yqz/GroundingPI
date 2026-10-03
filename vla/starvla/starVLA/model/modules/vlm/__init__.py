import json
import os


def _read_model_type(vlm_name):
    """Read `model_type` from a local checkpoint's config.json. Returns None if unreadable."""
    try:
        with open(os.path.join(vlm_name, "config.json")) as f:
            return json.load(f).get("model_type")
    except (OSError, ValueError, TypeError):
        return None


def get_vlm_model(config):
    """Build the Qwen3-VL interface used by QwenPI.

    Rex-Omni, PaliGemma, LocateAnything and RynnBrain are loaded through
    ``starVLA.model.modules.vlm.comparison_backbones`` (framework VLMBackbonePI).
    """
    vlm_name = str(config.framework.qwenvl.base_vlm)
    explicit_type = str(config.framework.qwenvl.get("vlm_type", "") or "").lower().replace("-", "").replace("_", "")
    model_type = _read_model_type(vlm_name)
    if explicit_type in {"qwen3vl", "qwen3"} or model_type == "qwen3_vl" or (not explicit_type and "Qwen3-VL" in vlm_name):
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    raise NotImplementedError(
        f"VLM model {vlm_name} (model_type={model_type!r}) is not supported by QwenPI. "
        "Use framework.name=VLMBackbonePI for the other comparison backbones."
    )
