#!/usr/bin/env python3
"""Create a lightweight vLLM-serving view of the GroundingPI checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHITECTURE = "GroundingPIForConditionalGeneration"
CONFIG_MODULE = "configuration_groundingpi.py"
VISION_MODEL_MODULE = "modeling_groundingpi_vision.py"
MODEL_MODULE = "modeling_groundingpi.py"
PROCESSOR_MODULE = "processing_groundingpi.py"
SERVING_SUFFIXES = {
    ".jinja",
    ".json",
    ".model",
    ".py",
    ".safetensors",
    ".tiktoken",
    ".txt",
}
EXCLUDED_FILES = {"args.json", "trainer_state.json", "zero_to_fp32.py"}


# Allow both direct CLI execution and package import from the source tree.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.vlm_compat import (
    _patched_config_source, _patched_vision_model_source,
    _patched_model_source, _patched_processor_source,
)


def create_overlay(model_dir: Path, output_dir: Path) -> None:
    import tempfile
    import shutil
    model_dir = model_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    root = PROJECT_ROOT.resolve()
    if not model_dir.is_relative_to(root) or not output_dir.is_relative_to(root):
        raise ValueError("model and overlay must be inside the project root")
    if output_dir.is_relative_to(model_dir) or model_dir.is_relative_to(output_dir):
        raise ValueError("model and overlay directories must not contain one another")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    config = json.loads((model_dir / "config.json").read_text())
    if (config.get("architectures") != [ARCHITECTURE]
            or config.get("model_type") != "groundingpi"
            or config.get("text_config", {}).get("model_type") != "qwen3"):
        raise ValueError("overlay requires GroundingPI with plain Qwen3")
    required = {"config.json", CONFIG_MODULE, VISION_MODEL_MODULE, MODEL_MODULE,
                PROCESSOR_MODULE, "image_processing_groundingpi.py", "media_utils.py",
                "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"}
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("weight_map must be nonempty")
        shards = set(mapping.values())
        if any(not isinstance(s, str) or Path(s).name != s or not s.endswith('.safetensors') for s in shards):
            raise ValueError("weight shards must be safetensors filenames inside the model directory")
        required |= shards | {"model.safetensors.index.json"}
    else:
        required.add("model.safetensors")
    missing = sorted(name for name in required if not (model_dir / name).is_file())
    if missing: raise FileNotFoundError(f"source model is incomplete: {missing}")
    sources = [s for s in sorted(model_dir.iterdir()) if s.is_file()
               and s.name not in EXCLUDED_FILES and s.suffix in SERVING_SUFFIXES]
    if any(not s.resolve().is_relative_to(root) for s in sources):
        raise ValueError("source file symlink escapes project")
    patchers = {CONFIG_MODULE: _patched_config_source, VISION_MODEL_MODULE: _patched_vision_model_source,
                MODEL_MODULE: _patched_model_source, PROCESSOR_MODULE: _patched_processor_source}
    # Validate every patch before creating any destination files.
    patched = {name: patcher((model_dir / name).read_text()) for name, patcher in patchers.items()}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".overlay-", dir=output_dir.parent))
    try:
        for source in sources:
            destination = temporary / source.name
            if source.name in patched:
                destination.write_text(patched[source.name])
            else:
                os.symlink(os.path.relpath(source, output_dir), destination)
        manifest = {"schema_version": 2, "architecture": ARCHITECTURE,
                    "source_model": str(model_dir.relative_to(root)),
                    "path_base": "project_root", "weight_files": sorted(n for n in required if n.endswith('.safetensors')),
                    "compatibility_patches": ["hf_config_4_5", "processor_mixin_multimodal_tokens",
                                              "vision_sdpa", "rmsnorm_epsilon", "vllm_transformers_attention_backend"]}
        (temporary / "vllm_overlay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    print(output_dir.relative_to(root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="source checkpoint directory")
    parser.add_argument("--output", required=True, type=Path, help="new serving overlay directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_overlay(args.model, args.output)
