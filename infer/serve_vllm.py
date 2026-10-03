"""Serve GroundingPI through the platform vLLM engine."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(model, overlay):
    from infer.overlay.prepare_model_overlay import create_overlay
    from models import vlm_compat
    for path in (model, overlay):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("model and overlay must remain inside the project")
    if not overlay.exists():
        create_overlay(model, overlay)
    manifest = json.loads((overlay / "vllm_overlay_manifest.json").read_text())
    if manifest["source_model"] != str(model.relative_to(ROOT)):
        raise ValueError("overlay belongs to a different checkpoint")
    transforms = {
        "configuration_groundingpi.py": vlm_compat._patched_config_source,
        "modeling_groundingpi_vision.py": vlm_compat._patched_vision_model_source,
        "modeling_groundingpi.py": vlm_compat._patched_model_source,
        "processing_groundingpi.py": vlm_compat._patched_processor_source,
    }
    for name, transform in transforms.items():
        if (overlay / name).read_text() != transform((model / name).read_text()):
            raise ValueError(f"stale or modified serving adapter: {name}; use a new overlay")
    for name in manifest["weight_files"]:
        if (overlay / name).resolve() != (model / name).resolve():
            raise ValueError(f"overlay weight does not reference the selected checkpoint: {name}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--overlay", default="outputs/vllm-model")
    p.add_argument("--platform", choices=("ppu", "gpu"), required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name", default="groundingpi")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    a = p.parse_args()
    model, overlay = [(ROOT / x).resolve() for x in (a.model, a.overlay)]
    import vllm
    import torch
    import transformers
    transformers_source = Path(transformers.__file__).resolve()
    expected_source = (ROOT / "vendor/transformers/src").resolve()
    if transformers.__version__ != "5.7.0" or not transformers_source.is_relative_to(expected_source):
        raise RuntimeError(
            "GroundingPI vLLM requires the bundled Transformers 5.7.0 fork; "
            "run python3 run.py setup serve --platform " + a.platform +
            f" (loaded {transformers.__version__} from {transformers_source})"
        )
    manifest = prepare(model, overlay)
    version = importlib.metadata.version("vllm")
    if not version.startswith("0.18.") or ("ppu" in version) != (a.platform == "ppu"):
        raise RuntimeError(f"incompatible vLLM {version} for {a.platform}; use its platform serve environment")
    if not torch.cuda.is_available():
        raise RuntimeError("no accelerator available")
    device = torch.cuda.get_device_name(0)
    if ("PPU" in device.upper()) != (a.platform == "ppu"):
        raise RuntimeError(f"device {device} does not match {a.platform}")
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(overlay), "--served-model-name", a.served_model_name,
               "--host", a.host, "--port", str(a.port), "--trust-remote-code",
               "--model-impl", "transformers", "--dtype", "bfloat16", "--enforce-eager",
               "--tensor-parallel-size", str(a.tensor_parallel_size),
               "--max-model-len", str(a.max_model_len), "--max-num-seqs", str(a.max_num_seqs),
               "--gpu-memory-utilization", str(a.gpu_memory_utilization),
               "--limit-mm-per-prompt", '{"image":1,"video":0}']
    evidence = dict(engine="vllm", engine_version=version, engine_file=vllm.__file__,
                    model_impl="transformers", adapter="models/vlm_compat.py",
                    adapter_sha256=hashlib.sha256((ROOT / "models/vlm_compat.py").read_bytes()).hexdigest(),
                    device=device, torch=torch.__version__,
                    transformers=transformers.__version__,
                    transformers_file=str(transformers_source),
                    checkpoint=manifest["source_model"], command=command)
    (overlay / "engine_runtime.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
