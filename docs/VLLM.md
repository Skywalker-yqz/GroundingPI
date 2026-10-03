# ⚡ vLLM Deployment

GroundingPI includes a vLLM serving launcher for NVIDIA GPUs and PPU. Start from a **Linux / Python 3.12** environment that already provides the accelerator-compatible **vLLM 0.18.x**, Torch, and kernel runtime. For PPU, use the matching PPU runtime image and its PPU build of vLLM. The setup command inherits these platform packages and installs the repository's bundled **Transformers 5.7.0 fork** into a separate serving environment.

Run the following from the GroundingPI repository root:

```bash
python3 -m pip install -r requirements.txt huggingface_hub
hf download GroundingPI/GroundingPI --local-dir weights/vlm
```

Choose the setup command for your accelerator:

```bash
# NVIDIA GPU
python3 run.py setup serve --platform gpu

# PPU: use this setup command instead, inside the matching PPU runtime image.
python3 run.py setup serve --platform ppu
```

Then start the service with the same command on either platform:

```bash
python3 run.py serve
```

The installer records the selected platform in `.venv-serve/runtime.json`; the launcher selects the corresponding configuration automatically. Setup expects a new environment directory. For a separate environment, pass the same `--venv` directory to both setup and launch.

| Setting | Default |
|:---|:---|
| OpenAI-compatible API | `http://127.0.0.1:8000/v1` |
| Served model ID | `groundingpi` |
| Precision / execution | BF16 / eager; CUDA Graph disabled |
| Context limit | 16,384 tokens, including image, prompt, and output tokens |
| Tensor parallel size / maximum sequences | 1 / 1 |
| Accelerator memory utilization | 0.7 |
| Multimodal input | One image per request; video input disabled |

To customize model paths, binding address, port, or serving options, edit the relevant configuration and select it explicitly:

```bash
# NVIDIA GPU
python3 run.py serve --config configs/release/vlm_vllm_gpu.yaml

# PPU
python3 run.py serve --config configs/release/vlm_vllm_ppu.yaml
```

The launcher uses vLLM's Transformers implementation and creates an adapter under `outputs/vllm-model`. It adapts the processor to `ProcessorMixin`, expands image-token placeholders, and uses the checkpoint's tokenizer, chat template, and custom model code. Weight shards are reused through the overlay. When changing checkpoints, select a new `--overlay` path in the service configuration.

In another terminal, send one image and a grounding prompt:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image = base64.b64encode(Path("example.jpg").read_bytes()).decode("ascii")
payload = {
    "model": "groundingpi",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
            {"type": "text", "text": "Locate the target referred to by the following description: the red car."},
        ],
    }],
    "temperature": 0,
    "max_tokens": 4096,
    "skip_special_tokens": False,
    "spaces_between_special_tokens": False,
}
request = Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=120) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
```

Preserve both special-token options: GroundingPI returns semantic labels and 0–999 coordinate tokens inside object-reference and box delimiters. Use `mode: GAM`, `model_id: groundingpi`, and `service_contract: openai` when evaluating this endpoint.

The launcher requires the bundled Transformers 5.7.0 source. In particular, the PPU image's system Transformers 4.57.0 can load the checkpoint but is incompatible with correct grounding output. Keep the complete checkpoint and use the provided setup and serving entrypoints; the setup does not install a platform accelerator runtime from scratch.

## 🔎 Check the running service

In another terminal, check `curl --fail http://127.0.0.1:8000/v1/models` before sending requests. The returned model list should contain `groundingpi`. Then send the single-image request above and inspect the returned GAM tokens; a model-list response alone does not validate grounding quality.

The installer records package versions in `.venv-serve/installed.freeze.txt` and platform/import checks in `.venv-serve/runtime.json`. The serving overlay records its exact vLLM command, checkpoint, adapter hash, device, and imported Transformers source in `outputs/vllm-model/engine_runtime.json`.

## ⚙️ Override serving settings

To use a different port or checkpoint, edit the `args` list in a copy of the matching release YAML and pass that project-relative file with `python3 run.py serve --config ...`. The launcher accepts `--model`, `--overlay`, `--platform`, `--host`, `--port`, `--served-model-name`, `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`, and `--tensor-parallel-size`. Keep model and overlay directories inside the repository and use a fresh overlay path when changing checkpoints. The supplied recipe uses TP=1; larger parallel settings need separate validation for the target hardware.

BF16, eager execution, the Transformers backend, and the one-image input limit are fixed by the provided launcher. It does not expose a CUDA Graph switch. Point evaluation to the matching API URL and model ID; use the [34-benchmark Evaluation Guide](../eval/README.md) for the full suite.

See the [vLLM Transformers-backend documentation](https://docs.vllm.ai/en/v0.18.0/models/supported_models/#transformers) and [OpenAI-compatible server reference](https://docs.vllm.ai/en/v0.18.0/serving/openai_compatible_server/) for the underlying engine interfaces. The project-specific dependency and checkpoint checks above still apply.

## 🗂️ Batch annotation

For image collections, use the resumable [JSONL batch example and guide](BATCH_INFERENCE.md). The client supports the OpenAI-compatible endpoint and preserves GAM coordinate tokens.

## 🐳 NVIDIA container starting point

If you do not already have a vLLM runtime, the [official vLLM container](https://docs.vllm.ai/en/v0.18.0/deployment/docker/) provides a starting point. This example uses a Linux x86_64 host with Docker, NVIDIA Container Toolkit, and a driver compatible with the image. It is a setup recipe, not an additional accelerator-validation result.

From your cloned repository on the host:

```bash
docker run --rm -it --gpus 'device=0' \
  --network host --shm-size 8g \
  -v "$PWD":/workspace/GroundingPI -w /workspace/GroundingPI \
  --entrypoint bash vllm/vllm-openai:v0.18.0
```

Inside the container, prepare a fresh serving environment:

```bash
python3 -m pip install -r requirements.txt huggingface_hub
hf download GroundingPI/GroundingPI --local-dir weights/vlm
python3 run.py setup serve --platform gpu
python3 run.py serve --config configs/release/vlm_vllm_gpu.yaml
```

The mounted checkout retains weights, environment files, and outputs. On restart with the same image and mount path, skip installation and run the serving command. The default API binds to loopback; Linux host networking lets host-side clients reach it at the URL above. PPU uses its vendor runtime instead of this NVIDIA image. Keep the model adapter and bundled Transformers fork; a stock `vllm serve` invocation alone does not install them.
