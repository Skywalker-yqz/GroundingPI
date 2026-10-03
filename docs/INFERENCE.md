# Inference

Prepare the model weights, tokenizer, processor, and model code in `weights/vlm/`. Run commands from the project root.

## Start the service

Install the serving environment, then start the model:

```bash
python3 run.py setup serve
python3 run.py serve
```

For PPU, use `python3 run.py setup serve --platform ppu` during installation. The service command is the same on both platforms. The setup installs the bundled Transformers 5.7.0 fork alongside the platform vLLM 0.18 runtime. The PPU image's system Transformers 4.57.0 loads the checkpoint but produces incorrect grounding output; `infer/serve_vllm.py` rejects that environment before starting the service.

The default endpoint is `http://127.0.0.1:8000/v1`, with model ID `groundingpi`.

To customize the model path, port, or serving parameters, edit the matching configuration in `configs/release/` and pass it with `--config`:

```bash
python3 run.py serve --config configs/release/vlm_vllm_gpu.yaml
```

Use `--venv` to select another installed environment. When switching checkpoints, select a new `--overlay` directory in the service configuration.

## Predict an image

With the service running, open another terminal:

```python
from grounding_pi import GroundingPi

client = GroundingPi()
result = client.predict("your_image.jpg", "the red car", task="bbox")
print(result.to_dict())
```

Use `task="point"` for point localization. See [examples](../examples/README.md) for visualization and command-line usage.

Custom API requests should preserve spatial tokens with `skip_special_tokens=false`. Set the evaluation recipe's `api_url` and `model_id` to match the service, and use `service_contract: openai`. See [Evaluation](EVALUATION.md).

## Native service

The native Transformers service uses the training environment:

```bash
.venv-train/bin/python scripts/run.py configs/release/vlm_serve.yaml
```

Configure its model path, address, and generation settings in `configs/release/vlm_serve.yaml`.
