# Data Preparation

## Resources and format

Run commands from the project root. Keep models in `weights/`, data and caches in `data/`, and runtime results in `outputs/`. Training resources and outputs must resolve inside the project; evaluation inputs can be mapped to external read-only datasets through `configs/datasets.yaml`. Model weights and datasets are supplied separately.

GAM JSONL uses `id/messages/images/source`; each `<image>` placeholder corresponds to an actual image. Spatial output preserves object_ref/box markers, `<0>` through `<999>`, and `</c>`. A bounding box has four coordinates and a point has two. Training requires Arrow caches prepared with the matching tokenizer/processor and a `cache_manifest.json`. A generic JSONL-to-training-cache converter is not included.

Rebind and validate image paths when moving caches. A full resume checkpoint includes optimizer, scheduler, and trainer state as well as model weights.

## Model validation

For a base model whose spatial vocabulary has already been expanded, generate the training manifest with:

```bash
.venv-train/bin/python -m train.tokenizer.validate_tokens weights/vlm --write-manifest weights/vlm/gam_tokenizer_manifest.json
```

VLM 使用带 Qwen3 文本骨干的 checkpoint。缓存清单的 `source_id`、tokenizer 摘要和预处理参数需与训练配置一致；`repeat` 与 `sample_count` 分别表示重复次数和抽样行数，不能同时设置 `repeat > 1` 与抽样。

## 已有合格缓存 → VLM YAML

需要 `data/spatial_train/cache/train`（Arrow）、`data/spatial_train/cache/cache_manifest.json` 和 `weights/vlm/gam_tokenizer_manifest.json`。修改 `configs/train/vlm.yaml`：`datasets[].path/manifest/source_id` 对应同一来源，`model.path/manifest` 对应实际模型；`runtime.image_max_token_num=1024`、`training.max_length=2048` 必须与实际预处理一致。不能把 raw JSONL 填到 Arrow 路径。

```bash
.venv-train/bin/python scripts/run.py configs/release/vlm_train.yaml --dry-run
.venv-train/bin/python scripts/run.py configs/release/vlm_train.yaml
```

首条只展开命令；第二条才执行资源准入与 ms-swift。
