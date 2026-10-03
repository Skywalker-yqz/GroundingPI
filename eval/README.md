<a id="evaluation"></a>

# 🧪 GroundingPI Evaluation

This is the canonical guide for the **34-benchmark suite** used in the GroundingPI paper. Run all commands from the repository root.

[Setup](#setup) · [Modes](#modes) · [Full suite](#full-suite) · [Check and run](#run) · [Outputs](#outputs)

<a id="setup"></a>

## 🛠️ Prepare the environment and model service

Use **Linux x86_64 and Python 3.12**. Install the setup dependencies and the separate evaluation environment once:

```bash
python3 -m pip install -r requirements.txt
python3 run.py setup eval
```

Setup installs the bundled evaluation engine and its dependencies. Serving and evaluation use separate environments. Reuse an existing evaluation environment; the installer does not overwrite it. See [Environment Setup](../environments/README.md).

Download the model and prepare serving through the [repository Quick Start](../README.md#quick-start). Start the matching service in another terminal and keep it running throughout evaluation.

```bash
python3 run.py serve
```

| Checkpoint | Evaluation template | Model path | API base URL | API model ID |
|:---|:---|:---|:---|:---|
| GroundingPI | `configs/eval/gam.yaml` | `weights/vlm` | `http://127.0.0.1:8000/v1` | `groundingpi` |

The documented serving setup selects the appropriate platform runtime. Evaluation connects to the existing service; it does not start it.

<a id="data"></a>

## 📦 Evaluation data

[Grounding-EvalData](https://huggingface.co/datasets/Skywalker0410/Grounding-EvalData)

<a id="modes"></a>

## 🧭 Evaluation modes

The evaluator supports **7 mode identifiers**. A mode selects prompts and output parsing independently of the serving engine.

| Mode | Supported models |
|:---|:---|
| **GAM** | **GroundingPI, GroundAnything, GroundAnything-VLM** |
| VLM | Generic vision-language baselines |
| REXOMNI | Rex-Omni |
| LOCATEANYTHING | LocateAnything |
| GROUNDINGDINO | GroundingDINO through a compatible service |
| DLM | Legacy diffusion-checkpoint alias for the GAM protocol |
| RLV2 | Legacy RL-checkpoint alias for the GAM protocol |

**All three released checkpoints use `mode: GAM`.** The `-VLM` suffix does not select the generic `VLM` evaluation mode. GAM uses native spatial tokens and a fixed 0–999 grid; omit `coordinate_mode` for these checkpoints. External baseline weights and services are supplied separately.

<a id="full-suite"></a>

## 📋 Configure the complete 34-benchmark suite

The registry contains **42 task recipes**. These **34 entries** match this paper's suite; selecting every registry entry also includes additional evaluation variants.

| Task group | Entries | Benchmark entries |
|:---|---:|:---|
| Detection | 4 | COCO, LVIS, Dense200, VisDrone |
| Referring boxes | 6 | RefCOCOg val/test; RefCOCO, RefCOCOg, RefCOCO+ family entries; HumanRef |
| Object pointing | 7 | RefCOCOg val/test, COCO, LVIS, Dense200, VisDrone, HumanRef |
| Spatial pointing | 4 | RefSpatial Location/Placement/Unseen; RoboSpatial Context |
| GUI grounding | 3 | ScreenSpot-Pro, ScreenSpot-V2, OSWorld-G |
| OCR | 4 | HierText, ICDAR2015, TotalText, SROIE |
| Document layout | 2 | DocLayNet, M6Doc |
| Visual prompting | 4 | FSC147, Dense200, COCO, LVIS |

The eight excluded recipes are the four `*_Labelless` and four `*_Boxonly` variants.

The RefCOCO family entry `gam_refcocog` is distinct from `gam_refcocog_val` and `gam_refcocog_test`. RefCOCO family and RefSpatial averages in paper tables summarize their component tasks; they are not additional tasks.

Run this block from the repository root. It checks the explicit task list and creates local full-suite configurations from the shipped templates, leaving the original templates unchanged.

```bash
python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
import yaml

tasks = [
    "gam_coco",
    "gam_lvis",
    "gam_dense200",
    "gam_visdrone",
    "gam_refcocog_val",
    "gam_refcocog_test",
    "gam_refcoco",
    "gam_refcocog",
    "gam_refcocoplus",
    "gam_rex_point_refcocog_val",
    "gam_rex_point_refcocog_test",
    "gam_rex_point_coco",
    "gam_rex_point_lvis",
    "gam_rex_point_dense200",
    "gam_rex_point_visdrone",
    "gam_refspatial_location",
    "gam_refspatial_placement",
    "gam_refspatial_unseen",
    "gam_robospatial_context",
    "gam_screenspot_pro",
    "gam_screenspot_v2",
    "gam_osworld_g",
    "gam_hiertext",
    "gam_icdar2015",
    "gam_totaltext",
    "gam_sroie",
    "gam_doclaynet",
    "gam_m6doc",
    "gam_fsc147",
    "gam_visual_dense200",
    "gam_humanref",
    "gam_rex_point_humanref",
    "gam_visual_coco",
    "gam_visual_lvis",
]
assert len(tasks) == 34 and len(set(tasks)) == 34
inventory = json.loads(Path("configs/eval/tasks.json").read_text())
assert set(tasks) <= set(inventory), "Unknown task ID"

recipes = [
    ("gam.yaml", "full_suite_gam.yaml", "groundingpi_full34"),
]
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
for source_name, output_name, label in recipes:
    source = Path("configs/eval") / source_name
    output = Path("configs/eval") / output_name
    config = yaml.safe_load(source.read_text())
    config.update(
        mode="GAM",
        tasks=tasks,
        limit=None,
        run_id=f"{label}_{stamp}",
    )
    with output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(output, "run_id=" + config["run_id"])
PY
```

The block refuses to overwrite existing generated configurations. For a later run, edit the configuration's `run_id` to a fresh value or choose a new output filename. Existing result directories are never reused.

Set `data_root` in each generated configuration and update `configs/datasets.yaml` for your local data locations. Verify `model_path`, `api_url`, and `model_id` against the running service. Keep model/output paths project-relative, retain `service_contract: openai`, and keep **`limit: null`** for the full suite. The original templates use `limit: 8` for a small smoke run.

Retain the task recipes' output-token budgets. An altered output budget is a different evaluation setting.

<a id="run"></a>

## ▶️ Check inputs and run

Validate the generated configuration and inspect `missing_inputs` before sending inference requests:

```bash
.venv-eval/bin/python scripts/evaluate.py \
  configs/eval/full_suite_gam.yaml --dry-run
```

This evaluator dry run checks configuration and reports missing local inputs without contacting the service. Resolve every missing input before running. In contrast, `python3 run.py eval --dry-run` only previews the downstream command.

With the GroundingPI service running:

```bash
python3 run.py eval --config configs/eval/full_suite_gam.yaml
```

Actual execution checks the service and required resources before creating the result directory. Keep the service running until evaluation finishes. For a smaller selection, use a separate configuration and run ID and change its `tasks` and sample `limit` explicitly.

<a id="outputs"></a>

## 📊 Inspect the results

Each run writes to `outputs/eval/<run_id>/`:

| Output | Contents |
|:---|:---|
| `run.json` | Selected configuration, launch plan, service checks, and sample scope |
| `responses.jsonl` | Raw predictions, token usage, and finish reasons |
| `log_eval/` and `cache/` | Task logs, engine outputs, and response caches |
| `summary.json` | Per-task metrics, source result files, completion status, and missing tasks |

Check `summary.json` for `status`, `returncode`, and `missing_tasks`. A full-suite run must complete every selected task. Inspect malformed or truncated predictions as well as metric values; a successful HTTP response alone does not establish a correct prediction. Keep the checkpoint revision, selected configuration, and raw results together when comparing runs.

The script reports **per-task results** and does not compute a paper-wide aggregate score. One invocation is **one complete run of the selected suite**; it does not reproduce the paper's ten-run statistical averaging by itself.

<a id="implementation"></a>

## 🗂️ Evaluation implementation

Task definitions live under `Grounding/`, `Dense/`, `Referring/`, `Pointing/`, `GUI/`, `OCR/`, `Layout/`, and `VisualPrompt/`. Shared parsers and data-location helpers are in `utils/`; detection metrics are in `metrics/`.

- [Task registry](../configs/eval/tasks.json)
- [Evaluation launcher](../scripts/evaluate.py)
- [Environment setup](../environments/README.md)
- [Return to the project README](../README.md)
