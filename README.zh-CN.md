# GroudingPi

[English](README.md) | 简体中文

## 快速开始

在仓库根目录执行命令。`requirements.txt` 安装客户端和安装工具所需的依赖，训练、服务与评测的依赖分别放在 `requirements/` 下。如需通过 `environment.yml` 创建 Conda 环境，参见[环境说明](environments/README.md)。

客户端连接已经启动的模型服务。如果需要自行部署模型，请先完成下方的[推理](#推理)步骤。客户端本身不加载权重，也不依赖 Torch。

```bash
python -m pip install -r requirements.txt
```

```python
from grounding_pi import GroundingPi, visualize

client = GroundingPi()
result = client.predict("your_image.jpg", "the red car", task="bbox")
print(result.to_dict())
if result.valid:
    visualize("your_image.jpg", result).save("prediction.png")
```

点定位使用 `task="point"`。返回结果包含原始输出、停止原因和词元用量；格式错误或被截断的响应会返回 `valid=False`。坐标采用 0–999 网格，可视化时映射到图片像素。

图片准备、JSON 输出和可视化命令见[示例说明](examples/README.md)。

## 训练

以下命令均在项目根目录执行。`run.py` 自动选择环境和默认配置；每个环境的安装命令只需执行一次。依赖与安装要求见[环境说明](environments/README.md)。

默认训练环境需要匹配的 PPU 厂商镜像。GPU 推理服务使用下文的独立安装方式。

将完整模型、分词器、处理器和匹配的模型代码放入 `weights/vlm/`。按照 `configs/train/vlm.yaml`，在 `data/spatial_train/cache/` 下准备训练缓存及清单。权重和数据需单独准备；项目不提供通用的 JSONL 到训练缓存转换器。

根据实际训练任务修改 `configs/train/vlm.yaml`，然后执行：

```bash
python3 run.py setup train
python3 run.py train
```

检查点保存到 `outputs/vlm_train/`。配置、数据检查和断点恢复方法见[训练说明](docs/TRAINING.md)。

部署新的检查点时，需要在 `weights/vlm/` 中准备完整模型及匹配的分词器、处理器和模型代码。切换模型时使用新的模型代码覆盖（overlay）输出目录，具体见[推理说明](docs/INFERENCE.md)。

## 推理

推理服务使用 vLLM。**只需在安装环境时选择一次平台**：

```bash
# GPU
python3 run.py setup serve

# PPU：在匹配的平台镜像中，改用以下命令
python3 run.py setup serve --platform ppu
```

两种平台使用相同的启动命令：

```bash
python3 run.py serve
```

启动器读取安装时保存的平台信息，选择匹配的 vLLM 配置。启动服务时无需再次指定平台。

默认地址为 `http://127.0.0.1:8000/v1`，模型 ID 为 `groundingpi`。服务就绪后，在另一个终端运行客户端示例。自定义配置及原生参考服务见[推理说明](docs/INFERENCE.md)。

## 评测

先启动模型服务；评测只连接已有服务。默认 [`configs/eval/gam.yaml`](configs/eval/gam.yaml) 是 8 条样本的 smoke 测试：

```yaml
mode: GAM
model_path: weights/vlm
api_url: http://127.0.0.1:8000/v1
data_root: data/eval
datasets: configs/datasets.yaml
tasks: [gam_refcocog_val]
run_id: gam_smoke_001
limit: 8
```

模型放在项目内的 `weights/vlm/`。将 `data_root` 设为数据根目录，在 `datasets` 中修改各任务输入路径；路径可相对 `data_root`，也可用绝对路径。`api_url` 和 `model_id` 要对应已启动的服务。任务 ID 见 `configs/eval/tasks.json`，数据映射说明见[评测文档](docs/EVALUATION.md)。

```bash
python3 run.py setup eval
python3 run.py eval --config configs/eval/gam.yaml
```

要运行**全部 GAM 任务**，将默认配置复制为本地文件，例如 `configs/eval/gam_job_full_local.yaml`。修改数据与服务路径，在 `tasks` 中列出 `configs/eval/tasks.json` 的全部 42 个任务 ID，设置 `limit: null`，并使用新的 `run_id`。然后执行：

```bash
cp configs/eval/gam.yaml configs/eval/gam_job_full_local.yaml
# 按上述说明修改本地 YAML。
.venv-eval/bin/python scripts/evaluate.py configs/eval/gam_job_full_local.yaml --dry-run
python3 run.py eval --config configs/eval/gam_job_full_local.yaml
```

确认 dry run 没有缺失输入。结果写入 `outputs/eval/<run_id>/`，全部任务结束后生成 `summary.json`。含内部路径的本地配置不要放入源码发布包。

## 配置与文档

使用 `python3 run.py --help` 查看命令。`--config` 指定自定义启动或评测 YAML，`--venv` 指定其他环境；添加 `--dry-run` 可预览实际执行命令。Pi 默认服务的预览需要已安装的环境及平台记录；显式指定 `--config` 时不需要该记录。实际执行时，底层脚本会检查完整配置和所需资源。

| 路径 | 用途 | 说明 |
|---|---|---|
| `run.py` | 安装、训练、推理服务与评测入口 | `python3 run.py --help` |
| `models/` | 训练与推理共用的模型定义 | [推理说明](docs/INFERENCE.md) |
| `train/` | 数据检查、分词器工具、训练与检查点处理 | [训练说明](docs/TRAINING.md) |
| `infer/` | 模型加载与 HTTP 服务 | [推理说明](docs/INFERENCE.md) |
| `eval/` | 任务、提示词、请求与指标计算 | [评测说明](docs/EVALUATION.md) |
| `configs/` | 启动、训练、评测和数据配置 | [数据准备](docs/DATA_PREPARATION.md) |

完整流程通过源码目录中的 `run.py` 执行；也可直接使用底层 `scripts/` 和 YAML 入口。wheel 安装包只包含独立的 `grounding_pi` 客户端。本项目可独立使用，训练和推理服务使用各自的独立环境。

## 当前状态与许可证

支持的使用流程见上文。模型权重和数据集需单独获取。

项目原创贡献采用 [Apache License 2.0](LICENSE)，项目方不额外附加限制；第三方代码、衍生模型代码及权重仍遵守各自适用的上游许可证。第三方声明、原始许可证及源码来源保留在 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)、`models/vlm/LICENSE` 和 `third_party/` 中。
