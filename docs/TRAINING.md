# Training

GroudingPi uses ms-swift. Prepare the environment, model, and training caches as described in [Data Preparation](DATA_PREPARATION.md), then run from the project root:

```bash
python3 run.py setup train
python3 run.py train
```

Checkpoints are saved to `outputs/vlm_train/` by default.

## Configuration

- `configs/train/vlm.yaml`: model and dataset paths, learning rate, batch size, sequence length, and checkpoint settings.
- `configs/release/vlm_train.yaml`: training entrypoint, environment variables, and distributed topology.

Update both configurations for your resources. Model, cache, and output paths are relative to the project root. The launcher converts the training YAML into ms-swift arguments.

Preview the configuration without starting training:

```bash
.venv-train/bin/python scripts/run.py configs/release/vlm_train.yaml --dry-run
```

The model vocabulary, tokenizer, processor, cache manifest, and preprocessing settings must match. Configure `runtime.image_max_token_num` and `training.max_length` consistently with the prepared cache. Supported training fields are listed in `train/release_parameters.py`.

## Distributed training

Set `distributed.nnodes`, `nproc_per_node`, `node_rank`, `master_addr`, and `master_port` in the launch configuration. For multiple nodes, use a reachable coordinator address and a unique rank on each node, and run the command on every node.

The global batch size is `training.per_device_train_batch_size` multiplied by the world size and `training.gradient_accumulation_steps`. Keep `runtime.expected_nodes`, `runtime.expected_gpus_per_node`, and `runtime.expected_world_size` consistent with the launch configuration. The VLM training configuration does not accept `training.expected_global_batch_size`.

## Resume training

Set `training.resume_from_checkpoint` in `configs/train/vlm.yaml` to the checkpoint directory, then launch training normally. Alternatively, use `checkpoint.resume_from_checkpoint` in the launch configuration; specify the path in only one location.

A resumable checkpoint includes the optimizer, scheduler, and training state along with the model weights.
