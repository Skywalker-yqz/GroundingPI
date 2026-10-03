# Upstream attribution: the named implementation credits and retained
# legacy remarks in this file come from public StarVLA source/history
# (https://github.com/starVLA/starVLA), including revision
# f18fbc22c317dd1810839cb621632ac45add93f1 where applicable. They identify
# upstream contributions, not the authors or affiliations of this submission.

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import (
    build_param_lr_groups,
    get_no_decay_param_names,
    sync_gradient_accumulation_steps,
)

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
gc.disable()

# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)

    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def get_resume_step_offset(cfg) -> int:
    if getattr(cfg.trainer, "is_resume", False) and getattr(cfg, "resume_from_checkpoint", None):
        return int(Path(cfg.resume_from_checkpoint).name.replace("step_", ""))
    return 0


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps + get_resume_step_offset(cfg),
        #scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )
    return optimizer, lr_scheduler

def setup_optimizer_and_scheduler_with_decay(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler (支持参数分组decay)"""
    
    lr_param_groups = build_param_lr_groups(model=model, cfg=cfg)
    no_decay_param_names = get_no_decay_param_names(model, ALL_LAYERNORM_LAYERS)
    param_name_map = {p: n for n, p in model.named_parameters()}
    base_weight_decay = cfg.trainer.optimizer.weight_decay
    
    final_optimizer_groups = []

    for lr_group in lr_param_groups:
        params_decay = []
        params_no_decay = []

        params_decay_name = []
        params_no_decay_name = []
        
        group_lr = lr_group['lr']
        group_name = lr_group['name']

        for p in lr_group['params']:
            if not p.requires_grad:
                continue
            
            param_name = param_name_map.get(p)
            
            if param_name is None:
                logger.warning(f"Parameter not found in model.named_parameters(). Applying default decay. Param ID: {id(p)}")
                params_decay.append(p)
                continue

            if param_name in no_decay_param_names:
                params_no_decay.append(p)
                params_no_decay_name.append(param_name)
            else:
                params_decay.append(p)
                params_decay_name.append(param_name)

        if dist.is_initialized() and dist.get_rank() == 0:
            print("params_no_decay", params_no_decay_name[0] if len(params_no_decay_name)>0 else params_no_decay_name)
            print("params_decay", params_decay_name[0] if len(params_decay_name)>0 else params_decay_name)

        if params_decay:
            final_optimizer_groups.append({
                "params": params_decay,
                "lr": group_lr,
                "name": f"{group_name}_decay",
                "weight_decay": base_weight_decay  # 应用配置的 decay
            })
        
        # 为 "no_decay" 参数创建子组
        if params_no_decay:
            final_optimizer_groups.append({
                "params": params_no_decay,
                "lr": group_lr,
                "name": f"{group_name}_no_decay",
                "weight_decay": 0.0  # 不应用 decay
            })

    optimizer = torch.optim.AdamW(
        final_optimizer_groups, 
        lr=cfg.trainer.learning_rate.base,  
        betas=tuple(cfg.trainer.optimizer.betas),
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(f"Using {len(final_optimizer_groups)} optimizer groups.")
        for i, group in enumerate(optimizer.param_groups):
            logger.info(
                f"Optimizer Group {i} ({group['name']}): "
                f"lr={group['lr']}, wd={group['weight_decay']}, "
                f"num_params={len(group['params'])}"
            )

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps + get_resume_step_offset(cfg),
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.resume_step_offset = get_resume_step_offset(cfg)
        self.total_batch_size = self._calculate_total_batch_size()
        self.consecutive_nonfinite_skips = 0
        self.vla_batches_consumed = 0

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            reference_embodiment_slot = (
                self.config.trainer.reference_embodiment_slot
                if hasattr(self.config.trainer, "reference_embodiment_slot")
                else None
            )
            self.model = self.load_pretrained_backbones(
                self.model,
                pretrained_checkpoint,
                reload_modules=reload_modules,
                reference_embodiment_slot=reference_embodiment_slot,
            )

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # 2. 打印可训练参数的数量和名称（用于调试）
        print(f"Found {len(trainable_params)} trainable parameters.")

        if rank == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print('requires_grad:', name)

        self.optimizer, self.lr_scheduler = setup_optimizer_and_scheduler_with_decay(model=self.model, cfg=self.config)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * int(self.config.trainer.gradient_accumulation_steps)
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        is_resume = getattr(self.config.trainer, "is_resume", False)
        resume_from_checkpoint = getattr(self.config, "resume_from_checkpoint", None)

        if is_resume and resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.lr_scheduler.last_epoch = self.resume_step_offset
        self.vla_batches_consumed = self._infer_dataloader_progress_from_step()
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path} at step {self.resume_step_offset}")

    def _infer_dataloader_progress_from_step(self):
        """Infer consumed dataloader batches from the checkpoint step."""
        train_batches = self.resume_step_offset * int(self.accelerator.gradient_accumulation_steps)

        if getattr(self.config.trainer, "skip_nonfinite_loss", False):
            self.accelerator.print(
                "Warning: inferring dataloader resume position from checkpoint step. "
                "If the previous run skipped nonfinite train batches, dataloader resume will not be exact."
            )
        return train_batches

    def _global_step(self):
        return self.resume_step_offset + self.completed_steps

    def _epoch(self):
        """Best-effort epoch for checkpoint naming (matches the logging epoch)."""
        try:
            return int(self.vla_epoch_count)
        except Exception:
            pass
        try:
            return int(self._global_step() / max(1, len(self.vla_train_dataloader)))
        except Exception:
            return 0

    def _save_checkpoint(self):
        """save current training state"""

        # 使用 accelerate 管理所有状态 (model, optimizer, scheduler)
        global_step = self._global_step()
        checkpoint_dir = os.path.join(self.checkpoint_dir, f"step_{global_step}")
        self.accelerator.save_state(checkpoint_dir)

        if accelerator.is_main_process:
            # Consolidated model weights -> a per-checkpoint FOLDER (uploadable as file_type='folder',
            # reusable as a `pretrained_checkpoint`: torch.load(<dir>/pytorch_model.pt)).
            ckpt_folder = os.path.join(self.checkpoint_dir, f"steps_{global_step}")
            os.makedirs(ckpt_folder, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(ckpt_folder, "pytorch_model.pt"))
            # ship the run config alongside so the uploaded folder is self-describing
            cfg_src = os.path.join(self.config.output_dir, "config.yaml")
            if os.path.exists(cfg_src):
                shutil.copy2(cfg_src, os.path.join(ckpt_folder, "config.yaml"))

            # save training metadata
            summary_data = {
                "steps": global_step,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {ckpt_folder}")

        accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        global_step = self._global_step()
        if global_step % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

                # add epoch info
                metrics["epoch"] = round(global_step / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=global_step)
                # debug output
                logger.info(f"Step {global_step}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        if self.vla_batches_consumed > 0:
            dataloader_len = len(self.vla_train_dataloader)
            if dataloader_len <= 0:
                raise RuntimeError("Cannot resume dataloader progress because dataloader length is zero.")

            self.vla_epoch_count = self.vla_batches_consumed // dataloader_len
            batch_offset = self.vla_batches_consumed % dataloader_len
            TrainerUtils._set_dataloader_epoch(self.vla_train_dataloader, self.vla_epoch_count)
            self.accelerator.print(
                f"Resuming dataloader: consumed_batches={self.vla_batches_consumed}, "
                f"epoch={self.vla_epoch_count}, batch_offset={batch_offset}, len={dataloader_len}"
            )

            if batch_offset > 0:
                if not callable(getattr(self.accelerator, "skip_first_batches", None)):
                    raise RuntimeError("Accelerator.skip_first_batches() is required to resume dataloader progress.")
                self.vla_iter = iter(self.accelerator.skip_first_batches(self.vla_train_dataloader, batch_offset))
            else:
                self.vla_iter = iter(self.vla_train_dataloader)
            return

        self.vla_epoch_count = 0
        TrainerUtils._set_dataloader_epoch(self.vla_train_dataloader, self.vla_epoch_count)
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
            self.vla_batches_consumed += 1
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0

            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)
            self.vla_batches_consumed += 1

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            total=self.resume_step_offset + self.config.trainer.max_train_steps,
            initial=self._global_step(),
            disable=not self.accelerator.is_local_main_process,
        )

        step_metrics = {}
        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            skipped_nonfinite = step_metrics.get("skipped_nonfinite_loss", 0.0) == 1.0

            # update progress
            if self.accelerator.sync_gradients and not skipped_nonfinite:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # evaluate model
            if (not skipped_nonfinite) and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics, examples=batch_vla)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            # save checkpoint
            if (not skipped_nonfinite) and self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None, examples=None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        self.accelerator.wait_for_everyone()

        if examples is None:
            raise ValueError("eval_action_model requires examples and must not consume the train iterator.")

        if self.accelerator.is_main_process:
            self.model.eval()

            #examples = self._get_next_batch() #会破坏主循环

            score = 0.0
            num_samples = len(examples)

            batch_images = [example["image"] for example in examples]
            if self.config.datasets.vla_data.get("use_separate", False):
                batch_extra_images = [example["extra_image"] for example in examples]
            else:
                batch_extra_images = None
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"] for example in examples]  # label
            actions_unnorm = [example["action_unnorm"] for example in examples]  # label
            state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

            embodiment_id = None
            if "embodiment_tag" in examples[0]:
                from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EMBODIMENT_TAG_MAPPING
                embodiment_id = torch.tensor(
                    [EMBODIMENT_TAG_MAPPING[example["embodiment_tag"]] for example in examples],
                    dtype=torch.long, device=self.accelerator.device,
                )

            with torch.no_grad():
                # Predict actions using the model
                output_dict = self.model.predict_action(
                    batch_images=batch_images, batch_extra_images=batch_extra_images, instructions=instructions, state=state, embodiment_id=embodiment_id, use_ddim=True, num_ddim_steps=20
                )

            normalized_actions = output_dict["normalized_actions"]  # B, T, D

            # import copy
            # #unapply是in-place操作
            # #actions_dict = {"action": normalized_actions}
            # normalized_actions_to_unnorm = copy.deepcopy(normalized_actions)
            # actions_dict = {"action": normalized_actions_to_unnorm}
            # #这里有问题，unapply是针对各个part的action，但是模型的输出只有一个完整的action，暂时粗暴去掉
            # action_unnormalized = self.vla_train_dataloader.dataset.datasets[0].transforms.unapply(actions_dict)['action']

            actions = np.array(actions)  # convert actions to numpy.ndarray
            actions_unnorm = np.array(actions_unnorm) 
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score

            # score_unnorm = TrainerUtils.euclidean_distance(action_unnormalized, actions_unnorm)
            # average_score_unnorm = score_unnorm / num_pots
            # step_metrics["mse_score_unnorm"] = average_score_unnorm

            self.model.train()

        self.accelerator.wait_for_everyone()
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _nonfinite_loss_status(self, total_loss, output_dict):
        local_finite = torch.isfinite(total_loss.detach()).all()
        finite_flag = local_finite.to(dtype=torch.int32)

        if dist.is_initialized():
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

        bad_keys = []
        if not local_finite.item():
            bad_keys.append("total_loss")
        for key, value in output_dict.items():
            if torch.is_tensor(value) and not torch.isfinite(value.detach()).all().item():
                bad_keys.append(key)

        return finite_flag.item() == 0, bad_keys

    def _assert_finite_loss(self, total_loss, output_dict):
        should_skip, bad_keys = self._nonfinite_loss_status(total_loss, output_dict)
        if should_skip:
            rank = dist.get_rank() if dist.is_initialized() else 0
            raise RuntimeError(
                f"Non-finite loss detected before backward at step {self._global_step()} "
                f"on rank {rank}; local non-finite keys: {bad_keys}"
            )

    def _handle_nonfinite_skip(self, bad_keys):
        self.optimizer.zero_grad()
        self.consecutive_nonfinite_skips += 1

        rank = dist.get_rank() if dist.is_initialized() else 0
        max_skips = self.config.trainer.get("max_consecutive_nonfinite_skips", 100)
        logger.warning(
            f"Skipping non-finite batch before backward at step {self._global_step()} "
            f"on rank {rank}; local non-finite keys: {bad_keys}; "
            f"consecutive skips: {self.consecutive_nonfinite_skips}/{max_skips}"
        )

        if self.consecutive_nonfinite_skips > max_skips:
            raise RuntimeError(
                f"Exceeded max_consecutive_nonfinite_skips={max_skips} at step {self._global_step()} "
                f"on rank {rank}; last local non-finite keys: {bad_keys}"
            )

        return {
            "skipped_nonfinite_loss": 1.0,
            "nonfinite_skip_count": float(self.consecutive_nonfinite_skips),
        }

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):

            # Inject training progress so frameworks can schedule step-dependent terms
            progress = self.completed_steps / max(self.config.trainer.max_train_steps, 1)
            for example in batch_vla:
                example["_training_progress"] = progress

            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss




            should_skip, bad_keys = self._nonfinite_loss_status(total_loss, output_dict)
            if should_skip:
                if self.config.trainer.get("skip_nonfinite_loss", False):
                    return self._handle_nonfinite_skip(bad_keys)
                rank = dist.get_rank() if dist.is_initialized() else 0
                raise RuntimeError(
                    f"Non-finite loss detected before backward at step {self._global_step()} "
                    f"on rank {rank}; local non-finite keys: {bad_keys}"
                )

            self.consecutive_nonfinite_skips = 0

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # Optimizer and scheduler advance only on a synchronized (global) step.
            # Advancing the scheduler on every micro-step makes cosine schedules
            # finish early and then rise again after their nominal endpoint.
            if self.accelerator.sync_gradients:
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.config.trainer.gradient_clipping
                    )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

        metrics = {"action_dit_loss": action_loss.item(), "skipped_nonfinite_loss": 0.0}
        return metrics     

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            cfg_src = os.path.join(self.config.output_dir, "config.yaml")
            if os.path.exists(cfg_src):
                shutil.copy2(cfg_src, os.path.join(final_checkpoint, "config.yaml"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")


        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()



def main(cfg) -> None:
    # Prevent misleading pctXX run names from silently using legacy num_shot.
    import re
    pct_match = re.search(r"(?:^|_)pct(25|50|75|100)_", str(getattr(cfg, "run_id", "")))
    if pct_match is not None:
        expected_pct = float(pct_match.group(1))
        data_cfg = cfg.datasets.vla_data
        actual_pct = data_cfg.get("data_percent", None)
        num_shot = data_cfg.get("num_shot", None)
        if actual_pct is None or float(actual_pct) != expected_pct or num_shot is not None:
            raise ValueError(
                f"Run {cfg.run_id!r} declares pct{int(expected_pct)}, but "
                f"data_percent={actual_pct!r}, num_shot={num_shot!r}. "
                "Percentage runs require matching data_percent and num_shot=null."
            )

    logger.info("VLA Training :: Warming Up")


    accum = sync_gradient_accumulation_steps(accelerator, cfg)
    logger.info(
        f"Synced gradient_accumulation_steps={accum} "
        f"(accelerator={accelerator.gradient_accumulation_steps})"
    )

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    # _finalize_training() already synchronizes every rank after writing the
    # final model.  NCCL can report a late CUDA device-busy/unavailable error
    # while tearing down a healthy completed job (for example when the
    # scheduler starts reclaiming devices).  Cleanup must not turn a
    # successfully saved training run into a failed job.
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:
            logger.warning("Ignoring distributed cleanup error after successful training: %s", exc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/pi_backbone_train_gr1.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
