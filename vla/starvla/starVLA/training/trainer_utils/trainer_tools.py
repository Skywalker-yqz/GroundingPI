"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

from typing import Tuple
import re
import json
import numpy as np
import torch
import torch.nn as nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from accelerate.logging import get_logger

logger = get_logger(__name__)


def sync_gradient_accumulation_steps(accelerator, cfg) -> int:
    """Apply cfg.trainer.gradient_accumulation_steps to Accelerator + DeepSpeed.

    Accelerate creates Accelerator/DeepSpeedPlugin before YAML is loaded, and
    ds_config.yaml historically hardcodes gradient_accumulation_steps=1. Without
    this sync, training silently runs with accum=1 while logs print the YAML value.
    """
    accum = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1) or 1)
    if accum < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {accum}")

    accelerator.gradient_accumulation_steps = accum
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is not None and getattr(plugin, "deepspeed_config", None) is not None:
        plugin.deepspeed_config["gradient_accumulation_steps"] = accum

    return accum


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    used_params = set()
    param_groups = []

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            params = list(module.parameters())
            if lr != 0.0:
                param_groups.append({"params": params, "lr": lr, "name": module_name})
            used_params.update(id(p) for p in params)
        except AttributeError:
            ReferenceError(f"⚠️ module path `{module_name}` not found in vla")

    # assign base learning rate to the remaining unused parameters
    other_params = [p for p in model.parameters() if id(p) not in used_params]

    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups

def get_no_decay_param_names(model: nn.Module, all_layernorm_layers: list) -> set:
    """
    获取所有不应进行权重衰减的参数的完整名称（biases 和 LayerNorms）。
    """
    no_decay_param_names = set()
    
    # 遍历所有模块，查找 LayerNorm
    for module_name, module in model.named_modules():
        is_layer_norm = False
        for layer_norm_class in all_layernorm_layers:
            if isinstance(module, layer_norm_class):
                is_layer_norm = True
                break
        
        if is_layer_norm:
            # 将 LayerNorm 模块中的所有参数（weight 和 bias）添加到 no_decay 集合
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                    no_decay_param_names.add(full_param_name)

    # 遍历所有参数，查找 biases
    for param_name, param in model.named_parameters():
        if param.requires_grad and param_name.endswith(".bias"):
            no_decay_param_names.add(param_name)
            
    return no_decay_param_names

import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from torchvision.ops import box_iou
from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


import torch.distributed as dist


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        print("#"*30)
        print(freeze_modules)
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # Check if the attribute is a direct nn.Parameter or an nn.Module
                    if isinstance(module, torch.nn.Parameter):
                        # Direct parameter: freeze it directly
                        module.requires_grad = False
                        frozen.append(path)
                    elif isinstance(module, torch.nn.Module):
                        # Module: freeze all its parameters
                        for param in module.parameters():
                            param.requires_grad = False
                        frozen.append(path)
                    else:
                        print(f"⚠️ {path} is neither nn.Module nor nn.Parameter, cannot freeze")
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        dist.barrier()  # synchronize when distributed training
        if dist.get_rank == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.get_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    # Per-embodiment adapter tensors split by role. A brand-new embodiment slot
    # (index >= the checkpoint's slot count) must NOT copy the raw action/state
    # columns of a reference slot, because two embodiments with different raw
    # layouts (different raw action widths) are NOT aligned
    # dim-by-dim. We instead:
    #   - copy SHARED feature-space layers from the reference slot (scale only), and
    #   - random-init the RAW-space boundary layers with a std calibrated to match
    #     the reference slot's output variance (so the pretrained DiT receives
    #     correctly-scaled features for the new embodiment from step 0).
    _WS_SHARED_COPY = (
        "state_encoder.layer1.b",
        "state_encoder.layer2.W",
        "state_encoder.layer2.b",
        "action_encoder.W1.b",
        "action_encoder.W2.W",
        "action_encoder.W2.b",
        "action_encoder.W3.W",
        "action_encoder.W3.b",
        "action_decoder.layer1.W",
        "action_decoder.layer1.b",
    )
    _WS_BOUNDARY_IN_W = ("state_encoder.layer1.W", "action_encoder.W1.W")  # input dim grows
    _WS_BOUNDARY_OUT_W = ("action_decoder.layer2.W",)  # output dim grows, input dim fixed
    _WS_BOUNDARY_ZERO_B = ("action_decoder.layer2.b",)  # new raw-output biases -> 0

    @staticmethod
    def _warmstart_new_embodiment_slots(new_state_dict, checkpoint_state_dict, model_state_dict, ref_slot):
        """In-place: initialize brand-new embodiment slots (indices beyond the
        checkpoint's slot count) by warm-starting from a TRAINED reference slot.

        Output-variance-preserving math (unit-variance iid input):
          per-output-unit var = (in_dim) * mean(W^2). To match a reference slot's
          per-output variance with a new (possibly different) input dim:
            sigma_new = sqrt(old_in / new_in) * rms(W_ref[:old_in])
          For an output-growing layer (input dim unchanged): sigma_new = rms(W_ref).
        This reproduces the reference adapter's feature scale without copying any
        raw action/state column (verified in tools/compute_slot_init.py).
        """
        gen = torch.Generator().manual_seed(12345)  # deterministic across ranks
        rank0 = dist.get_rank() == 0
        summary = []
        for k in list(new_state_dict.keys()):
            is_shared = any(k.endswith(s) for s in TrainerUtils._WS_SHARED_COPY)
            is_in_w = any(k.endswith(s) for s in TrainerUtils._WS_BOUNDARY_IN_W)
            is_out_w = any(k.endswith(s) for s in TrainerUtils._WS_BOUNDARY_OUT_W)
            is_zero_b = any(k.endswith(s) for s in TrainerUtils._WS_BOUNDARY_ZERO_B)
            if not (is_shared or is_in_w or is_out_w or is_zero_b):
                continue
            if k not in checkpoint_state_dict:
                continue
            ck_t = checkpoint_state_dict[k]
            full = new_state_dict[k]
            ck_slots, model_slots = ck_t.shape[0], full.shape[0]
            if model_slots <= ck_slots:
                continue  # no genuinely-new slot for this tensor
            if ref_slot < 0 or ref_slot >= ck_slots:
                raise ValueError(
                    f"reference_embodiment_slot={ref_slot} out of range for checkpoint "
                    f"slots [0,{ck_slots}) on key {k}"
                )
            work = full.clone().float()
            for new_slot in range(ck_slots, model_slots):
                if is_shared:
                    work[new_slot] = work[ref_slot].clone()
                elif is_in_w:
                    old_in, new_in = ck_t.shape[1], work.shape[1]
                    rms_ref = work[ref_slot][:old_in].pow(2).mean().clamp_min(1e-24).sqrt().item()
                    sigma = (old_in / new_in) ** 0.5 * rms_ref
                    work[new_slot] = torch.randn(work[new_slot].shape, generator=gen) * sigma
                elif is_out_w:
                    old_out = ck_t.shape[2]
                    rms_ref = work[ref_slot][:, :old_out].pow(2).mean().clamp_min(1e-24).sqrt().item()
                    work[new_slot] = torch.randn(work[new_slot].shape, generator=gen) * rms_ref
                elif is_zero_b:
                    work[new_slot] = 0.0
            new_state_dict[k] = work.to(model_state_dict[k].dtype)
            if rank0:
                kind = "shared-copy" if is_shared else ("zero-bias" if is_zero_b else "rand-calibrated")
                summary.append(f"   - {k}: slots [{ck_slots},{model_slots}) <- {kind} (ref slot {ref_slot})")
        if rank0:
            print(f"🌱 warm-started {len(summary)} new-embodiment adapter tensors from reference slot {ref_slot}:")
            for line in summary:
                print(line)

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None, reference_embodiment_slot=None):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)
        - if reference_embodiment_slot is set (full-load only), brand-new embodiment
          slots are warm-started from that trained reference slot instead of the
          tiny 0.02 fresh init (fixes the DiT-input scale mismatch for a new robot).

        return:
            replace, loaded_modules: list of module paths that successfully loaded parameters; if global load, then ["<full_model>"]
        """
        if not checkpoint_path:
            return []
        if dist.get_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        loaded_modules = []

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if dist.get_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                model_state_dict = model.state_dict()
                checkpoint_state_dict = checkpoint
                new_state_dict = {}
                ignored_keys = []

                for k, v in checkpoint_state_dict.items():
                    if k not in model_state_dict:
                        new_state_dict[k] = v
                        continue

                    EXPANDABLE_KEY_PATTERNS = (
                        "action_encoder.W1.",
                        "action_encoder.W2.",
                        "action_encoder.W3.",
                        "state_encoder.",
                        "action_decoder.",
                    )

                    if v.shape == model_state_dict[k].shape:
                        new_state_dict[k] = v
                    elif (
                        any(pat in k for pat in EXPANDABLE_KEY_PATTERNS)
                        and len(v.shape) == len(model_state_dict[k].shape)
                        and all(cs <= ms for cs, ms in zip(v.shape, model_state_dict[k].shape))
                    ):
                        expanded = model_state_dict[k].clone()
                        slices = tuple(slice(0, s) for s in v.shape)
                        expanded[slices] = v
                        new_state_dict[k] = expanded
                        if dist.get_rank() == 0:
                            print(f"🔄 EXPANDED {k}: checkpoint {v.shape} → model {model_state_dict[k].shape}")
                    else:
                        if dist.get_rank() == 0:
                            print(f"⚠️ IGNORING size mismatch for {k}: checkpoint shape {v.shape} vs model shape {model_state_dict[k].shape}")
                        ignored_keys.append(k)

                if reference_embodiment_slot is not None:
                    TrainerUtils._warmstart_new_embodiment_slots(
                        new_state_dict, checkpoint_state_dict, model_state_dict, int(reference_embodiment_slot)
                    )

                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                if dist.get_rank() == 0:
                    print(f"✅ loaded <full_model> model parameters except missing {missing} and unexpeced {unexpected}")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}")
        return model

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _set_dataloader_epoch(dataloader, epoch):
        """Set epoch on both distributed sampler and dataset-level sampler."""
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch)

        base_dataloader = getattr(dataloader, "base_dataloader", dataloader)
        dataset = getattr(base_dataloader, "dataset", None)
        if not callable(getattr(dataset, "set_epoch", None)):
            raise RuntimeError("Dataloader dataset must expose set_epoch() for deterministic epoch sampling.")
        dataset.set_epoch(epoch)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed sampler + dataset-level deterministic sampling)
        TrainerUtils._set_dataloader_epoch(dataloader, epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        compute the cosine angle between two groups of gradient vectors (degrees), and calculate the average angle and variance.
        grads_a, grads_v: gradient Tensor list corresponding to the same parameter list interface_params
        return:
            mean_angle_deg: average angle (degrees)
            angle_variance: angle variance
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # dist.barrier()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        apply PCGrad projection to the second group of gradients grads_v, suppress negative transfer between grads_a and grads_v
        if the dot product of two groups of gradients < 0, then:
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        return the new grads_v list
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        evaluate QwenQFormerDiT model, compute IoU and action distance.

        Args:
            qwenpi: QwenQFormerDiT model instance.
            dataloader: data loader.
            num_batches: number of batches to evaluate.

        Returns:
            dict: contains IoU and action distance evaluation results.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        extract valid JSON part from string and convert to dictionary.

        Args:
            input_string (str): string containing extra characters.

        Returns:
            dict: dictionary extracted and parsed.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None


import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0
