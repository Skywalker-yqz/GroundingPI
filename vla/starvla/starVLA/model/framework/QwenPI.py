# Upstream attribution: the named implementation credits and retained
# legacy remarks in this file come from public StarVLA source/history
# (https://github.com/starVLA/starVLA), including revision
# f18fbc22c317dd1810839cb621632ac45add93f1 where applicable. They identify
# upstream contributions, not the authors or affiliations of this submission.

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
QwenPI framework
Qwen3-VL backbone + the fixed layerwise flow-matching action expert (PI-style) that
directly predicts continuous actions. The flow-matching head derives from GR00T N1.5.
"""
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model, LayerwiseFlowmatchingActionHead
from starVLA.model.modules.projector.PiBackboneConditioner import (
    PiBackboneConditioner,
    normalized_depth_indices,
)
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen3-VL interface for fused language/vision token embeddings
      - Layer-wise cross DiT diffusion head 
      

    Focus: Predict future continuous actions conditioned on images + instruction.
    """
# 
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Qwen-pi fine-tunes the language backbone but keeps the pretrained
        # vision tower fixed, matching the validated freeze-ViT GR1 setting.
        qwen_model = self.qwen_vl_interface.model
        visual = getattr(getattr(qwen_model, "model", None), "visual", None)
        if visual is None:
            visual = getattr(qwen_model, "visual", None)
        if visual is None:
            raise AttributeError("QwenPI could not locate the Qwen vision tower to freeze")
        visual.requires_grad_(False)

        text_config = getattr(self.qwen_vl_interface.model.config, "text_config", self.qwen_vl_interface.model.config)
        llm_layers = int(text_config.num_hidden_layers)
        llm_hidden_size = int(text_config.hidden_size)
        self.tap_indices = normalized_depth_indices(llm_layers, 8)
        self.backbone_conditioner = PiBackboneConditioner(llm_hidden_size)

        # Keep the StarVLA pi expert identical across evaluated backbones.
        action_cfg = self.config.framework.action_model
        action_cfg.hidden_size = 1024
        action_cfg.DiTConfig = {
            "num_layers": 16,
            "input_embedding_dim": 1024,
            "attention_head_dim": 64,
            "num_attention_heads": 16,
        }
        action_cfg.diffusion_model_cfg.num_layers = 16
        action_cfg.diffusion_model_cfg.num_attention_heads = 16
        action_cfg.diffusion_model_cfg.attention_head_dim = 64
        action_cfg.diffusion_model_cfg.output_dim = 1024
        action_cfg.diffusion_model_cfg.cross_attention_dim = 1024
        action_cfg.diffusion_model_cfg.interleave_self_attention = True
        action_cfg.num_target_vision_tokens = 32
        action_cfg.repeated_diffusion_steps = 2
        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        训练前向：直接回归未来动作（无扩散）。

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            qwenvl_outputs = output[0] if isinstance(output, tuple) else output
            attention_mask = output[1] if isinstance(output, tuple) else qwen_inputs.get("attention_mask")
            all_hidden = qwenvl_outputs.hidden_states
            # hidden_states includes the embedding output at index 0.
            raw_taps = [all_hidden[index + 1] for index in self.tap_indices]
            vl_embs_list = self.backbone_conditioner(
                raw_taps, key_padding_mask=attention_mask
            )
            base_hidden = vl_embs_list[-1]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = int(
                self.config.framework.action_model.repeated_diffusion_steps
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # 对每层特征做 repeat
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)



        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            qwenvl_outputs = output[0] if isinstance(output, tuple) else output
            attention_mask = output[1] if isinstance(output, tuple) else qwen_inputs.get("attention_mask")
            all_hidden = qwenvl_outputs.hidden_states
            raw_taps = [all_hidden[index + 1] for index in self.tap_indices]
            vl_embs_list = self.backbone_conditioner(
                raw_taps, key_padding_mask=attention_mask
            )
            base_hidden = vl_embs_list[-1]

        state = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state)  # (B, chunk_len, action_dim)

        # Keep the native StarVLA inference contract: modality-transform
        # ``unapply`` consumes tensors and performs the final conversion after
        # action denormalization.  Returning NumPy here bypasses that contract.
        normalized_actions = pred_actions.detach().cpu()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def get_action(self, data: Dict[str, Any]) -> Any:
        """Eval entry used by ``VLAInferencePolicy``.

        This keeps QwenPI on the same inference contract as the existing
        StarVLA pi-style frameworks: the evaluation transform supplies a
        batched video tensor, language annotation, and normalized state; the
        model returns normalized action chunks for the transform to unapply.
        """
        from transformers.feature_extraction_utils import BatchFeature

        video_data = data["video"]
        if hasattr(video_data, "ndim") and video_data.ndim == 6:
            video_data = video_data.squeeze(1)

        batch_size = int(video_data.shape[0])
        target_size = tuple(self.config.datasets.vla_data.image_size)
        batch_images: List[List[Image.Image]] = []
        for i in range(batch_size):
            images = []
            for j in range(video_data[i].shape[0]):
                frame = video_data[i][j]
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().numpy()
                images.append(Image.fromarray(np.asarray(frame)).resize(target_size))
            batch_images.append(images)

        instructions = data.get("annotation.task_index")
        if instructions is None:
            instructions = data.get("annotation.human.coarse_action")
        if instructions is None:
            instructions = [""] * batch_size
        elif isinstance(instructions, np.ndarray):
            instructions = instructions.tolist()
        elif isinstance(instructions, torch.Tensor):
            instructions = instructions.detach().cpu().tolist()
        if isinstance(instructions, str):
            instructions = [instructions] * batch_size

        state = data.get("state")
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()

        outputs = self.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            state=state,
        )
        return BatchFeature(data=outputs)



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/pi_backbone_train_gr1.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    model = Qwen_PI(cfg)
    print(model)


    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
