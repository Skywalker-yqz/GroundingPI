# Upstream attribution: the named implementation credits and retained
# legacy remarks in this file come from public StarVLA source/history
# (https://github.com/starVLA/starVLA), including revision
# f18fbc22c317dd1810839cb621632ac45add93f1 where applicable. They identify
# upstream contributions, not the authors or affiliations of this submission.

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].
import os
import torch
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoProcessor
# NOTE: Qwen3VLForConditionalGeneration is imported LAZILY in __init__ (dual-mode: stock
# transformers on the new >=5.9 env, else the fork's vendored modeling_qwen3_vl on the old
# conda env). Importing the vendored file eagerly here would crash on transformers>=5.9.
from starVLA.model.modules.vlm._visual_capture import ImageEmbedsCapture
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")

        # Dual-mode load so this interface runs on BOTH environments:
        #   NEW env (system-python bootstrap, transformers>=5.9): STOCK Qwen3VL. Its output
        #     has no `image_embeds` field, so we inject it via ImageEmbedsCapture (below),
        #     so newer environments keep working.
        #   OLD env (conda starVLA, transformers 4.5x): stock class is absent -> fall back to
        #     the fork's vendored modeling_qwen3_vl, which exposes image_embeds natively.
        # Discriminate by transformers VERSION, not by ImportError: the old conda env may well
        # ship a stock Qwen3VLForConditionalGeneration (it entered transformers ~4.57), so a
        # try/except would wrongly pick stock there and change the validated conda behavior.
        # The real constraint is the cutover at 5.9: vendored modeling_qwen3_vl can't load on
        # >=5.9, and stock's LM output drops `image_embeds` on >=5.9 (needs capture).
        # 原来的starvla里用的版本<5.9，有自己的qwen3vl fork，所以这里需要判断版本
        import transformers as _tf
        from packaging import version as _ver
        if _ver.parse(_tf.__version__) >= _ver.parse("5.9"):
            # 记录这次是用哪种方式加载的 Qwen3-VL，从而决定要不要挂 ImageEmbedsCapture shim 去拿 image_embeds。
            from transformers import Qwen3VLForConditionalGeneration as _Qwen3VL
            self._use_capture = True            # stock: inject image_embeds via ImageEmbedsCapture
        else:
            from starVLA.model.modules.vlm.modeling_qwen3_vl import Qwen3VLForConditionalGeneration as _Qwen3VL
            self._use_capture = False           # vendored: exposes image_embeds natively

        attn_impl = os.environ.get("STARVLA_ATTN_IMPL") or qwenvl_config.get(
            "attn_implementation", "flash_attention_2"
        )
        model = _Qwen3VL.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        processor.image_processor.size = {"shortest_edge": 196*16*16, "longest_edge": 1024*16*16}

        self.model = model
        self.processor = processor
        self.config = config

        # expose the text hidden size at the top level of the config
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        select_layer = config.framework.qwenvl.select_layer
        print(f"Selected LLM Layer: {select_layer}")

        if select_layer != -1:
            if hasattr(self.model.language_model, "model"):
                while len(self.model.language_model.model.layers) > select_layer:
                    self.model.language_model.model.layers.pop(-1)
            else:
                while len(self.model.language_model.layers) > select_layer:
                    self.model.language_model.layers.pop(-1)

        # Stock transformers' Qwen3-VL LM output lacks `image_embeds`; capture it from the
        # model's own get_image_features (no vision recompute). Vendored modeling already
        # provides it natively, so we only attach the shim on the stock path.
        self._img_capture = ImageEmbedsCapture(self.model) if self._use_capture else None

        # transformers 4.x returned hidden_states[-1] BEFORE the final RMSNorm; >=5.x returns it
        # AFTER. Checkpoints trained on 4.x learned their
        # action head on the pre-norm features, so evaluating them here needs the 4.x convention.
        # Opt-in only: models trained on THIS branch (5.x) must keep the post-norm default.
        self._legacy_pre_norm_last_hidden = (
            os.environ.get("STARVLA_LEGACY_LAST_HIDDEN", "0") == "1"
        )
        if self._legacy_pre_norm_last_hidden:
            print("Qwen3-VL interface: legacy 4.x hidden_states[-1] (pre-final-norm) enabled")
            self._pre_norm_cache = {}

            def _capture_norm_input(module, args):
                self._pre_norm_cache["h"] = args[0]

            self.model.model.language_model.norm.register_forward_pre_hook(_capture_norm_input)

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to the underlying Qwen3-VL backbone.
        Returns (outputs, attention_mask), the contract QwenPI consumes.
        """

        # Qwen3-VL M-RoPE (transformers>=5) requires mm_token_type_ids (text=0/image=1/video=2)
        # whenever image_grid_thw/video_grid_thw is passed. Derive it from input_ids here
        # (same rule as Processor.create_mm_token_type_ids) when it is missing.
        if (
            kwargs.get("mm_token_type_ids") is None
            and kwargs.get("input_ids") is not None
            and (kwargs.get("image_grid_thw") is not None or kwargs.get("video_grid_thw") is not None)
        ):
            input_ids = kwargs["input_ids"]
            image_token_id = getattr(self.model.config, "image_token_id", IMAGE_TOKEN_INDEX)
            video_token_id = getattr(self.model.config, "video_token_id", VIDEO_TOKEN_INDEX)
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == image_token_id] = 1
            mm_token_type_ids[input_ids == video_token_id] = 2
            kwargs["mm_token_type_ids"] = mm_token_type_ids

        if self._img_capture is not None:
            self._img_capture.reset()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        # On stock transformers, attach the captured ViT tokens (no recompute) so
        # ``outputs.image_embeds`` is available on both load paths.
        if self._img_capture is not None and getattr(outputs, "image_embeds", None) is None \
           and self._img_capture.cached is not None:
            outputs["image_embeds"] = self._img_capture.cached

        # Legacy 4.x eval (opt-in STARVLA_LEGACY_LAST_HIDDEN): swap hidden_states[-1] with the
        # pre-final-RMSNorm features that 4.x-trained action heads expect (see __init__ hook).
        if self._legacy_pre_norm_last_hidden and getattr(outputs, "hidden_states", None):
            pre_norm = self._pre_norm_cache.pop("h", None)
            if pre_norm is not None:
                outputs["hidden_states"] = tuple(outputs.hidden_states[:-1]) + (pre_norm,)

        return outputs, kwargs['attention_mask']

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Two-stage path: chat template text + process_vision_info images, then
        # processor(...). Relying only on apply_chat_template(tokenize=True) can drop
        # pixel_values, which then trips UnboundLocalError on image_embeds in the
        # vendored modeling_qwen3_vl.forward.
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(solutions is None))
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return batch_inputs.to(self.model.device)
