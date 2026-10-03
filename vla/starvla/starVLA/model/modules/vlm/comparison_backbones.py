"""StarVLA-native adapters for the additional fixed-backbone comparison.

Each adapter implements the existing QwenPI contract: ``model``,
``build_qwenvl_inputs`` and ``forward -> (outputs, attention_mask)``.  No
OpenWAM architecture or action module is imported here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn


class _StarVLAHFInterface(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model_id = str(config.framework.qwenvl.base_vlm)
        self.model, self.processor = self._load()
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    def _load(self):
        raise NotImplementedError

    @property
    def _text_config(self):
        return getattr(self.model.config, "text_config", self.model.config)

    @property
    def hidden_size(self) -> int:
        return int(self._text_config.hidden_size)

    @property
    def num_layers(self) -> int:
        return int(self._text_config.num_hidden_layers)

    def get_visual_module(self):
        for path in (
            ("model", "visual"),
            ("visual",),
            ("vision_tower",),
            ("model", "vision_tower"),
            ("vision_model",),
        ):
            module = self.model
            for name in path:
                module = getattr(module, name, None)
                if module is None:
                    break
            if module is not None:
                return module
        return None

    def forward(self, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
        return outputs, attention_mask

    def _prompt(self, instruction: str) -> str:
        cot = self.config.datasets.vla_data.get("CoT_prompt", None)
        return cot.replace("{instruction}", instruction) if cot else instruction


class RexOmniInterface(_StarVLAHFInterface):
    def _load(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation=self.config.framework.qwenvl.get(
                "attn_implementation", "flash_attention_2"
            ),
            device_map="cuda",
        )
        self._validate_tokenizer(processor, model)
        return model, processor

    def _validate_tokenizer(self, processor, model) -> None:
        path = Path(self.model_id) / "tokenizer_config.json"
        if not path.is_file():
            return
        decoder = json.loads(path.read_text()).get("added_tokens_decoder", {})
        declared = {
            str(entry["content"]): int(token_id)
            for token_id, entry in decoder.items()
            if isinstance(entry, dict) and "content" in entry
        }
        runtime = processor.tokenizer.get_added_vocab()
        wrong = {token: idx for token, idx in declared.items() if runtime.get(token) != idx}
        if wrong:
            sample = list(wrong.items())[:3]
            raise ValueError(
                "Rex-Omni tokenizer IDs disagree with its checkpoint even with "
                f"use_fast=False: {sample}. Refusing to risk embedding overflow."
            )
        max_token = max(runtime.values(), default=-1)
        embeddings = model.get_input_embeddings().num_embeddings
        if max_token >= embeddings:
            raise ValueError(
                f"Rex-Omni tokenizer id {max_token} exceeds {embeddings} embedding rows"
            )

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        from qwen_vl_utils import process_vision_info

        messages = []
        for sample_images, instruction in zip(images, instructions):
            content = [
                {"type": "image", "image": image} for image in sample_images
            ]
            content.append({"type": "text", "text": self._prompt(instruction)})
            messages.append([{"role": "user", "content": content}])
        texts = [
            self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)


class PaliGemmaInterface(_StarVLAHFInterface):
    def _load(self):
        import warnings

        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        # PaliGemma is a prefix-LM: image + prompt tokens attend bidirectionally.
        # Only the sdpa/eager paths honour HF's 4D prefix mask; under
        # flash_attention_2 the model silently runs GemmaAttention with
        # is_causal=True and image tokens never see the instruction.
        attn_implementation = str(
            self.config.framework.qwenvl.get("attn_implementation", "sdpa")
        )
        if attn_implementation == "flash_attention_2":
            warnings.warn(
                "PaliGemma with attn_implementation=flash_attention_2 runs fully "
                "causal instead of prefix-bidirectional. Kept as configured so "
                "checkpoints trained this way still evaluate as trained; use "
                "sdpa for new training runs.",
                stacklevel=2,
            )
        processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map="cuda",
        )
        return model, processor

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        # PaliGemma supports multiple images per prompt as a nested image list.
        inputs = self.processor(
            text=[self._prompt(text) for text in instructions],
            images=images,
            padding=True,
            truncation=True,
            max_length=int(self.config.framework.qwenvl.get("max_length", 2048)),
            return_tensors="pt",
        )
        return inputs.to(self.model.device)


class RynnBrain2Interface(_StarVLAHFInterface):
    def _load(self):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16, attn_implementation=self.config.framework.qwenvl.get("attn_implementation", "sdpa"), device_map="cuda")
        return model, processor
    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        messages=[]
        for sample_images, instruction in zip(images, instructions):
            content=[{"type":"image","image":image} for image in sample_images]
            content.append({"type":"text","text":self._prompt(instruction)})
            messages.append([{"role":"user","content":content}])
        inputs=self.processor.apply_chat_template(messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        return inputs.to(self.model.device)

def get_comparison_vlm_model(config):
    model_path = str(config.framework.qwenvl.base_vlm)
    try:
        model_type = json.loads((Path(model_path) / "config.json").read_text()).get(
            "model_type"
        )
    except (OSError, ValueError):
        model_type = None

    explicit = str(config.framework.qwenvl.get("vlm_type", "")).lower()
    if model_type == "paligemma" or explicit == "paligemma":
        return PaliGemmaInterface(config)
    if model_type == "qwen2_5_vl" or explicit in {"rex", "rex_omni"}:
        return RexOmniInterface(config)
    if explicit in {"rynnbrain2", "rynn2"}:
        return RynnBrain2Interface(config)
    if model_type == "locateanything" or explicit in {"locate", "locateanything"}:
        from starVLA.model.modules.vlm.locate_anything import LocateAnythingInterface

        return LocateAnythingInterface(config)
    raise NotImplementedError(
        f"Unsupported comparison backbone at {model_path} (model_type={model_type!r})"
    )


__all__ = [
    "PaliGemmaInterface",
    "RexOmniInterface",
    "RynnBrain2Interface",
    "get_comparison_vlm_model",
]
