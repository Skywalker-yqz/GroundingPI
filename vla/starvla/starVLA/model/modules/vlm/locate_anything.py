"""LocateAnything adapter for StarVLA's native VLMBackbonePI framework."""

from __future__ import annotations

import inspect
import os
import sys

import torch

from starVLA.model.modules.vlm.comparison_backbones import _StarVLAHFInterface


class LocateAnythingInterface(_StarVLAHFInterface):
    def __init__(self, config) -> None:
        if not bool(config.framework.qwenvl.get("allow_remote_code", False)):
            raise ValueError(
                "LocateAnything executes checkpoint-provided Python. Set "
                "framework.qwenvl.allow_remote_code=true for this trusted local checkpoint."
            )
        self._isolate_module_cache()
        super().__init__(config)

    @staticmethod
    def _isolate_module_cache() -> None:
        rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
        base = os.environ.get(
            "HF_MODULES_CACHE",
            os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "modules",
            ),
        )
        suffix = f"_starvla_rank{rank}"
        target = base if base.endswith(suffix) else base.rstrip("/") + suffix
        os.makedirs(target, exist_ok=True)
        init_file = os.path.join(target, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, "a").close()
        os.environ["HF_MODULES_CACHE"] = target
        if target not in sys.path:
            sys.path.insert(0, target)
        # transformers may already have cached this constant at import time.
        import transformers.dynamic_module_utils as dynamic_modules

        dynamic_modules.HF_MODULES_CACHE = target

    def _load(self):
        from transformers import AutoConfig, AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )
        model_config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        for subconfig in (model_config, model_config.text_config):
            subconfig._attn_implementation = "sdpa"
            subconfig._attn_implementation_internal = "sdpa"
        model = AutoModel.from_pretrained(
            self.model_id,
            config=model_config,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
        # The remote constructor independently propagates the checkpoint's
        # flash_attention_2 value into its custom Qwen2Model, whose forward only
        # implements the bespoke block mask for "sdpa" (or unavailable "magi").
        text_model = model.language_model.model
        text_model._attn_implementation = "sdpa"
        text_model.config._attn_implementation = "sdpa"
        model.language_model.config._attn_implementation = "sdpa"
        model.config.text_config._attn_implementation = "sdpa"
        return model, processor

    @property
    def _text_config(self):
        # LocateAnything keeps Qwen geometry under text_config.
        return self.model.config.text_config

    def get_visual_module(self):
        return self.model.vision_model

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
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
        flat_images = [image for sample_images in images for image in sample_images]
        inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            truncation=True,
            max_length=int(self.config.framework.qwenvl.get("max_length", 2048)),
            return_tensors="pt",
        )
        # Remote processors occasionally return numpy grid metadata.
        for key, value in list(inputs.items()):
            if not torch.is_tensor(value):
                try:
                    inputs[key] = torch.as_tensor(value)
                except (TypeError, ValueError):
                    inputs.pop(key)
        return inputs.to(self.model.device)

    @staticmethod
    def _call(module, inputs, **controls):
        params = inspect.signature(module.forward).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            controls = {key: value for key, value in controls.items() if key in params}
        return module(**inputs, **controls)

    def forward(self, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        # QwenPI supplies these controls; this adapter sets the same values
        # explicitly below after filtering against the remote forward signature.
        for key in (
            "output_hidden_states",
            "output_attentions",
            "return_dict",
            "use_cache",
            "past_key_values",
        ):
            kwargs.pop(key, None)
        model_inputs, original_length = self._with_prefix_boundary(kwargs)

        language_model = getattr(self.model, "language_model", None)
        overrides = [
            (getattr(language_model, "model", None), True),
            (language_model, False),
        ]
        saved = [(module, module.training) for module, _ in overrides if module is not None]
        for module, flag in overrides:
            if module is not None:
                module.training = flag
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self._call(
                    self.model,
                    model_inputs,
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            for module, old_flag in saved:
                module.training = old_flag

        if getattr(outputs, "hidden_states", None) is not None:
            outputs.hidden_states = tuple(
                hidden[:, :original_length] for hidden in outputs.hidden_states
            )
        if getattr(outputs, "logits", None) is not None:
            outputs.logits = outputs.logits[:, :original_length]
        return outputs, attention_mask

    def _with_prefix_boundary(self, inputs):
        input_ids = inputs["input_ids"]
        batch, length = input_ids.shape
        device = input_ids.device
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            real_lengths = torch.full((batch, 1), length, device=device)
        else:
            real_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        real_lengths = real_lengths.clamp(min=1, max=length)
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", 0) or 0

        extended = dict(inputs)
        extended["input_ids"] = torch.cat(
            [input_ids, torch.full((batch, 1), pad_id, dtype=input_ids.dtype, device=device)],
            dim=1,
        )
        if attention_mask is not None:
            extended["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_zeros((batch, 1))], dim=1
            )
        positions = torch.arange(length + 1, device=device).unsqueeze(0)
        extended["position_ids"] = torch.where(
            positions < real_lengths, positions, positions - real_lengths
        )
        return extended, length


__all__ = ["LocateAnythingInterface"]
