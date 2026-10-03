# Third-party source note: upstream names, author credits and public URLs in
# this derived model implementation identify external sources, not submission
# authors. See THIRD_PARTY_NOTICES.md; original license notices are retained.
# GroundingPI is the public model name. GroundingPI remains in internal module names
# and serialized model_type values for checkpoint compatibility. Historical
# third-party attribution is documented in THIRD_PARTY_NOTICES.md.

"""Processor glue for GroundingPI with Kimi-K3 MoonViT preprocessing."""

from transformers.feature_extraction_utils import BatchFeature

from .media_utils import MediaInput
from .image_processing_groundingpi import GroundingPIImageProcessor


class GroundingPIProcessor:
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        del kwargs
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template or getattr(tokenizer, "chat_template", None)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        cls._auto_class = auto_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        import os
        from transformers import AutoTokenizer

        kwargs.pop("_from_auto", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        with open(os.path.join(pretrained_model_name_or_path, "preprocessor_config.json"), encoding="utf-8") as f:
            processor_config = json.load(f)
        image_processor = GroundingPIImageProcessor(
            media_proc_cfg=processor_config["media_proc_cfg"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **kwargs
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def apply_chat_template(self, messages, **kwargs):
        if self.chat_template and "chat_template" not in kwargs:
            kwargs["chat_template"] = self.chat_template
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(
        self,
        text=None,
        images=None,
        return_tensors="pt",
        padding=False,
        **kwargs,
    ):
        if isinstance(text, str):
            text = [text]
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            **kwargs,
        )
        data = dict(text_inputs)
        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                return_tensors=return_tensors,
            )
            data.update(image_inputs)
        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["GroundingPIProcessor"]
