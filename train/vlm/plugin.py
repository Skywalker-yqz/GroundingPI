"""SWIFT registration for GroundingPI with the Kimi-K3 MoonViT3D tower."""

import os
from typing import Any, Dict, Literal

import torch

from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.model_arch import MultiModelKeys, register_model_arch
from swift.template import register_template
from swift.template.templates.qwen import Qwen2VLTemplate, QwenTemplateMeta
from swift.template.template_inputs import StdTemplateInputs


# Avoid exhausting the node-wide file table in long-lived DataLoader workers.
torch.multiprocessing.set_sharing_strategy("file_system")


def _install_model_only_scheduler_handoff() -> None:
    """Restore only the remapped LR scheduler for a cross-world-size handoff.

    Rigid ZeRO-1 optimizer partitions cannot be loaded after a DP world-size
    change. SWIFT's resume_only_model intentionally skips optimizer, scheduler
    and RNG. This option restores the
    world-size-independent LambdaLR state and synchronize optimizer-group LRs.
    The behavior is disabled unless the launcher sets the explicit guard env.
    """

    if os.environ.get("GAM_OXYGEN_MODEL_ONLY_SCHEDULER_HANDOFF") != "1":
        return

    from pathlib import Path

    # This pinned Swift vendor exposes ``SwiftMixin`` (the trainer MRO mixin),
    # not the upstream/newer ``TrainerMixin`` name.  Patch the class that owns
    # the common Trainer/Seq2SeqTrainer behavior so every rank follows the same
    # handoff path.
    from swift.trainers.mixin import SwiftMixin
    from swift.utils import get_logger
    from transformers.trainer import Trainer as HfTrainer

    expected_step = int(os.environ["GAM_OXYGEN_HANDOFF_STEP"])
    original = HfTrainer._load_optimizer_and_scheduler
    logger = get_logger()

    def _load_optimizer_and_scheduler(self, checkpoint, *args, **kwargs):
        if not self.args.resume_only_model:
            return original(self, checkpoint, *args, **kwargs)
        checkpoint_path = Path(checkpoint).resolve(strict=True)
        scheduler_path = checkpoint_path / "scheduler.pt"
        state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
        if int(state.get("last_epoch", -1)) != expected_step:
            raise RuntimeError(
                f"VLM scheduler restore: step={state.get('last_epoch')} expected={expected_step}"
            )
        self.lr_scheduler.load_state_dict(state)
        resumed_lrs = list(self.lr_scheduler.get_last_lr())
        param_groups = self.lr_scheduler.optimizer.param_groups
        if len(param_groups) != len(resumed_lrs):
            raise RuntimeError(
                f"VLM scheduler LR groups={len(resumed_lrs)} optimizer groups={len(param_groups)}"
            )
        for group, lr in zip(param_groups, resumed_lrs):
            group["lr"] = float(lr)
        logger.info(
            "VLM model-only resume loaded scheduler at "
            f"step={expected_step}, lrs={resumed_lrs}"
        )

    SwiftMixin._load_optimizer_and_scheduler = _load_optimizer_and_scheduler


_install_model_only_scheduler_handoff()


MODEL_TYPE = "groundingpi"
TEMPLATE_TYPE = "groundingpi"
MODEL_PATH = "weights/vlm"


class GroundingPITemplate(Qwen2VLTemplate):
    version = "v2"

    def replace_tag(
        self,
        media_type: Literal["image", "video", "audio"],
        index: int,
        inputs: StdTemplateInputs,
    ):
        if media_type != "image":
            raise ValueError("GroundingPI GAM training supports image and text data only")
        from qwen_vl_utils import fetch_image

        inputs.images[index] = fetch_image(
            {"image": inputs.images[index]},
            image_patch_size=self.processor.image_processor.patch_size,
        )
        return ["<|vision_start|><|image_pad|><|vision_end|>"]

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def _get_position_ids(self, inputs: Dict[str, Any]):
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[-1]
        # GroundingPI's language backbone is plain Qwen3, whose rotary embedding
        # expects [batch, sequence].  Qwen2-VL templates normally return a
        # three-axis mRoPE tensor; retaining that extra axis makes Qwen3 create
        # a 5-D cosine tensor and fails at the first attention layer.
        position_ids = torch.arange(seq_len, device=input_ids.device).view(1, seq_len)
        return {"position_ids": position_ids}

    def packing_row(self, row):
        """Pack plain-Qwen3 text positions without Qwen2-VL's 3-D mRoPE.

        ``Qwen2VLTemplate.packing_row`` expects its position tensor to be 3-D.
        GroundingPI uses a plain Qwen3 language tower and intentionally produces a
        2-D ``[1, seq]`` tensor; passing that tensor through the generic list
        merge triggers an ambiguous tensor truth-value check.  Flatten each
        sample to a Python list before the base packer concatenates positions.
        This is equivalent to Qwen3's monotonically increasing text positions
        and preserves the original packed sample boundaries.
        """

        from swift.template.base import Template

        for item in row:
            item_copy = item.copy()
            input_ids = item_copy["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            if input_ids.ndim == 1:
                input_ids = input_ids[None]
            item_copy["input_ids"] = input_ids
            position_ids = self._get_position_ids(item_copy)["position_ids"]
            item["position_ids"] = position_ids.reshape(-1).tolist()
        return Template.packing_row(self, row)

    def _data_collator(self, batch, *, padding_to=None):
        result = super()._data_collator(batch, padding_to=padding_to)
        text_position_ids = result.pop("text_position_ids", None)
        if text_position_ids is not None:
            # Qwen2-VL splits the text axis from mRoPE.  GroundingPI forwards that
            # axis into plain Qwen3, whose rotary embedding requires
            # ``[batch, sequence]`` rather than a 1-D sequence tensor.
            result["position_ids"] = (
                text_position_ids.unsqueeze(0)
                if text_position_ids.ndim == 1
                else text_position_ids
            )
        elif result.get("position_ids") is not None and result["position_ids"].shape[0] == 0:
            result.pop("position_ids")
        return result


register_model_arch(
    MultiModelKeys(
        MODEL_TYPE,
        language_model=["model.language_model", "lm_head"],
        aligner="model.visual.projector",
        vision_tower="model.visual.vision_tower",
    ),
    exist_ok=True,
)

register_template(
    QwenTemplateMeta(
        TEMPLATE_TYPE,
        template_cls=GroundingPITemplate,
        default_system=None,
    ),
    exist_ok=True,
)

register_model(
    ModelMeta(
        MODEL_TYPE,
        [
            ModelGroup(
                [Model(model_path=MODEL_PATH)],
                template=TEMPLATE_TYPE,
                tags=["vision", "kimi-k3-vit", "groundingpi"],
            )
        ],
        template=TEMPLATE_TYPE,
        model_arch=MODEL_TYPE,
        architectures=["GroundingPIForConditionalGeneration"],
        is_multimodal=True,
        additional_saved_files=[
            "configuration_groundingpi.py",
            "modeling_groundingpi.py",
            "configuration_groundingpi_vision.py",
            "modeling_groundingpi_vision.py",
            "processing_groundingpi.py",
            "image_processing_groundingpi.py",
            "media_utils.py",
            "streammind_gate.py",
            "preprocessor_config.json",
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
            "groundingpi_build_manifest.json",
        ],
    ),
    exist_ok=True,
)
