"""Shared machinery for HuggingFace-backed VLM backbones.

Every VLM under evaluation reaches the Action Expert through the same three
operations — build processor inputs, batch them, run the language stack and read
hidden states — and those are identical across model families. What actually
differs is only:

- **which HF class loads the checkpoint** (:meth:`load_backbone`), and
- **which submodule to call for the forward** (:meth:`run_backbone`), because
  some families expose the language stack as ``model.model`` and others want the
  top-level ``forward``.

Two more hooks exist for the awkward families: :meth:`format_prompt` (chat
template vs. a bare prompt string) and :attr:`EXTRA_TENSOR_KEYS` (per-family
processor outputs that must be concatenated when collating).

Subclasses that override nothing else get correct batching, padding, dtype/device
placement, ``extract_features`` and layer-wise extraction for free — which is the
point: the comparison is only meaningful if every backbone is driven identically.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from openwam.model.vlm_backbone.base import VlmBackbone

logger = logging.getLogger(__name__)


class HFVlmBackbone(VlmBackbone):
    """A frozen-or-cotrained HuggingFace VLM presented through the OpenWAM contract."""

    #: Processor outputs beyond ``input_ids`` / ``attention_mask`` that are
    #: carried through when collating a list of samples. Anything not listed is
    #: dropped, so a key the model's forward needs must appear here or the batch
    #: silently loses it. ``mm_token_type_ids`` is what transformers 5.x emits
    #: for the Qwen family; PaliGemma adds ``token_type_ids`` in its subclass.
    EXTRA_TENSOR_KEYS: Tuple[str, ...] = (
        "pixel_values",
        "image_grid_thw",
        "image_grid_hws",  # LocateAnything's MoonViT; required by its forward
        "pixel_values_videos",
        "video_grid_thw",
        "mm_token_type_ids",
    )

    #: Processor outputs that must take the model dtype (everything else only
    #: moves device).
    PIXEL_KEYS: Tuple[str, ...] = ("pixel_values", "pixel_values_videos")

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype = torch.bfloat16,
        load_pretrained: bool = True,
        max_length: int = 512,
    ):
        super().__init__()
        self._max_length = max_length
        self.dtype = dtype
        self._checkpoint_path = checkpoint_path
        self.vlm_model, self.processor = self.load_backbone(
            checkpoint_path, dtype=dtype, load_pretrained=load_pretrained
        )
        if checkpoint_path:
            self._repair_added_token_ids(checkpoint_path)

    # ------------------------------------------------------------------
    # Tokenizer repair
    # ------------------------------------------------------------------

    @staticmethod
    def _declared_added_token_ids(checkpoint_path: str) -> dict:
        """``{content: id}`` from ``tokenizer_config.json``'s ``added_tokens_decoder``.

        This is the checkpoint's own, explicit statement of where its added tokens
        live. Empty dict when the file or the field is absent.
        """
        import json

        try:
            from transformers.utils import cached_file

            path = cached_file(
                checkpoint_path,
                "tokenizer_config.json",
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        except Exception:  # offline, private repo, unusual layout — not fatal
            path = None
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                decoder = json.load(fh).get("added_tokens_decoder")
        except (OSError, ValueError):
            return {}
        if not isinstance(decoder, dict):
            return {}
        declared = {}
        for token_id, entry in decoder.items():
            content = entry.get("content") if isinstance(entry, dict) else None
            if content is not None:
                declared[str(content)] = int(token_id)
        return declared

    def _repair_added_token_ids(self, checkpoint_path: str) -> None:
        """Force the tokenizer onto the ids the checkpoint declares.

        A checkpoint may **re-purpose** existing vocabulary entries as new tokens
        rather than extend the vocabulary — Rex-Omni turns the last 1000 Qwen BPE
        fragments into the coordinate tokens ``<0>``…``<999>`` at ids 150643-151642,
        which is why its 151665-token vocabulary still fits 151936 embedding rows.
        Upstream records that in ``added_tokens_decoder`` and, per the author,
        expects it also written into ``tokenizer.json`` with explicit ids.

        The published Rex-Omni repo ships **no** ``tokenizer.json``. Without it the
        fast tokenizer is converted from ``vocab.json`` + ``merges.txt``, sees those
        ids already occupied, and *appends* instead — shifting every added token by
        +1000, past the end of the embedding. The failure surfaces as an async
        ``indexSelectLargeIndex`` device assert whose traceback points at whatever
        kernel ran next, so it is worth catching here instead.

        The slow tokenizer builds its table straight from ``added_tokens_decoder``
        and is correct, so that is the repair. No-op when the ids already agree,
        which is every other backbone.
        """
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "get_added_vocab"):
            return
        declared = self._declared_added_token_ids(checkpoint_path)
        if not declared:
            return
        runtime = tokenizer.get_added_vocab()
        mismatched = {c: i for c, i in declared.items() if runtime.get(c) != i}
        if not mismatched:
            return

        sample = sorted(mismatched.items())[:3]
        detail = ", ".join(f"{c!r} declared {i} but got {runtime.get(c)}" for c, i in sample)
        from transformers import AutoTokenizer

        try:
            slow = AutoTokenizer.from_pretrained(
                checkpoint_path,
                use_fast=False,
                trust_remote_code=getattr(self, "_allow_remote_code", False),
            )
        except Exception as e:
            raise ValueError(
                f"{type(self).__name__}: tokenizer assigned ids that contradict "
                f"{checkpoint_path}'s added_tokens_decoder ({len(mismatched)} tokens: {detail}), "
                f"and the slow tokenizer could not be built to repair it ({type(e).__name__}: {e})."
            ) from e

        slow_runtime = slow.get_added_vocab()
        still_wrong = {c: i for c, i in declared.items() if slow_runtime.get(c) != i}
        if still_wrong:
            raise ValueError(
                f"{type(self).__name__}: neither the fast nor the slow tokenizer honours "
                f"{checkpoint_path}'s added_tokens_decoder ({len(still_wrong)} tokens still wrong). "
                "Refusing to run: the model would be fed token ids the checkpoint was not trained on."
            )

        self.processor.tokenizer = slow
        # Some processors cache token ids at construction; recompute any that the
        # swap invalidated, or the image placeholders stop matching.
        for attr in [a for a in dir(type(self.processor)) if a.endswith("_token_id")] + [
            a for a in vars(self.processor) if a.endswith("_token_id")
        ]:
            token = getattr(self.processor, attr[: -len("_id")], None)
            if isinstance(token, str):
                try:
                    setattr(self.processor, attr, slow.convert_tokens_to_ids(token))
                except AttributeError:  # read-only property that derives from the tokenizer
                    pass
        logger.warning(
            "%s: fast tokenizer contradicted %s's added_tokens_decoder (%d tokens: %s); "
            "switched to the slow tokenizer, which honours the declared ids.",
            type(self).__name__,
            checkpoint_path,
            len(mismatched),
            detail,
        )

    # ------------------------------------------------------------------
    # Family hooks
    # ------------------------------------------------------------------

    def load_backbone(self, checkpoint_path: str, *, dtype: torch.dtype, load_pretrained: bool):
        """Return ``(model, processor)``. Subclasses must implement."""
        raise NotImplementedError(f"{type(self).__name__} must implement load_backbone().")

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Trade recompute for activation memory on the co-trained VLM.

        Without this the whole backbone's activations are retained for backward
        while the video path — which honours the same flag — keeps only block
        boundaries. That asymmetry does not change any number, but it does cap
        the batch size the VLM side can reach, and the cap is set by the
        heaviest backbone since the protocol requires one batch size for all.

        No-ops when the model is frozen (nothing to back-propagate through) or
        when the family does not expose HuggingFace's toggle.
        """
        if enabled == getattr(self, "_gradient_checkpointing", False):
            return
        model = self.vlm_model
        if enabled and not any(p.requires_grad for p in model.parameters()):
            return
        enable = getattr(model, "gradient_checkpointing_enable", None)
        disable = getattr(model, "gradient_checkpointing_disable", None)
        if enable is None or disable is None:
            logger.warning(
                "%s: %s exposes no gradient_checkpointing_enable(); activations stay resident.",
                type(self).__name__,
                type(model).__name__,
            )
            return
        try:
            if enabled:
                # use_reentrant=False keeps checkpointing compatible with the
                # hidden-state outputs this path reads.
                enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                disable()
        except Exception as e:  # a remote-code family may reject the kwarg
            logger.warning(
                "%s: could not toggle gradient checkpointing (%s: %s).", type(self).__name__, type(e).__name__, e
            )
            return
        self._gradient_checkpointing = enabled

    def run_backbone(self, model_inputs: dict, *, output_hidden_states: bool):
        """Run the language stack and return an object with ``.last_hidden_state``
        (and ``.hidden_states`` when requested).

        Default targets ``self.vlm_model.model`` — the inner model, skipping the
        LM head, which is what the Qwen family wants. Families whose top-level
        ``forward`` is the right entry point override this.
        """
        return self._call_module(
            self.vlm_model.model,
            model_inputs,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    @staticmethod
    def _call_module(module, model_inputs: dict, **control_kwargs):
        """Call ``module`` with only the control kwargs its forward accepts.

        Model families disagree about which of ``past_key_values`` / ``use_cache``
        / ``return_dict`` their forward takes, especially the ones shipping their
        own ``modeling_*.py``. Filtering against the real signature turns a hard
        ``TypeError`` on an unfamiliar family into a working call, without
        silently dropping the *inputs* (those are always passed through).
        """
        import inspect

        try:
            params = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):  # C-implemented or otherwise opaque
            return module(**model_inputs, **control_kwargs)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            accepted = control_kwargs
        else:
            accepted = {k: v for k, v in control_kwargs.items() if k in params}
            dropped = set(control_kwargs) - set(accepted)
            if dropped:
                logger.debug("%s.forward does not accept %s; omitting.", type(module).__name__, sorted(dropped))
        return module(**model_inputs, **accepted)

    def format_prompt(self, prompt: str) -> str:
        """Wrap the instruction the way this family's processor expects.

        Default is the single-image chat template shared by the Qwen family.
        Families without a chat template (PaliGemma) return the prompt unchanged.
        """
        if self.processor is None:
            return prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def text_config(self):
        """The language-stack config.

        Some families nest it under ``text_config``, others keep
        ``hidden_size`` / ``num_hidden_layers`` at the top level; both are read
        through here so the geometry properties never have to care.
        """
        config = self.vlm_model.config
        return getattr(config, "text_config", config)

    @property
    def hidden_size(self) -> int:
        return int(self.text_config().hidden_size)

    @property
    def num_layers(self) -> int:
        """Number of language-model transformer blocks (embeddings excluded)."""
        return int(self.text_config().num_hidden_layers)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.vlm_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def get_submodule(self, name: str) -> Optional[nn.Module]:
        return getattr(self, name, None)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _pad_token_id(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        config_pad = getattr(getattr(self.vlm_model, "config", None), "pad_token_id", None)
        if config_pad is not None:
            return int(config_pad)
        raise ValueError(
            f"{type(self).__name__}: cannot determine pad_token_id from processor.tokenizer "
            "or model.config. Pass a processor whose tokenizer has pad_token_id set, "
            "or set pad_token_id on the model config."
        )

    def prepare_vlm_inputs(self, prompts: list[str], images: list[Any]) -> dict[str, torch.Tensor]:
        """Build processor inputs from OpenWAM prompts + observation images.

        Array-like outputs are converted rather than dropped. Some processors
        return grid metadata the model's forward *requires* as a numpy array
        (LocateAnything's ``image_grid_hws``); filtering on
        ``isinstance(v, torch.Tensor)`` alone would silently discard it and the
        model would fail deep inside its vision tower on a ``None``.
        """
        if self.processor is None:
            raise RuntimeError(f"{type(self).__name__}.prepare_vlm_inputs requires a loaded processor.")
        texts = [self.format_prompt(prompt) for prompt in prompts]
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        out: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                out[key] = value
                continue
            tensor = self._as_tensor(value)
            if tensor is not None:
                out[key] = tensor
        return out

    @staticmethod
    def _as_tensor(value) -> Optional[torch.Tensor]:
        """Best-effort conversion of a processor output to a tensor, else None.

        Deliberately conservative: anything that is not cleanly a numeric array
        (strings, nested ragged lists, processor metadata objects) is left out,
        because passing it to the model's forward would be worse than omitting
        it.
        """
        if value is None or isinstance(value, (str, bytes)):
            return None
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return None if value.dtype == object else torch.from_numpy(value)
            if isinstance(value, (list, tuple)):
                arr = np.asarray(value)
                return None if arr.dtype == object else torch.from_numpy(arr)
        except Exception:  # noqa: BLE001 - conversion is opportunistic
            return None
        return None

    def batch_vlm_inputs(self, vlm_inputs: dict | list[dict]) -> dict[str, torch.Tensor]:
        if isinstance(vlm_inputs, dict):
            # Only tensor entries are forwarded to the VLM; non-tensor metadata (e.g.
            # processor attributes, image grids stored as lists) is dropped. The list
            # input path below is stricter — it validates required tensors explicitly.
            return {
                key: value for key, value in vlm_inputs.items() if isinstance(value, torch.Tensor) and value is not None
            }

        if not isinstance(vlm_inputs, list) or not vlm_inputs:
            raise ValueError("vlm_inputs must be a non-empty dict or list of dicts.")

        input_ids_list = [item["input_ids"] for item in vlm_inputs]
        attention_mask_list = [
            item.get("attention_mask", torch.ones_like(item["input_ids"], dtype=torch.long)) for item in vlm_inputs
        ]
        for idx, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            if ids.ndim != 2:
                raise ValueError(f"vlm_inputs[{idx}]['input_ids'] must be 2D [B, L], got shape {tuple(ids.shape)}")
            if mask.shape != ids.shape:
                raise ValueError(
                    f"vlm_inputs[{idx}]['attention_mask'] must match input_ids shape "
                    f"{tuple(ids.shape)}, got {tuple(mask.shape)}"
                )
        max_seq_len = max(ids.shape[1] for ids in input_ids_list)
        pad_token_id = self._pad_token_id()

        padded_ids = []
        padded_masks = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad = max_seq_len - ids.shape[1]
            if pad > 0:
                padded_ids.append(F.pad(ids, (0, pad), value=pad_token_id))
                padded_masks.append(F.pad(mask, (0, pad), value=0))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)

        batched = {
            "input_ids": torch.cat(padded_ids, dim=0),
            "attention_mask": torch.cat(padded_masks, dim=0),
        }
        for key in self.EXTRA_TENSOR_KEYS:
            values = [item.get(key) for item in vlm_inputs]
            if any(value is not None for value in values):
                if not all(value is not None for value in values):
                    raise ValueError(f"Mixed missing/non-missing {key} in vlm_inputs list.")
                # Token-aligned extras are padded like input_ids; everything else
                # (flattened image patches, grids) concatenates on the batch axis.
                if values[0].ndim == 2 and values[0].shape[1] == input_ids_list[0].shape[1]:
                    values = [
                        F.pad(v, (0, max_seq_len - v.shape[1])) if v.shape[1] < max_seq_len else v for v in values
                    ]
                batched[key] = torch.cat(values, dim=0)
        return batched

    def _to_model_inputs(self, vlm_inputs: dict | list[dict]) -> dict[str, torch.Tensor]:
        """Collate, then move to device — pixel tensors also take the model dtype.

        Shared by :meth:`extract_features` and
        :meth:`extract_layerwise_features` so the two cannot drift.
        """
        batch = self.batch_vlm_inputs(vlm_inputs)
        device = self.device
        model_inputs = {}
        for key, value in batch.items():
            if key in self.PIXEL_KEYS:
                value = value.to(device=device, dtype=self.dtype)
            else:
                value = value.to(device=device)
            model_inputs[key] = value
        return model_inputs

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, vlm_inputs: dict | list[dict]) -> torch.Tensor:
        """``vlm_inputs`` -> ``(B, L, hidden_size)`` last hidden state.

        **Alignment assumption** shared by every family here: the processor
        expands image placeholders in ``input_ids`` before the model forward, so
        ``hidden_states.shape[1] == input_ids.shape[1] == attention_mask.shape[1]``.
        That is what lets a consumer derive its token mask straight from
        ``vlm_inputs["attention_mask"]``.

        Freeze/no_grad policy is owned centrally by
        ``BaseWAMArchitecture.freeze_modules``; this method does no freeze
        inspection of its own.
        """
        outputs = self.run_backbone(self._to_model_inputs(vlm_inputs), output_hidden_states=False)
        return outputs.last_hidden_state

    def extract_layerwise_features(
        self, vlm_inputs: dict | list[dict], block_indices: Sequence[int]
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Return only the requested transformer blocks' hidden states.

        Caveat worth knowing when reading results: HuggingFace applies the final
        norm to ``hidden_states[-1]`` but not to the intermediate entries, so the
        deepest tap is normalized while the others are raw block outputs — an
        asymmetry the video backbones do not have. The conditioner's own
        parameter-free ``depth_norm`` largely washes it out, so this is
        documented rather than special-cased.
        """
        model_inputs = self._to_model_inputs(vlm_inputs)
        n = self.num_layers
        wanted = sorted({int(i) for i in block_indices})
        if wanted and not (0 <= wanted[0] and wanted[-1] < n):
            raise ValueError(f"block_indices must lie in [0, {n - 1}], got {wanted}")

        outputs = self.run_backbone(model_inputs, output_hidden_states=True)
        hidden = outputs.hidden_states
        if len(hidden) != n + 1:
            raise RuntimeError(
                f"{type(self).__name__} returned {len(hidden)} hidden states, expected {n + 1} "
                f"(embedding output + {n} blocks). The tap-to-block mapping is no longer valid."
            )
        # hidden[0] is the embedding output, which is not a transformer block,
        # so block i is hidden[i + 1]. Getting this offset wrong would shift
        # every tap by one layer without failing anything.
        taps = {i: hidden[i + 1] for i in wanted}
        return taps, model_inputs["attention_mask"].to(torch.bool)

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Copy the VLM checkpoint into the deploy dir so deploy is self-contained
        (no dependency on the training-time checkpoint_path). Idempotent. Invoked by
        the architecture's save_assets_for_deployment, same hook as video backbones."""
        if not self._checkpoint_path:
            return
        dest = os.path.join(output_dir, "vlm_backbone")
        if os.path.exists(dest):
            return
        shutil.copytree(self._checkpoint_path, dest)
        logger.info("Copied VLM checkpoint to %s", dest)


__all__ = ["HFVlmBackbone"]
