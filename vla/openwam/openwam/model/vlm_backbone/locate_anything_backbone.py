"""LocateAnything-3B backbone (MoonViT vision encoder + Qwen2.5-3B language stack).

Unlike every other backbone here, ``nvidia/LocateAnything-3B`` ships its own
``modeling_locateanything.py`` and is only loadable with
``trust_remote_code=True``.

**That is a code-execution surface**: loading the checkpoint runs Python from the
checkpoint directory, at training and at deploy. The Qwen and PaliGemma wrappers
deliberately avoid it. It is opt-in here because the model cannot be loaded any
other way, and it is gated on an explicit config flag so nobody enables it by
accident — point ``checkpoint_path`` only at a directory you trust.

Two structural differences from the Qwen family:

- The language stack is reached through the model's **top-level** ``forward``,
  which propagates ``output_hidden_states`` down to ``self.language_model`` and
  returns a ``CausalLMOutputWithPast`` carrying ``hidden_states``. There is no
  ``.model`` attribute to call, so :meth:`run_backbone` is overridden.
- Its ``forward`` also drives a parallel box-decoding head that this path never
  uses; only the hidden states are read.

Geometry: 36 blocks / hidden 2048 (Qwen2.5-3B-Instruct), so the depth taps land
at 0, 5, 10, 15, 20, 25, 30, 35 — the same positions as Qwen3-VL-4B.
"""

from __future__ import annotations

import logging
import os
import sys

import torch

from openwam.model.vlm_backbone.hf_vlm_base import HFVlmBackbone


def _module_cache_suffix() -> str:
    """Per-process suffix for the dynamic-module cache.

    Keyed on LOCAL_RANK rather than RANK so that ranks on the same node get
    distinct directories -- the race is between processes sharing a filesystem,
    and a node's local ranks are exactly that set. PID is not used: the directory
    would then be recreated on every launch and never reused.
    """
    return "_rank" + str(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


logger = logging.getLogger(__name__)


class LocateAnythingBackbone(HFVlmBackbone):
    """LocateAnything through the OpenWAM VLM contract.

    Args:
        allow_remote_code: must be True. Exists so enabling remote code is a
            deliberate, greppable act in the config rather than a silent default.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype = torch.bfloat16,
        load_pretrained: bool = True,
        max_length: int = 512,
        allow_remote_code: bool = False,
    ):
        if not allow_remote_code:
            raise ValueError(
                "LocateAnythingBackbone loads custom modeling code from the checkpoint directory "
                "(trust_remote_code=True). Set `allow_remote_code: true` in the vlm_backbone config "
                "to acknowledge that, and only point checkpoint_path at a directory you trust."
            )
        self._allow_remote_code = True
        super().__init__(
            checkpoint_path=checkpoint_path,
            dtype=dtype,
            load_pretrained=load_pretrained,
            max_length=max_length,
        )

    @staticmethod
    def _isolate_dynamic_module_cache() -> None:
        """Give this process its own HF dynamic-module cache directory.

        ``trust_remote_code`` loading copies the checkpoint's ``.py`` files into
        ``~/.cache/huggingface/modules/transformers_modules/<name>/`` and then
        imports them. The copy is not atomic and the destination is shared, so
        when several ranks reach it at the same instant they interleave writes to
        the same file. Multi-node training does exactly that -- 8 ranks per node
        with a cold cache -- and it fails as a truncated source file::

            File ".../processing_locateanything.py", line 233, in <module>
                VIDEO_READER_BACKENDS = {
            NameError: name 'VIDEO_REA' is not defined

        The identifier is cut mid-word: the file on disk was half-written when
        another rank imported it. Which rank dies and where the truncation lands
        vary between launches, which is what a write race looks like. Single-node
        runs miss it only because the cache is usually already warm.

        Pointing each process at its own directory removes the shared
        destination. The copies are a few hundred KB and land in the container's
        own filesystem, so the cost is negligible next to loading the weights.

        Only the *dynamic module* cache moves. HF_HOME and the weight cache stay
        where they are -- those are read-mostly and shared on purpose.
        """
        cache_root = os.environ.get("HF_MODULES_CACHE")
        if cache_root and cache_root.endswith(_module_cache_suffix()):
            return  # already isolated by an outer call
        base = cache_root or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "modules"
        )
        isolated = base.rstrip("/") + _module_cache_suffix()
        os.makedirs(isolated, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = isolated
        # transformers reads HF_MODULES_CACHE at import time into a module-level
        # constant and appends it to sys.path, so setting the variable alone is
        # not enough once that import has already happened.
        import transformers.dynamic_module_utils as _dmu

        _dmu.HF_MODULES_CACHE = isolated
        if isolated not in sys.path:
            sys.path.insert(0, isolated)
        init_file = os.path.join(isolated, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w"):
                pass
        logger.info("HF dynamic-module cache isolated to %s (avoids a multi-rank write race)", isolated)

    def load_backbone(self, checkpoint_path: str, *, dtype: torch.dtype, load_pretrained: bool):
        from transformers import AutoConfig, AutoModel, AutoProcessor

        self._isolate_dynamic_module_cache()

        processor = None
        if checkpoint_path:
            try:
                processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
            except Exception:
                if load_pretrained:
                    raise
        # Pin sdpa. The checkpoint's config asks for ``magi``, a custom kernel from
        # a package that is not on PyPI; the release code silently falls back to
        # sdpa (or flash-attention-2) when it is missing, so which attention runs
        # would otherwise depend on what happens to be installed — not something a
        # backbone comparison can afford to leave to the machine.
        if load_pretrained:
            model = AutoModel.from_pretrained(
                checkpoint_path, dtype=dtype, trust_remote_code=True, attn_implementation="sdpa"
            )
        else:
            cfg = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
            model = AutoModel.from_config(cfg, trust_remote_code=True, attn_implementation="sdpa").to(dtype=dtype)
        return model, processor

    def run_backbone(self, model_inputs: dict, *, output_hidden_states: bool):
        """Top-level forward, with two repairs its release code needs.

        **1. ``position_ids`` must be supplied.** The language stack builds them
        itself when absent, but as ``arange(L).view(-1, L)`` — shape ``(1, L)``
        regardless of batch (``modeling_qwen2.py:1247-1250``). Its block mask is
        then built at batch 1 and the attention layer rejects it. Anything past
        batch 1 fails without this.

        **2. The mask must be built the training way.** ``modeling_qwen2.py:1332``
        picks the mask builder off ``self.training``:

        - training → ``create_block_diff_mask_by_pe_4d(position_ids, x0_len, …)``
        - eval → ``_prepare_block_mask_for_inference``, which indexes
          ``input_ids[b]`` (``:1306``)

        but the top-level forward calls ``self.language_model(inputs_embeds=…)``
        without ``input_ids`` (``modeling_locateanything.py:244``), and passing
        both is rejected outright (``modeling_qwen2.py:1218``). So the eval branch
        is unreachable in one piece — it dies on ``NoneType`` subscripting.

        **3. The prefix boundary must be signalled through ``position_ids``.** The
        training builder is a *block-diffusion* mask: everything before ``x0_len``
        is the causal prompt, everything after is denoising blocks. ``x0_len`` is
        not passed in — it is recovered as the first index where the position ids
        **drop**, and is ``-1`` when they never do
        (``mask_sdpa_utils.py:find_prefix_seq_length_by_pe``).

        Plain ``arange`` position ids never drop, so ``x0_len = -1``, and then in
        ``create_block_diff_mask_by_pe_4d`` both ``x0_flag_q`` and ``x0_flag_kv``
        are empty, which kills ``block_causal`` and ``block_prefix`` and leaves only
        ``block_mutual`` — a **block-diagonal** mask in which every token sees just
        the 6 tokens of its own block and nothing else, not even the image. It runs,
        produces finite numbers, and is meaningless.

        Restarting the position ids over the padded tail puts the drop exactly at
        each sample's real length, which is what that length means here: prompt =
        prefix, no denoising blocks. ``block_causal`` then covers the real tokens
        causally, and padding lands in the ``~x0_flag`` region that real tokens
        cannot attend to — so this branch ignoring ``attention_mask`` costs nothing.
        Junk RoPE phases on the padded tail are harmless for the same reason.

        The longest sample in a batch has no padding and therefore no drop, so one
        extra pad column is appended to give it one, and the hidden states are
        trimmed back to the caller's length before returning. The caller's
        ``model_inputs`` is left untouched, so the key-padding mask it derives still
        matches.

        **4. …but the enclosing ``Qwen2ForCausalLM`` must not be in training mode.**
        Its ``if self.training`` return (``modeling_qwen2.py:1534-1541``) hands back
        ``(CausalLMOutputWithPast, pos_loss_list)``, and ``pos_loss_list`` is only
        bound inside ``if labels is not None`` (``:1518``). Without labels it raises
        ``UnboundLocalError``; with them it returns a tuple, and the caller's very
        next line is ``outputs.logits`` (``modeling_locateanything.py:253``). Either
        way the top-level forward cannot survive its own language model being in
        training mode.

        So the two flags are driven in opposite directions for the duration of the
        call — inner ``Qwen2Model`` on, outer ``Qwen2ForCausalLM`` off — and both
        restored after. In each forward ``self.training`` gates only the branches
        above plus a ``gradient_checkpointing and self.training`` guard that is
        inert while gradient checkpointing is off. Child modules keep their own
        flags, so dropout is untouched and co-training gradients still flow.
        """
        language_model = getattr(self.vlm_model, "language_model", None)
        # (module, flag it must hold during the call)
        overrides = [
            (getattr(language_model, "model", None), True),
            (language_model, False),
        ]

        input_ids = model_inputs.get("input_ids")
        length = None
        if input_ids is not None and model_inputs.get("position_ids") is None:
            model_inputs, length = self._with_prefix_boundary(model_inputs)

        saved = [(module, module.training) for module, _ in overrides if module is not None]
        for module, flag in overrides:
            if module is not None:
                module.training = flag
        try:
            outputs = self._call_module(
                self.vlm_model,
                model_inputs,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        finally:
            for module, was in saved:
                module.training = was

        if length is not None:
            if getattr(outputs, "hidden_states", None) is not None:
                outputs.hidden_states = tuple(h[:, :length] for h in outputs.hidden_states)
            for key in ("last_hidden_state", "logits"):
                value = getattr(outputs, key, None)
                if value is not None:
                    setattr(outputs, key, value[:, :length])
        return outputs

    def _with_prefix_boundary(self, model_inputs: dict) -> tuple[dict, int]:
        """Append one pad column and restart position ids over each padded tail.

        Returns the extended inputs and the caller's original sequence length, so
        the outputs can be trimmed back. See :meth:`run_backbone` for why the
        position-id drop is what tells the block mask where the prompt ends.
        """
        input_ids = model_inputs["input_ids"]
        batch, length = input_ids.shape
        device = input_ids.device

        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            real_lengths = torch.full((batch, 1), length, dtype=torch.long, device=device)
        else:
            real_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        # A zero-length row would leave the position ids monotonic and hand the
        # mask builder -1 again; one real token is the minimum that can drop.
        real_lengths = real_lengths.clamp(min=1, max=length)

        try:
            filler = self._pad_token_id()
        except ValueError:
            # The appended column sits in the non-prefix region no real token can
            # attend to, and is trimmed off the outputs, so its identity never
            # reaches anything. Not worth refusing to run over.
            filler = 0

        extended = dict(model_inputs)
        pad_column = torch.full((batch, 1), filler, dtype=input_ids.dtype, device=device)
        extended["input_ids"] = torch.cat([input_ids, pad_column], dim=1)
        if attention_mask is not None:
            extended["attention_mask"] = torch.cat(
                [attention_mask, torch.zeros((batch, 1), dtype=attention_mask.dtype, device=device)], dim=1
            )
        positions = torch.arange(length + 1, device=device).unsqueeze(0)
        extended["position_ids"] = torch.where(positions < real_lengths, positions, positions - real_lengths)
        return extended, length


__all__ = ["LocateAnythingBackbone"]
