"""Initialise the Action Expert by resampling a pretrained video-DiT's blocks.

The Action Expert is 16 blocks of 1024 with a 4096 FFN; Wan2.2-TI2V-5B is 30
blocks of 3072 with a 14336 FFN. Every tensor therefore has to be resampled on
**both** axes:

- **depth**, 30 → 16, with the same normalized-depth rule the conditioner uses
  for its taps (``ℓ_j = round(j·(N_B−1)/(K−1))``), so the initialisation reads
  the backbone at the same spread of depths the conditions are drawn from;
- **width**, via area interpolation over the weight matrix, which averages each
  3×3 (or 3×3.5 for the FFN) source block into one target entry.

Averaging shrinks the spread: the mean of ``k`` roughly independent weights has
``1/√k`` the standard deviation, and a narrower layer needs a *wider* spread to
hold its output variance (``fan_in`` fell by 3×). Left alone the Action Expert
would start with activations orders of magnitude too small and spend the early
steps just re-growing them. Each resampled tensor is therefore rescaled to the
standard deviation the target's own default initialisation would have had, which
keeps the structure the interpolation captured and the scale the architecture
expects.

**This makes the Action Expert's starting point depend on the source model, not
on the run's backbone.** Point every run at the *same* source or the comparison
is over: the whole protocol rests on the Action Expert being byte-identical
across backbones, and per-backbone initialisation would put a different model on
each row of the table with nothing in the logs to say so. Using Wan's weights
for a Qwen3-VL run is intentional and fine; using each run's own backbone is not.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

from openwam.model.action_backbone.backbone_conditioner import normalized_depth_indices

logger = logging.getLogger(__name__)

#: Source block → target block attribute, per Action Expert block kind. Cross
#: blocks take the video DiT's cross-attention, self blocks its self-attention,
#: so each target reads the source projection that plays the same role.
_ATTENTION_SOURCE = {True: "cross_attn", False: "self_attn"}
_ATTENTION_PAIRS = (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "to_out"))
#: FFN projections are addressed through ``AtomicBlock.ff_linears`` rather than by
#: index into ``block.ff``: that Sequential gains and loses Dropout entries with
#: the config, and an index that silently lands on a Dropout would make this
#: initialiser skip the projection without reporting anything.
_FFN_PAIRS = (("ffn.0", 0), ("ffn.2", 1))


def _resample(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Area-resample ``source`` onto ``target_shape`` (1-D or 2-D)."""
    if source.shape == target_shape:
        return source.clone()
    src = source.detach().float()
    if src.dim() == 1:
        out = F.interpolate(src[None, None], size=(int(target_shape[0]),), mode="area")
        return out[0, 0]
    if src.dim() == 2:
        out = F.interpolate(src[None, None], size=tuple(int(s) for s in target_shape), mode="area")
        return out[0, 0]
    raise ValueError(f"Cannot resample a {src.dim()}-D tensor.")


def _rescale_to(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Match ``tensor``'s standard deviation to ``reference``'s, preserving shape.

    ``reference`` is the target parameter as the module built it, so this restores
    the scale the architecture was designed around after interpolation flattened
    it. A near-constant source (a bias that is all zeros, say) is passed through
    rather than amplified by a huge factor.
    """
    src_std = float(tensor.std())
    ref_std = float(reference.detach().float().std())
    if src_std < 1e-8 or ref_std < 1e-8:
        return tensor
    return tensor * (ref_std / src_std)


def _load_wan_block_tensors(model_path: str) -> dict:
    """Read every ``blocks.*`` tensor out of a Wan checkpoint directory."""
    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(model_path, "diffusion_pytorch_model-*.safetensors")))
    if not shards:
        shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards under {model_path}; cannot read the source weights.")
    tensors = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                if key.startswith("blocks."):
                    tensors[key] = handle.get_tensor(key)
    if not tensors:
        raise ValueError(f"{model_path} has no `blocks.*` tensors; is this a Wan DiT checkpoint?")
    return tensors


def interpolate_init_action_expert(action_backbone, source_path: str, *, rescale: bool = True) -> dict:
    """Resample a Wan DiT's blocks into ``action_backbone``'s 16 atomic blocks.

    Returns a summary dict (blocks mapped, tensors copied, tensors left random).
    Modules with no counterpart keep their own initialisation and are reported:

    - ``norm1`` (AdaLayerNorm) — the source modulates with a bare ``(1, 6, dim)``
      parameter rather than a timestep-driven linear, so there is nothing to map;
    - ``norm3`` — parameter-free here;
    - the timestep / action / state encoders and the decoder, which have no
      counterpart in a video DiT at all.
    """
    src = _load_wan_block_tensors(source_path)
    n_source = 1 + max(int(k.split(".")[1]) for k in src)
    n_target = len(action_backbone.blocks)
    depth_map = normalized_depth_indices(n_source, num_taps=n_target)

    copied, skipped = 0, []
    with torch.no_grad():
        for tgt_idx, src_idx in enumerate(depth_map):
            block = action_backbone.blocks[tgt_idx]
            attn_src = _ATTENTION_SOURCE[bool(block.is_cross)]
            ff_linears = block.ff_linears
            targets = [(f"{attn_src}.{a}", getattr(block.attn, b)) for a, b in _ATTENTION_PAIRS]
            targets += [(src, ff_linears[idx]) for src, idx in _FFN_PAIRS]
            for src_suffix, tgt_mod in targets:
                for kind in ("weight", "bias"):
                    src_key = f"blocks.{src_idx}.{src_suffix}.{kind}"
                    tgt_param = getattr(tgt_mod, kind, None)
                    if src_key not in src or tgt_param is None:
                        continue
                    value = _resample(src[src_key], tgt_param.shape)
                    if rescale:
                        value = _rescale_to(value, tgt_param)
                    tgt_param.copy_(value.to(dtype=tgt_param.dtype, device=tgt_param.device))
                    copied += 1
            skipped.append(f"block{tgt_idx}.norm1")

    summary = {
        "source": source_path,
        "source_blocks": n_source,
        "target_blocks": n_target,
        "depth_map": depth_map,
        "tensors_copied": copied,
        "left_random": ["norm1 (AdaLN)", "timestep/action/state encoders", "action decoder", "planning/pos embeddings"],
        "rescaled_to_target_std": rescale,
    }
    logger.warning(
        "[Fixed-16 π-style] Action Expert initialised from %s: %d blocks -> %d via %s, %d tensors resampled%s. "
        "Every backbone in a comparison must use this same source, or their Action Experts no longer match.",
        source_path,
        n_source,
        n_target,
        depth_map,
        copied,
        " (rescaled to the target's own init std)" if rescale else "",
    )
    return summary


def maybe_interpolate_init(action_backbone, cfg, _cfg_get) -> Optional[dict]:
    """Apply :func:`interpolate_init_action_expert` when the config asks for it."""
    source = _cfg_get(cfg, "action_expert_init_source", None)
    if not source:
        return None
    rescale = bool(_cfg_get(cfg, "action_expert_init_rescale", True))
    return interpolate_init_action_expert(action_backbone, str(source), rescale=rescale)


__all__ = ["interpolate_init_action_expert", "maybe_interpolate_init"]
