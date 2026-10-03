"""Optimizer group helpers for scale-oriented OpenWAM training."""

from __future__ import annotations


def build_trainable_parameters(model, *, action_lr=None, video_lr=None, vlm_lr=None, bridge_lr=None):
    """Bucket every trainable top-level param by owner, then build optimizer groups:

    - ``video_backbone`` → optional ``video_lr``
    - ``action_backbone`` → optional ``action_lr``
    - ``vlm_backbone`` → optional ``vlm_lr``
    - ``backbone_conditioner`` → optional ``bridge_lr``
    - every other trainable top-level module (a co-trained
      ``understanding_expert``, …) → the optimizer base ``learning_rate``, never
      a per-group override.

    ``bridge_lr`` covers the Fixed-16 π-style projector + resampler + depth
    embeddings. They are randomly initialised and want the action-side LR, while
    the pretrained backbone next to them wants a much smaller one; leaving them on
    the base LR made the split depend on what ``learning_rate`` happened to be.

    ``vlm_lr`` exists so a co-trained VLM can be adapted on the same terms as a
    co-trained video DiT: without it the VLM would ride the base LR while the
    video backbone had its own knob, which is not a fair comparison. It is inert
    when the VLM is frozen (no trainable params reach the optimizer).

    Source of truth: ``BaseWAMArchitecture.get_trainable_modules()`` (top-level
    children that hold at least one ``requires_grad`` param). Only ``requires_grad``
    params are collected, so the model yaml ``freeze:`` list — which flips
    ``requires_grad`` — is the single gate for trainability; frozen params never
    reach the optimizer. With no per-group LR override, returns a flat param list;
    otherwise returns groups.
    """
    action_params, video_params, vlm_params, bridge_params, other_params = [], [], [], [], []
    for mod_name, mod in model.architecture.get_trainable_modules().items():
        if mod_name == "video_backbone":
            bucket = video_params
        elif mod_name == "action_backbone":
            bucket = action_params
        elif mod_name == "vlm_backbone":
            bucket = vlm_params
        elif mod_name == "backbone_conditioner":
            bucket = bridge_params
        else:
            bucket = other_params
        bucket.extend(param for param in mod.parameters() if param.requires_grad)

    if action_lr is None and video_lr is None and vlm_lr is None and bridge_lr is None:
        return action_params + video_params + vlm_params + bridge_params + other_params

    # action / video / vlm / bridge may carry their own LR; "other" always rides
    # the base LR (lr=None).
    groups = []
    for params, lr in [
        (action_params, action_lr),
        (video_params, video_lr),
        (vlm_params, vlm_lr),
        (bridge_params, bridge_lr),
        (other_params, None),
    ]:
        if not params:
            continue
        group = {"params": params}
        if lr is not None:
            group["lr"] = lr
        groups.append(group)
    return groups


__all__ = ["build_trainable_parameters"]
