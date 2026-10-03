"""Runtime action-normalization scheme patches for LeRobot mixtures.

DataConfigs ship a default (typically mean_std + chunk normalizer, including
relative rotations). Call ``apply_norm_scheme_to_mixture`` *after*
``LeRobotMixtureDataset`` construction (so merged metadata is already applied)
to switch schemes without editing every DataConfig.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform

if TYPE_CHECKING:
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset

logger = logging.getLogger(__name__)

# Default / identity: leave DataConfig modes untouched.
DEFAULT_ACTION_NORM_SCHEME = "mean_std"
VALID_ACTION_NORM_SCHEMES = ("mean_std", "ori_nonorm_q99")


def is_ori6d_key(key: str) -> bool:
    """True for orientation / rotation-6d action (or state) keys."""
    kl = key.lower()
    if "orientation_6d" in kl:
        return True
    if kl.endswith("_rotation") or kl.endswith("rotation_6d"):
        return True
    if "ee_pose_6d_rotation" in kl or "rotation_6d" in kl:
        return True
    return False


def _iter_state_action_transforms(transform) -> Iterable[StateActionTransform]:
    if isinstance(transform, StateActionTransform):
        yield transform
        return
    for child in getattr(transform, "transforms", []) or []:
        yield from _iter_state_action_transforms(child)


def _validate_scheme(scheme: str) -> str:
    scheme = str(scheme or DEFAULT_ACTION_NORM_SCHEME).lower().strip()
    if scheme not in VALID_ACTION_NORM_SCHEMES:
        raise ValueError(
            f"Unsupported action_norm_scheme={scheme!r}; "
            f"expected one of {VALID_ACTION_NORM_SCHEMES}"
        )
    return scheme


def _patch_transform(
    modality_transform,
    scheme: str,
    *,
    metadata=None,
) -> int:
    if scheme == "mean_std":
        return 0

    n = 0
    for tr in _iter_state_action_transforms(modality_transform):
        modes = getattr(tr, "normalization_modes", None)
        if not modes:
            continue
        new = dict(modes)
        ucn = list(getattr(tr, "use_chunk_normalizer", []) or [])
        clip = dict(getattr(tr, "chunk_normalizer_clip", None) or {})

        for key in list(new):
            if not key.startswith("action."):
                continue
            if is_ori6d_key(key):
                del new[key]
            elif new[key] in ("mean_std", "min_max", "q99"):
                new[key] = "q99"

        ucn = [key for key in ucn if not is_ori6d_key(key)]
        clip = {key: value for key, value in clip.items() if not is_ori6d_key(key)}

        tr.normalization_modes = new
        tr.use_chunk_normalizer = ucn
        if hasattr(tr, "chunk_normalizer_clip"):
            tr.chunk_normalizer_clip = clip
        tr._normalizers = {}
        tr._chunk_normalizers = {}
        if metadata is not None:
            tr.set_metadata(metadata)
        n += 1
    return n


def apply_norm_scheme_to_transform(modality_transform, scheme: str) -> int:
    """Apply a training action-normalization scheme to one inference transform.

    Call this before ``modality_transform.set_metadata(...)`` so q99 normalizers
    are built from the checkpoint's dataset statistics.
    """
    scheme = _validate_scheme(scheme)
    n = _patch_transform(modality_transform, scheme)
    logger.info(
        "action_norm_scheme=%s: patched %d inference StateActionTransform(s)",
        scheme,
        n,
    )
    return n


def apply_norm_scheme_to_mixture(mixture: "LeRobotMixtureDataset", scheme: str) -> int:
    """Patch action normalization on every dataset in ``mixture``.

    Schemes:
      - ``mean_std``: no-op (keep DataConfig defaults).
      - ``ori_nonorm_q99``: drop rot/ori6d from action normalization and chunk
        normalizers; set remaining action modes to ``q99``.

    Returns the number of ``StateActionTransform`` instances that were updated.
    """
    scheme = _validate_scheme(scheme)

    if scheme == "mean_std":
        logger.info(
            "action_norm_scheme=mean_std: keeping DataConfig normalization unchanged"
        )
        return 0

    n = 0
    for ds in mixture.datasets:
        modality_transform = getattr(ds, "transforms", None)
        metadata = getattr(ds, "dataset_metadata", None) or getattr(ds, "metadata", None)
        n += _patch_transform(modality_transform, scheme, metadata=metadata)

    logger.info(
        "action_norm_scheme=ori_nonorm_q99: patched %d StateActionTransform(s) "
        "(rot/ori6d unnormalized; other action dims q99)",
        n,
    )
    return n
