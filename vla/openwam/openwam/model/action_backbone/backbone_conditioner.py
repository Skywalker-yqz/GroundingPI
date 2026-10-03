"""Backbone conditioner for the Fixed-16 π-style Layerwise Action DiT.

Turns a backbone's per-block hidden states into the fixed-shape conditions the
Action Expert cross-attends to, so every video-generation (or VLM) backbone
presents an identical interface to an identical Action Expert::

    H_0 .. H_{N_B-1}                      one full backbone forward
      → pick 8 normalized-depth taps      ℓ_j = round(j · (N_B - 1) / 7)
      → Z_j = LN(H_ℓj) W_B + e_j          shared LN + shared projector + depth emb
      → C_j = Resampler_64(Z_j)           shared resampler
      → 8 × (B, 64, 1024)

Invariants the fair-comparison protocol depends on:

- ``W_B`` is created **once** and shared across the 8 depths — one projector per
  depth is explicitly disallowed.
- The resampler is likewise created **once** and shared across the 8 depths.
- The LayerNorm is parameter-free. The published projector budget is
  ``P_W_B = 1024·D_B + 1024``, i.e. exactly ``nn.Linear(D_B, 1024)`` with no
  LayerNorm affine term.
- Condition width is pinned to 1024 and token count to 64 for every backbone, so
  the Action Expert's cross-attention K/V geometry never varies.

``W_B`` is backbone-specific (its input width is ``D_B``), which is why this
module lives outside the Action Expert and is reported separately from the
242.7M effective Action Head.

Unlike the VLM case, OpenWAM's video backbones may hand back a 5-D
``(B, T, H, W, D)`` grid (CosmosPredict25); :func:`flatten_backbone_hidden`
normalizes that to the ``(B, L, D)`` token layout before anything else runs.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from openwam.model.action_backbone.components import get_attention_fn

logger = logging.getLogger(__name__)

CONDITION_DIM = 1024
NUM_DEPTH_TAPS = 8
TOKENS_PER_TAP = 64

SAMPLING_MODES = ("normalized_depth", "last_hidden", "final_8", "bin_average")

#: How the tapped hidden states are normalized before the shared projector.
#:
#: ``parameter_free`` is report §2.1's ``LN(H_ℓj)·W_B``, and the only mode whose
#: projector budget matches §8's ``1024·D_B + 1024`` — a LayerNorm with affine
#: weights would not fit that number. It is the default.
#:
#: It also erases the backbone's feature *scale*: a tap with std 1.0 and one with
#: std 0.01 produce the same output, so the action loss cannot tell them apart and
#: nothing holds the scale in place. LayerNorm's backward carries a ``1/σ`` factor,
#: so as the scale drifts down the gradient reaching the backbone grows — a loop
#: that has no counter-force once ``lambda_video=0`` removes the video loss that
#: normally anchors those features.
#:
#: ``affine`` keeps the normalization but gives the scale a learnable degree of
#: freedom (``+2·D_B`` parameters, so report the projector count accordingly).
#: ``none`` drops the norm entirely, which is what starVLA's WanPI does — it
#: projects the raw hidden states, so a shrinking scale degrades the action loss
#: directly and is pushed back on.
DEPTH_NORM_MODES = ("parameter_free", "affine", "none")


def normalized_depth_indices(num_backbone_blocks: int, num_taps: int = NUM_DEPTH_TAPS) -> List[int]:
    """Uniformly spaced tap indices ``ℓ_j = round(j · (N_B - 1) / (num_taps - 1))``.

    Reproduces the published depth table exactly: ``N_B=28`` →
    ``[0, 4, 8, 12, 15, 19, 23, 27]``, ``N_B=30`` → ``[0, 4, 8, 12, 17, 21, 25, 29]``,
    ``N_B=40`` → ``[0, 6, 11, 17, 22, 28, 33, 39]``.

    Index 0 is the output of the **first** transformer block; an embedding output
    is not counted as a block.
    """
    if num_taps < 2:
        raise ValueError(f"num_taps must be >= 2, got {num_taps}")
    if num_backbone_blocks < num_taps:
        raise ValueError(
            f"Backbone has {num_backbone_blocks} blocks but {num_taps} depth taps were requested. "
            "The main protocol requires every backbone to have at least as many blocks as taps."
        )
    return [round(j * (num_backbone_blocks - 1) / (num_taps - 1)) for j in range(num_taps)]


def flatten_backbone_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Normalize a captured hidden state to ``(B, L, D)``.

    Wan backbones already emit ``(B, L, D)``. CosmosPredict25 lays its hidden
    state out as ``(B, T, H, W, D)``; the spatial axes are folded into a single
    token axis, matching what the native OpenWAM cross-attention bridge does before its
    bridge cross-attention.
    """
    if hidden.ndim == 5:
        b, t, h, w, d = hidden.shape
        return hidden.reshape(b, t * h * w, d)
    if hidden.ndim != 3:
        raise ValueError(f"backbone hidden state must be (B, L, D) or (B, T, H, W, D), got {tuple(hidden.shape)}")
    return hidden


def _bin_edges(num_backbone_blocks: int, num_taps: int) -> List[range]:
    """Split ``range(N_B)`` into ``num_taps`` contiguous, near-equal bins."""
    base, extra = divmod(num_backbone_blocks, num_taps)
    bins, start = [], 0
    for j in range(num_taps):
        stop = start + base + (1 if j < extra else 0)
        bins.append(range(start, stop))
        start = stop
    return bins


class TokenResampler(nn.Module):
    """Compress a variable-length token sequence to a fixed set of query tokens.

    One cross-attention layer over ``num_queries`` learned queries, plus a
    residual connection and an output LayerNorm. Deliberately no FFN — this is a
    token-count adapter, not extra Action Expert capacity.
    """

    def __init__(
        self,
        dim: int = CONDITION_DIM,
        num_queries: int = TOKENS_PER_TAP,
        num_heads: int = 16,
        head_dim: int = 64,
    ):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.dim = dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: ``(B, S, dim)`` projected backbone tokens.
            key_padding_mask: optional ``(B, S)`` bool mask, True = attend.
        Returns:
            ``(B, num_queries, dim)``.
        """
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be (B, S, dim), got {tuple(hidden_states.shape)}")
        batch_size = hidden_states.shape[0]
        queries = self.queries.to(dtype=hidden_states.dtype).expand(batch_size, -1, -1)

        q = self._split_heads(self.to_q(queries))
        k = self._split_heads(self.to_k(hidden_states))
        v = self._split_heads(self.to_v(hidden_states))

        if key_padding_mask is None:
            out = get_attention_fn()(q, k, v)
        else:
            if key_padding_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"key_padding_mask must be {tuple(hidden_states.shape[:2])}, got {tuple(key_padding_mask.shape)}"
                )
            # (B, S) -> (B, 1, 1, S), broadcast over heads and queries.
            attn_mask = key_padding_mask.to(torch.bool)[:, None, None, :]
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(1, 2).reshape(batch_size, self.num_queries, -1)
        return self.norm_out(queries + self.to_out(out))


class BackboneConditioner(nn.Module):
    """Backbone hidden states → 8 fixed-shape conditions for the Action Expert.

    Args:
        backbone_hidden_dim: ``D_B``, the backbone's residual width.
        num_backbone_blocks: ``N_B``, the number of transformer blocks.
        dim: condition width, pinned to 1024 by the protocol.
        num_depth_taps: number of conditions produced, pinned to 8.
        tokens_per_tap: condition tokens per tap, pinned to 64.
        sampling: one of :data:`SAMPLING_MODES`. ``normalized_depth`` is the main
            method; the other three back the published ablations
            (``last_hidden`` = the controlled last-hidden baseline, ``final_8`` =
            the backbone's last 8 blocks, ``bin_average`` = mean within 8
            contiguous depth bins).
        add_depth_embedding: add a learned per-depth embedding ``e_j``.
    """

    def __init__(
        self,
        backbone_hidden_dim: int,
        num_backbone_blocks: int,
        dim: int = CONDITION_DIM,
        num_depth_taps: int = NUM_DEPTH_TAPS,
        tokens_per_tap: int = TOKENS_PER_TAP,
        num_heads: int = 16,
        head_dim: int = 64,
        sampling: str = "normalized_depth",
        add_depth_embedding: bool = True,
        depth_norm_mode: str = "parameter_free",
        use_resampler: bool = True,
    ):
        super().__init__()
        if sampling not in SAMPLING_MODES:
            raise ValueError(f"Unknown sampling '{sampling}'. Choose from: {', '.join(SAMPLING_MODES)}")
        if depth_norm_mode not in DEPTH_NORM_MODES:
            raise ValueError(f"Unknown depth_norm_mode '{depth_norm_mode}'. Choose from: {', '.join(DEPTH_NORM_MODES)}")

        self.backbone_hidden_dim = int(backbone_hidden_dim)
        self.num_backbone_blocks = int(num_backbone_blocks)
        self.dim = int(dim)
        self.num_depth_taps = int(num_depth_taps)
        self.tokens_per_tap = int(tokens_per_tap)
        self.sampling = sampling
        self.add_depth_embedding = bool(add_depth_embedding)
        self.use_resampler = bool(use_resampler)
        #: Padding mask from the most recent forward. Only set when the resampler
        #: is off: the conditions then carry the backbone's own sequence, padded
        #: tail included, and the Action Expert has to be told which keys are
        #: real. Not a buffer — per-batch plumbing, never checkpoint state.
        self.last_key_padding_mask: Optional[torch.Tensor] = None

        self._bins: Optional[List[range]] = None
        if sampling == "normalized_depth":
            self.tap_indices = normalized_depth_indices(self.num_backbone_blocks, self.num_depth_taps)
        elif sampling == "last_hidden":
            self.tap_indices = [self.num_backbone_blocks - 1] * self.num_depth_taps
        elif sampling == "final_8":
            if self.num_backbone_blocks < self.num_depth_taps:
                raise ValueError(
                    f"Backbone has {self.num_backbone_blocks} blocks, cannot take the final {self.num_depth_taps}."
                )
            self.tap_indices = list(range(self.num_backbone_blocks - self.num_depth_taps, self.num_backbone_blocks))
        else:  # bin_average
            self._bins = _bin_edges(self.num_backbone_blocks, self.num_depth_taps)
            self.tap_indices = [b.stop - 1 for b in self._bins]

        # See DEPTH_NORM_MODES for what each mode costs and what it does to the
        # gradient reaching the backbone.
        self.depth_norm_mode = depth_norm_mode
        if depth_norm_mode == "none":
            self.depth_norm = nn.Identity()
        else:
            self.depth_norm = nn.LayerNorm(self.backbone_hidden_dim, elementwise_affine=(depth_norm_mode == "affine"))
        # Only a deviation *within* the report pathway is worth warning about. On
        # the starVLA pathway `none` is the specification, not a deviation.
        if depth_norm_mode != "parameter_free" and self.use_resampler:
            logger.warning(
                "[Fixed-16 π-style] depth_norm_mode=%s: off-protocol (report §2.1 specifies a "
                "parameter-free LayerNorm, and §8's projector budget 1024*D_B+1024 assumes it). "
                "Every backbone in a comparison must use the same mode.",
                depth_norm_mode,
            )
        # ONE projector, shared across all depths.
        self.projector = nn.Linear(self.backbone_hidden_dim, self.dim)
        if self.add_depth_embedding:
            self.depth_embedding = nn.Parameter(torch.zeros(self.num_depth_taps, self.dim))
            nn.init.normal_(self.depth_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter("depth_embedding", None)
        # ONE resampler, shared across all depths — or none at all, which is
        # what starVLA's WanPI does: it cross-attends over the backbone's full
        # sequence and has no resampler, no depth embedding and no depth norm.
        if self.use_resampler:
            self.resampler = TokenResampler(
                dim=self.dim,
                num_queries=self.tokens_per_tap,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        else:
            self.resampler = None

    def extra_repr(self) -> str:
        return (
            f"D_B={self.backbone_hidden_dim}, N_B={self.num_backbone_blocks}, "
            f"sampling={self.sampling}, taps={self.tap_indices}, "
            f"tokens={self.tokens_per_tap if self.use_resampler else 'backbone sequence'}, "
            f"dim={self.dim}"
        )

    # -- lightweight backbone surface -------------------------------------
    # The architecture lists this module in ``backbones`` so it is moved and
    # saved alongside the real backbones; those two hooks are all that requires.

    def set_dtype_device(self, dtype, device) -> None:
        self.to(dtype=dtype, device=device)

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """No external artifacts: the weights live in the safetensors checkpoint."""

    @property
    def required_block_indices(self) -> List[int]:
        """Backbone block ids this conditioner actually reads.

        Lets the architecture capture only what is needed instead of retaining
        every block output. ``bin_average`` needs every block, since it averages
        within each depth bin.
        """
        if self._bins is not None:
            return list(range(self.num_backbone_blocks))
        return sorted(set(self.tap_indices))

    def _select_taps(self, backbone_hidden_states) -> List[torch.Tensor]:
        if isinstance(backbone_hidden_states, dict):
            missing = [i for i in self.required_block_indices if i not in backbone_hidden_states]
            if missing:
                raise ValueError(
                    f"backbone_hidden_states is missing block ids {missing}; "
                    f"required_block_indices={self.required_block_indices}."
                )
        elif len(backbone_hidden_states) != self.num_backbone_blocks:
            raise ValueError(
                f"Expected {self.num_backbone_blocks} backbone block outputs, got "
                f"{len(backbone_hidden_states)}. To pass a subset, use a "
                "{block_id: tensor} dict covering required_block_indices."
            )
        lookup = backbone_hidden_states.__getitem__

        if self._bins is not None:
            return [
                torch.stack([flatten_backbone_hidden(lookup(i)) for i in bin_range], dim=0).mean(dim=0)
                for bin_range in self._bins
            ]
        return [flatten_backbone_hidden(lookup(i)) for i in self.tap_indices]

    def forward(
        self,
        backbone_hidden_states,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            backbone_hidden_states: either ``N_B`` tensors ordered first block →
                last block, or a ``{block_id: tensor}`` dict covering
                :attr:`required_block_indices`. Each tensor is ``(B, L, D_B)`` or
                the 5-D CosmosPredict25 grid.
            key_padding_mask: optional ``(B, L)`` bool mask over backbone tokens,
                True = attend. Consumed inside the resampler, so the Action
                Expert always sees 64 valid condition tokens.
        Returns:
            ``num_depth_taps`` tensors of ``(B, tokens_per_tap, dim)``.
        """
        taps = self._select_taps(backbone_hidden_states)
        conditions = []
        for j, hidden in enumerate(taps):
            if hidden.shape[-1] != self.backbone_hidden_dim:
                raise ValueError(f"Tap {j} has width {hidden.shape[-1]}, expected D_B={self.backbone_hidden_dim}.")
            z = self.projector(self.depth_norm(hidden))
            if self.depth_embedding is not None:
                z = z + self.depth_embedding[j].to(dtype=z.dtype)
            conditions.append(self.resampler(z, key_padding_mask=key_padding_mask) if self.resampler else z)

        if self.resampler is not None:
            # The resampler absorbed the mask; downstream sees 64 valid tokens.
            self.last_key_padding_mask = None
            return conditions
        if key_padding_mask is not None and key_padding_mask.shape[:2] != conditions[0].shape[:2]:
            raise ValueError(
                f"key_padding_mask {tuple(key_padding_mask.shape)} does not match the condition sequence "
                f"{tuple(conditions[0].shape[:2])}. An unmasked padded tail would receive attention mass."
            )
        self.last_key_padding_mask = key_padding_mask
        return conditions


__all__ = [
    "CONDITION_DIM",
    "NUM_DEPTH_TAPS",
    "SAMPLING_MODES",
    "TOKENS_PER_TAP",
    "BackboneConditioner",
    "TokenResampler",
    "flatten_backbone_hidden",
    "normalized_depth_indices",
]
