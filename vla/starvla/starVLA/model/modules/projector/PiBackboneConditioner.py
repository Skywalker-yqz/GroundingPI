"""Shared fixed-shape conditioner for StarVLA's pi-style frameworks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def normalized_depth_indices(num_layers: int, num_taps: int = 8) -> list[int]:
    if num_taps < 2:
        raise ValueError(f"num_taps must be >= 2, got {num_taps}")
    if num_layers < num_taps:
        raise ValueError(f"need at least {num_taps} backbone layers, got {num_layers}")
    return [round(index * (num_layers - 1) / (num_taps - 1)) for index in range(num_taps)]


class TokenResampler(nn.Module):
    """One shared cross-attention layer that produces fixed learned queries."""

    def __init__(
        self,
        dim: int = 1024,
        num_queries: int = 64,
        num_heads: int = 16,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def _split_heads(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        return hidden.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = hidden.shape[0]
        queries = self.queries.to(dtype=hidden.dtype).expand(batch, -1, -1)
        attention_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != hidden.shape[:2]:
                raise ValueError(
                    f"key_padding_mask must be {tuple(hidden.shape[:2])}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )
            # SDPA bool masks use True for positions that participate in attention.
            attention_mask = key_padding_mask.to(torch.bool)[:, None, None, :]
        output = F.scaled_dot_product_attention(
            self._split_heads(self.to_q(queries)),
            self._split_heads(self.to_k(hidden)),
            self._split_heads(self.to_v(hidden)),
            attn_mask=attention_mask,
        )
        output = output.transpose(1, 2).reshape(batch, queries.shape[1], -1)
        return self.norm_out(queries + self.to_out(output))


class PiBackboneConditioner(nn.Module):
    """Map eight backbone taps to eight ``(B, 64, 1024)`` conditions.

    The normalization, projector, and resampler are each instantiated once and
    shared across depth.  Depth identity is injected before resampling, matching
    the fixed-16 pi reference topology.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        tokens_per_tap: int = 64,
        num_taps: int = 8,
        num_heads: int = 16,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_taps = int(num_taps)
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.depth_embedding = nn.Parameter(torch.empty(num_taps, hidden_dim))
        self.resampler = TokenResampler(
            dim=hidden_dim,
            num_queries=tokens_per_tap,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        nn.init.normal_(self.depth_embedding, std=0.02)

    def forward(
        self,
        taps: list[torch.Tensor],
        key_padding_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if len(taps) != self.num_taps:
            raise ValueError(f"expected {self.num_taps} taps, got {len(taps)}")
        conditions = []
        for depth, tap in enumerate(taps):
            if tap.ndim != 3 or tap.shape[-1] != self.input_dim:
                raise ValueError(
                    f"tap {depth} must be (B, S, {self.input_dim}), got {tuple(tap.shape)}"
                )
            projected = self.input_proj(self.input_norm(tap))
            projected = projected + self.depth_embedding[depth].to(dtype=projected.dtype)
            conditions.append(
                self.resampler(projected, key_padding_mask=key_padding_mask)
            )
        return conditions


__all__ = ["PiBackboneConditioner", "TokenResampler", "normalized_depth_indices"]
