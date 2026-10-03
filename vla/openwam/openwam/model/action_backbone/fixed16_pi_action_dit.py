"""Fixed-16 π-style Layerwise Action DiT — the unified Action Expert.

A single Action Expert shared verbatim by every backbone under evaluation::

    Action input X_0 = [E_s(s); Q_1..32; E_a(a_t, t)]
      ↓ Block  0:  CrossAttn(C_0) → FFN
      ↓ Block  1:  SelfAttn       → FFN
      ↓ ...
      ↓ Block 14:  CrossAttn(C_7) → FFN
      ↓ Block 15:  SelfAttn       → FFN
      ↓ decode only the action-token positions

16 **atomic** blocks — 8 layer-wise cross-attention blocks interleaved with 8
action self-attention blocks and 16 FFNs. This is deliberately *not* OpenWAM's
:class:`~openwam.model.action_backbone.separate_action_dit.CrossAttnActionDiTBlock`,
whose paired layout runs ``self + video-cross + context-cross + FFN`` per block
and whose depth/width follow the video backbone. Differences that matter:

============================  ==========================  ======================
dimension                     native OpenWAM ActionDiT     this module
============================  ==========================  ======================
block type                    paired                       atomic
attention per block           self + 2 cross               cross **or** self
depth                         follows video DiT            fixed 16
backbone taps                 follows action depth         fixed 8
attention inner width         follows video head geometry  fixed 1024
raw text/proprio cross-attn   present                      **removed**
timestep conditioning         9-way AdaLN gating           AdaLN before attention
position encoding             1-D RoPE                     learned embedding
============================  ==========================  ======================

Removing the raw-context branch is the point, not an optimization: language must
reach control only through the evaluated backbone's hidden states, otherwise the
comparison measures the bypass rather than the representation. Proprioceptive
state enters as a state token in the action sequence, never through the video
backbone's text context.

The module owns its flow matching end to end (Beta(1.5, 1.0) timestep sampling,
``a_t = (1-t)ε + t·a``, ``v* = a - ε``, 10 Euler steps) so that every backbone
trains and samples on an identical schedule. Note this is the *opposite sign
convention* from the joint-denoising OpenWAM action scheduler (``eps - a``),
which predicts ``ε - a`` along a shifted-sigmoid σ grid; the two are equivalent
under ``σ = 1 - t`` but must not be mixed.

Effective parameter count is ``242,704,384 + 2049·a + 1024·s`` — 242.73M for a
7-D action/state interface, 242.95M for OpenWAM's 80-D unified action space.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from openwam.model.action_backbone.base import ActionDiTBackbone
from openwam.model.action_backbone.components import (
    ActionEncoder,
    TimestepEmbedding,
    get_attention_fn,
)

#: Pinned Action Expert geometry — identical for every backbone by design, which
#: is why none of it is derived from a backbone property.
NUM_ACTION_BLOCKS = 16
NUM_CROSS_BLOCKS = 8
ACTION_DIM_HIDDEN = 1024
NUM_HEADS = 16
HEAD_DIM = 64
FFN_DIM = 4096
MAX_SEQ_LEN = 1024
NUM_PLANNING_TOKENS = 32
FREQ_DIM = 256


class MLP(nn.Module):
    """Two-layer ``Linear → ReLU → Linear`` projector.

    Used for the state encoder and the action decoder. Deliberately without a
    LayerNorm: the published budget is ``1024·s + 1,050,624`` for the state
    encoder, which a normalization layer's affine parameters would break, and
    keeping the exact module shape makes the head bit-comparable with the
    reference implementation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = ACTION_DIM_HIDDEN, output_dim: int = ACTION_DIM_HIDDEN):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class AdaLayerNorm(nn.Module):
    """Timestep-conditioned shift/scale applied **before** attention.

    ``temb → SiLU → Linear(d, 2d) → (shift, scale)``, over a parameter-free
    LayerNorm. There is no output gate: the residual add is unmodulated. This is
    the atomic-block modulation, not OpenWAM's 9-way ``self/cross/ffn`` gating,
    which belongs to a different block definition.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(self.silu(temb)).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class Attention(nn.Module):
    """Single attention op with a fixed 1024-wide Q/K/V space.

    Both the self and the cross variant read K/V at ``kv_dim == dim == 1024``:
    conditions are projected to the Action Expert's width by the backbone
    conditioner before they get here, so the K/V projections never change shape
    with the backbone. No Q/K RMSNorm and no RoPE — the published per-block
    budget is ``4d² + 4d`` for attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        kv_dim: Optional[int] = None,
    ):
        super().__init__()
        kv_dim = dim if kv_dim is None else kv_dim
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(kv_dim, inner_dim)
        self.to_v = nn.Linear(kv_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, S_q, dim)`` query sequence.
            kv: ``(B, S_kv, kv_dim)`` key/value sequence; ``None`` for self-attention.
            kv_padding_mask: optional ``(B, S_kv)`` bool, True = attend. Only used
                when the conditions carry the backbone's unpooled sequence, where
                the padded tail must not receive attention mass.
        """
        kv = x if kv is None else kv
        q = rearrange(self.to_q(x), "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(self.to_k(kv), "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(self.to_v(kv), "b s (n d) -> b n s d", n=self.num_heads)

        if kv_padding_mask is None:
            # Unchanged fast path: the flash/SDPA dispatcher, no mask.
            out = get_attention_fn()(q, k, v)
        else:
            if kv_padding_mask.shape != kv.shape[:2]:
                raise ValueError(
                    f"kv_padding_mask must be {tuple(kv.shape[:2])}, got {tuple(kv_padding_mask.shape)}"
                )
            # (B, S_kv) -> (B, 1, 1, S_kv), broadcast over heads and queries.
            attn_mask = kv_padding_mask.to(torch.bool)[:, None, None, :]
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.to_out(rearrange(out, "b n s d -> b s (n d)", n=self.num_heads))


class AtomicBlock(nn.Module):
    """One atomic block: ``AdaLN → (Cross|Self)Attn → +res``, then ``LN → FFN → +res``.

    Even blocks cross-attend to a backbone condition; odd blocks self-attend over
    the action sequence. Both cost the same ``14d² + 11d`` parameters because the
    condition has already been projected to the action width.
    """

    def __init__(
        self,
        dim: int = ACTION_DIM_HIDDEN,
        ffn_dim: int = FFN_DIM,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
        is_cross: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.is_cross = bool(is_cross)
        self.norm1 = AdaLayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads, head_dim, kv_dim=dim)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

    @property
    def ff_linears(self) -> list:
        """The FFN's two projections, by identity rather than by index.

        Used by the interpolated initialiser and the topology summary so neither
        hard-codes ``ff[0]`` / ``ff[2]``. An index that silently lands on a
        non-Linear would make the initialiser skip a projection without saying so.
        """
        return [m for m in self.ff if isinstance(m, nn.Linear)]

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        condition_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_cross and condition is None:
            raise ValueError("A cross-attention atomic block requires a condition.")
        kv = condition if self.is_cross else None
        # The mask describes the condition sequence, so it is meaningless on a
        # self-attention block, whose K/V is the action sequence itself.
        kv_padding_mask = condition_padding_mask if self.is_cross else None
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states, temb), kv, kv_padding_mask=kv_padding_mask
        )
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        return hidden_states


class Fixed16PiActionDiT(ActionDiTBackbone):
    """The unified Action Expert. Identical for every backbone under evaluation.

    Args:
        action_dim: raw action width (80 for OpenWAM's unified action space).
        state_dim: proprioceptive state width; 0 disables the state token.
        action_horizon: number of action steps predicted per chunk.
        num_inference_steps: Euler steps at sampling time (protocol default 10).
        noise_beta_alpha / noise_beta_beta / noise_s: timestep sampling law.
        num_timestep_buckets: discretization for the timestep encoder.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int = 0,
        action_horizon: int = 32,
        num_inference_steps: int = 10,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        num_planning_tokens: int = NUM_PLANNING_TOKENS,
        max_seq_len: int = MAX_SEQ_LEN,
        dim: int = ACTION_DIM_HIDDEN,
        ffn_dim: int = FFN_DIM,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
        num_blocks: int = NUM_ACTION_BLOCKS,
        compute_fp32: bool = False,
        position_embedding_scope: str = "sequence",
    ):
        super().__init__()
        if num_blocks % 2 != 0:
            raise ValueError(f"num_blocks must be even so Cross/Self alternate evenly, got {num_blocks}")
        if num_heads * head_dim != dim:
            raise ValueError(f"Attention inner width must equal the residual width: {num_heads}x{head_dim} != {dim}.")

        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim or 0)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.num_timestep_buckets = int(num_timestep_buckets)
        self.noise_s = float(noise_s)
        self.num_planning_tokens = int(num_planning_tokens)
        self.max_seq_len = int(max_seq_len)
        self.dim = int(dim)
        self._num_heads = int(num_heads)
        self._head_dim = int(head_dim)
        self._num_layers = int(num_blocks)
        self.num_cross_blocks = self._num_layers // 2
        #: Run this module's math in fp32 whatever dtype its parameters are.
        #: See :meth:`predict_velocity`.
        self.compute_fp32 = bool(compute_fp32)
        if position_embedding_scope not in ("sequence", "action"):
            raise ValueError(
                f"position_embedding_scope must be 'sequence' or 'action', got {position_embedding_scope!r}."
            )
        #: "sequence" positions the whole ``[state; planning; action]`` stack;
        #: "action" positions the action chunk alone (0..T-1), as starVLA does,
        #: leaving state and planning unpositioned — they need none, being a
        #: single token and a set of already-distinct learned tokens.
        self.position_embedding_scope = str(position_embedding_scope)

        self.beta_dist = torch.distributions.Beta(float(noise_beta_alpha), float(noise_beta_beta))

        self.timestep_encoder = TimestepEmbedding(FREQ_DIM, dim)
        self.action_encoder = ActionEncoder(self.action_dim, dim)
        self.state_encoder = MLP(self.state_dim, dim, dim) if self.state_dim else None
        self.action_decoder = MLP(dim, dim, self.action_dim)

        self.future_tokens = nn.Embedding(self.num_planning_tokens, dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        self.position_embedding = nn.Embedding(self.max_seq_len, dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Even blocks cross-attend, odd blocks self-attend, first block is Cross.
        self.blocks = nn.ModuleList(
            [
                AtomicBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    head_dim,
                    is_cross=(i % 2 == 0),
                )
                for i in range(self._num_layers)
            ]
        )

    # ------------------------------------------------------------------
    # ActionDiTBackbone interface
    # ------------------------------------------------------------------

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def uses_proprioception(self) -> bool:
        """State enters as an action-sequence token, not through the text context."""
        return self.state_encoder is not None

    def topology_summary(self) -> str:
        """The line the implementation checklist wants in the logs."""
        return (
            f"{self._num_layers} blocks = {self.num_cross_blocks} cross + "
            f"{self._num_layers - self.num_cross_blocks} self | d={self.dim}, "
            f"{self._num_heads}x{self._head_dim} heads, ffn={self.blocks[0].ff_linears[0].out_features} | "
            f"{self.num_planning_tokens} planning tokens, horizon={self.action_horizon} | "
            f"pos_embed={self.position_embedding_scope} fp32={self.compute_fp32}"
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _validate_conditions(self, conditions: Sequence[torch.Tensor]) -> None:
        if len(conditions) != self.num_cross_blocks:
            raise ValueError(
                f"Expected {self.num_cross_blocks} conditions (one per cross block), got {len(conditions)}."
            )
        for j, c in enumerate(conditions):
            if c.ndim != 3 or c.shape[-1] != self.dim:
                raise ValueError(
                    f"Condition {j} must be (B, tokens, {self.dim}), got {tuple(c.shape)}. "
                    "Project backbone hidden states through the BackboneConditioner first."
                )

    def predict_velocity(
        self,
        noisy_actions: torch.Tensor,
        timestep_buckets: torch.Tensor,
        conditions: Sequence[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        condition_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the network, in fp32 when ``compute_fp32`` is set.

        DeepSpeed casts the whole engine module to bf16 at ``prepare()`` and gives
        no per-module escape, so the Action Expert cannot simply be held in fp32 --
        anything cast before ``prepare()`` is flattened back. What works, and what
        starVLA does, is ``autocast(float32)``: it casts at each op boundary, so the
        bf16 parameters are upcast for every matmul *and* the bf16 activations
        arriving from the backbone are upcast with them. No dtype boundary is left
        to reconcile by hand, and the casts sit inside the graph, so gradients land
        back on the bf16 parameters with ZeRO-2 and its fp32 master weights
        untouched.

        Verified on this machine (torch 2.7.1, bf16 parameters): the result is
        bit-identical to a true fp32 matmul and differs from bf16 by 3.3e-3, so the
        arithmetic really is fp32 rather than a bf16 product widened afterwards.

        Costs autocast's fp32 weight cache (~0.97 GB for this head) plus fp32
        activations, and fp32 matmuls over 243M parameters.

        This does **not** rescue a saturated softmax. The 6.7e6 logit gap measured
        on block 0 of the diverged Wan run underflows to probability exactly 1.0 in
        fp32 as it does in bf16, leaving the gradient exactly zero. fp32 widens the
        window before that point; it does not remove it.
        """
        # CPU autocast has no fp32 mode and needs none: nothing downcasts there.
        if not self.compute_fp32 or noisy_actions.device.type != "cuda":
            return self._forward_impl(noisy_actions, timestep_buckets, conditions, state, condition_padding_mask)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            return self._forward_impl(noisy_actions, timestep_buckets, conditions, state, condition_padding_mask)

    def _forward_impl(
        self,
        noisy_actions: torch.Tensor,
        timestep_buckets: torch.Tensor,
        conditions: Sequence[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        condition_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Network forward: noisy action chunk → predicted flow velocity.

        Args:
            noisy_actions: ``(B, T, action_dim)``.
            timestep_buckets: ``(B,)`` long, the discretized flow timestep.
            conditions: 8 tensors of ``(B, 64, dim)`` from the conditioner.
            state: optional ``(B, 1, state_dim)`` proprioceptive state.
        Returns:
            ``(B, T, action_dim)`` velocity at the action positions only.
        """
        self._validate_conditions(conditions)
        batch_size, horizon, _ = noisy_actions.shape

        action_features = self.action_encoder(noisy_actions, timestep_buckets)

        if self.position_embedding_scope == "action":
            pos_ids = torch.arange(action_features.shape[1], device=action_features.device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0).to(
                action_features.dtype
            )

        planning = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).to(action_features.dtype)
        parts = [planning, action_features]
        if self.state_encoder is not None:
            if state is None:
                raise ValueError("state_dim > 0 but no proprioceptive state was provided.")
            state_features = self.state_encoder(state.to(action_features.dtype))
            parts.insert(0, state_features)
        x = torch.cat(parts, dim=1)
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f"Action sequence length {x.shape[1]} exceeds max_seq_len {self.max_seq_len}.")

        # Over the whole `[state; planning; action]` sequence, per report §11:
        #   x = x + action_position_embedding[: x.shape[1]]
        # Adding it to the action tokens alone would leave the state and planning
        # tokens unpositioned and shift the action tokens' indices down to 0 — same
        # parameter count, different positions, so it will not show up in any
        # budget check.
        if self.position_embedding_scope == "sequence":
            pos_ids = torch.arange(x.shape[1], device=x.device)
            x = x + self.position_embedding(pos_ids).unsqueeze(0).to(x.dtype)

        temb = self.timestep_encoder(timestep_buckets.to(action_features.dtype))

        for i, block in enumerate(self.blocks):
            # Cross block i reads condition i // 2, so the 8 cross blocks consume
            # conditions 0..7 in order. Indexing conditions[i] instead would skip
            # every other condition and leave half the taps without gradient.
            condition = conditions[i // 2] if block.is_cross else None
            x = block(x, temb, condition, condition_padding_mask=condition_padding_mask)

        return self.action_decoder(x[:, -horizon:])

    # ------------------------------------------------------------------
    # Flow matching
    # ------------------------------------------------------------------

    def sample_time(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Beta-distributed flow timestep in ``[0, 1]``; ``t=1`` is clean data."""
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def to_buckets(self, t: torch.Tensor) -> torch.Tensor:
        return (t * self.num_timestep_buckets).long().clamp(0, self.num_timestep_buckets - 1)

    def sample_flow_batch(self, actions: torch.Tensor):
        """Draw one flow-matching training example per sample.

        Returns ``(noisy_actions, velocity_target, timestep_buckets)`` with
        ``a_t = (1-t)ε + t·a`` and ``v* = a - ε``. Split out from
        :meth:`flow_matching_loss` so the architecture can run the network via
        its own ``forward`` (keeping module hooks live) and still use exactly
        this schedule.
        """
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], actions.device, actions.dtype)[:, None, None]
        noisy_actions = (1 - t) * noise + t * actions
        velocity_target = actions - noise
        return noisy_actions, velocity_target, self.to_buckets(t[:, 0, 0])

    @staticmethod
    def masked_mse(
        predicted: torch.Tensor, target: torch.Tensor, action_is_pad: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``||v̂ - v*||²``, averaged over valid action cells only.

        ``action_is_pad`` is an OpenWAM addition the reference protocol does not
        need: under ``unify_action=true`` the action vector is an 80-D unified
        space in which only the embodiment's own slots are meaningful, so
        unmasked MSE would train the head to regress velocity on padding.
        """
        per_element = (predicted.float() - target.float()) ** 2
        if action_is_pad is None:
            return per_element.mean()
        valid = (~action_is_pad.to(device=per_element.device, dtype=torch.bool)).to(per_element.dtype)
        if valid.shape != per_element.shape:
            if valid.ndim != 2 or valid.shape != per_element.shape[:2]:
                raise ValueError(
                    f"action_is_pad must be {tuple(per_element.shape)} or {tuple(per_element.shape[:2])}, "
                    f"got {tuple(valid.shape)}"
                )
            valid = valid.unsqueeze(-1).expand_as(per_element)
        return (per_element * valid).sum() / valid.sum().clamp(min=1.0)

    def flow_matching_loss(
        self,
        conditions: Sequence[torch.Tensor],
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
        condition_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``L_FM = ||v̂ - (a - ε)||²`` — the self-contained reference path."""
        noisy_actions, velocity_target, buckets = self.sample_flow_batch(actions)
        predicted = self.predict_velocity(
            noisy_actions, buckets, conditions, state, condition_padding_mask=condition_padding_mask
        )
        return self.masked_mse(predicted, velocity_target, action_is_pad)

    @torch.no_grad()
    def predict_action(
        self,
        conditions: Sequence[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        active_action_mask: Optional[torch.Tensor] = None,
        condition_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler-integrate the flow from noise to actions, reusing cached conditions.

        The backbone is not touched here: all ``num_inference_steps`` steps read
        the same ``conditions``, so one observation costs exactly one backbone
        forward.

        ``active_action_mask`` — ``(action_dim,)`` or broadcastable — names the
        dimensions this embodiment actually uses. Under the unified 80-D action
        space most dimensions are padding, and the training loss masks them out,
        so the head never learns what velocity to predict there. Left free, those
        untrained dimensions drift on every Euler step and flow back into the
        supervised ones through the action self-attention — a feedback loop that
        gets worse with more steps. Holding them at their initial noise keeps the
        sampler on the subspace the loss actually covers.
        """
        self._validate_conditions(conditions)
        reference = conditions[0]
        batch_size = reference.shape[0] if batch_size is None else batch_size
        actions = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )

        keep = None
        if active_action_mask is not None:
            keep = active_action_mask.to(device=reference.device, dtype=torch.bool)
            if keep.numel() != self.action_dim:
                raise ValueError(
                    f"active_action_mask must cover action_dim={self.action_dim}, got {keep.numel()} entries."
                )
            keep = keep.reshape(1, 1, self.action_dim)

        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = step * dt
            buckets = torch.full(
                (batch_size,),
                int(t * self.num_timestep_buckets),
                device=reference.device,
                dtype=torch.long,
            )
            velocity = self.predict_velocity(
                actions, buckets, conditions, state, condition_padding_mask=condition_padding_mask
            )
            if keep is not None:
                velocity = velocity * keep
            actions = actions + dt * velocity
        return actions


    def forward(self, *args, **kwargs) -> torch.Tensor:  # type: ignore[override]
        """Alias for :meth:`predict_velocity` (the ABC declares ``forward`` abstract)."""
        return self.predict_velocity(*args, **kwargs)


__all__ = [
    "ACTION_DIM_HIDDEN",
    "FFN_DIM",
    "HEAD_DIM",
    "NUM_ACTION_BLOCKS",
    "NUM_CROSS_BLOCKS",
    "NUM_HEADS",
    "NUM_PLANNING_TOKENS",
    "AdaLayerNorm",
    "AtomicBlock",
    "Fixed16PiActionDiT",
    "MLP",
]
