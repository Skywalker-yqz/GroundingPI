"""Checklist tests for OpenWAM's Fixed-16 π-style Layerwise Action DiT.

One test per item of the architecture report's implementation checklist, plus
the published depth table and parameter budgets. The parameter numbers are
asserted exactly, not approximately: if a refactor moves any of them the paper's
parameter table is wrong, and that should fail loudly.

Runs on CPU against the repo's ``_MockVideoBackbone`` — no real video backbone
is loaded.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from openwam.model.action_backbone.backbone_conditioner import (
    SAMPLING_MODES,
    BackboneConditioner,
    TokenResampler,
    flatten_backbone_hidden,
    normalized_depth_indices,
)
from openwam.model.action_backbone.fixed16_pi_action_dit import (
    AtomicBlock,
    Fixed16PiActionDiT,
)
from tests.test_openwam_trainer import _make_fake_loss_inputs, _MockVideoBackbone

DIM = 1024
NUM_BLOCKS = 16
NUM_CROSS = 8
TOKENS_PER_TAP = 64

# Published budgets.
ATOMIC_BLOCK = 14_691_328  # 14d^2 + 11d
ALL_BLOCKS = 235_061_248
TIMESTEP_ENCODER = 1_312_768
RESAMPLER = 4_265_984
HEAD_CONST = 242_704_384  # effective head = HEAD_CONST + 2049a + 1024s

# Mock-backbone geometry: Wan2.2-TI2V-5B block count, tiny width.
N_B, D_B = 30, 64
ACTION_DIM = STATE_DIM = 7
HORIZON = 5


def _n(module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Depth-tap table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_blocks,expected",
    [
        (28, [0, 4, 8, 12, 15, 19, 23, 27]),  # CosmosPredict25
        (30, [0, 4, 8, 12, 17, 21, 25, 29]),  # Wan2.2-TI2V-5B
        (32, [0, 4, 9, 13, 18, 22, 27, 31]),
        (36, [0, 5, 10, 15, 20, 25, 30, 35]),
        (40, [0, 6, 11, 17, 22, 28, 33, 39]),
    ],
)
def test_normalized_depth_tap_table(num_blocks, expected):
    assert normalized_depth_indices(num_blocks) == expected


def test_tap_selection_requires_enough_blocks():
    with pytest.raises(ValueError, match="at least as many blocks as taps"):
        normalized_depth_indices(7)


# ---------------------------------------------------------------------------
# Checklist 1: 16 atomic blocks = 8 cross + 8 self, first block is Cross
# ---------------------------------------------------------------------------


def test_atomic_block_topology():
    head = Fixed16PiActionDiT(action_dim=ACTION_DIM, state_dim=STATE_DIM)
    cross = [i for i, b in enumerate(head.blocks) if b.is_cross]
    selfa = [i for i, b in enumerate(head.blocks) if not b.is_cross]
    assert len(head.blocks) == NUM_BLOCKS
    assert cross == list(range(0, NUM_BLOCKS, 2))
    assert selfa == list(range(1, NUM_BLOCKS, 2))
    assert cross[0] == 0, "the first atomic block must be Cross"
    assert "16 blocks = 8 cross + 8 self" in head.topology_summary()


def test_atomic_block_parameter_budget():
    """Cross and self blocks cost the same: the condition is already 1024-wide."""
    assert _n(AtomicBlock(is_cross=True)) == ATOMIC_BLOCK
    assert _n(AtomicBlock(is_cross=False)) == ATOMIC_BLOCK
    assert 14 * DIM * DIM + 11 * DIM == ATOMIC_BLOCK


def test_ffn_dim_is_four_times_residual_width():
    assert AtomicBlock().ff[0].out_features == 4 * DIM


# ---------------------------------------------------------------------------
# Checklist 2: each cross block reads condition 0..7, in order
# ---------------------------------------------------------------------------


def test_cross_blocks_read_conditions_zero_through_seven():
    head = Fixed16PiActionDiT(action_dim=ACTION_DIM, state_dim=0)
    seen: list = []
    for block in head.blocks:
        original = block.forward

        def patched(hidden_states, temb, condition=None, _orig=original, **kwargs):
            seen.append(None if condition is None else int(condition[0, 0, 0].item()))
            return _orig(hidden_states, temb, condition)

        block.forward = patched

    conditions = [torch.full((1, TOKENS_PER_TAP, DIM), float(j)) for j in range(NUM_CROSS)]
    head.predict_velocity(torch.randn(1, HORIZON, ACTION_DIM), torch.tensor([10]), conditions)

    assert [s for s in seen if s is not None] == list(range(NUM_CROSS))
    assert [seen[i] for i in range(1, NUM_BLOCKS, 2)] == [None] * NUM_CROSS


def test_head_rejects_wrong_condition_count():
    head = Fixed16PiActionDiT(action_dim=ACTION_DIM, state_dim=0)
    with pytest.raises(ValueError, match="Expected 8 conditions"):
        head.predict_velocity(
            torch.randn(1, HORIZON, ACTION_DIM),
            torch.tensor([10]),
            [torch.randn(1, TOKENS_PER_TAP, DIM) for _ in range(6)],
        )


# ---------------------------------------------------------------------------
# Checklist 3/4/5/6/7: conditioner invariants
# ---------------------------------------------------------------------------


def test_all_eight_taps_receive_gradient():
    conditioner = BackboneConditioner(backbone_hidden_dim=D_B, num_backbone_blocks=N_B)
    hidden = [torch.randn(2, 10, D_B, requires_grad=True) for _ in range(N_B)]
    sum(c.sum() for c in conditioner(hidden)).backward()
    with_grad = [i for i, h in enumerate(hidden) if h.grad is not None]
    assert with_grad == conditioner.tap_indices
    assert len(with_grad) == NUM_CROSS


def test_projector_and_resampler_are_shared_across_depths():
    conditioner = BackboneConditioner(backbone_hidden_dim=3072, num_backbone_blocks=N_B)
    projectors = [m for m in conditioner.modules() if isinstance(m, nn.Linear) and m.in_features == 3072]
    assert len(projectors) == 1, "the D_B -> 1024 projector must be shared across the 8 depths"
    assert len([m for m in conditioner.modules() if isinstance(m, TokenResampler)]) == 1


def test_projector_budget_matches_published_formula():
    """P_W_B = 1024*D_B + 1024, i.e. the depth LayerNorm carries no parameters."""
    for d_b in (2048, 3072, 5120):
        conditioner = BackboneConditioner(backbone_hidden_dim=d_b, num_backbone_blocks=N_B)
        assert _n(conditioner.projector) + _n(conditioner.depth_norm) == 1024 * d_b + 1024
        assert _n(conditioner.resampler) == RESAMPLER
        assert conditioner.depth_embedding.numel() == NUM_CROSS * DIM


@pytest.mark.parametrize("d_b,seq_len", [(2048, 40), (3072, 391), (5120, 7)])
def test_condition_shape_is_fixed_regardless_of_backbone(d_b, seq_len):
    conditioner = BackboneConditioner(backbone_hidden_dim=d_b, num_backbone_blocks=28)
    conditions = conditioner([torch.randn(3, seq_len, d_b) for _ in range(28)])
    assert len(conditions) == NUM_CROSS
    assert all(c.shape == (3, TOKENS_PER_TAP, DIM) for c in conditions)


def test_cosmos_five_dim_grid_is_flattened():
    """CosmosPredict25 hands back (B, T, H, W, D) rather than (B, L, D)."""
    grid = torch.randn(2, 3, 4, 5, 64)
    assert flatten_backbone_hidden(grid).shape == (2, 60, 64)
    conditioner = BackboneConditioner(backbone_hidden_dim=64, num_backbone_blocks=28)
    conditions = conditioner([grid.clone() for _ in range(28)])
    assert conditions[0].shape == (2, TOKENS_PER_TAP, DIM)


def test_padding_mask_is_consumed_by_the_resampler():
    torch.manual_seed(0)
    conditioner = BackboneConditioner(backbone_hidden_dim=64, num_backbone_blocks=8).eval()
    hidden = [torch.randn(1, 12, 64) for _ in range(8)]
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[:, 6:] = False
    with torch.no_grad():
        masked = conditioner(hidden, key_padding_mask=mask)[0]
        unmasked = conditioner(hidden)[0]
        truncated = conditioner([h[:, :6] for h in hidden])[0]
    assert not torch.allclose(masked, unmasked), "mask had no effect"
    assert torch.allclose(masked, truncated, atol=1e-5)


def test_sampling_modes_select_expected_blocks():
    expected = {
        "normalized_depth": [0, 4, 8, 12, 17, 21, 25, 29],
        "last_hidden": [29] * 8,
        "final_8": [22, 23, 24, 25, 26, 27, 28, 29],
    }
    for mode, taps in expected.items():
        assert BackboneConditioner(64, 30, sampling=mode).tap_indices == taps
    binned = BackboneConditioner(64, 30, sampling="bin_average")
    assert binned.required_block_indices == list(range(30))
    assert len(binned([torch.randn(1, 4, 64) for _ in range(30)])) == NUM_CROSS
    assert "normalized_depth" in SAMPLING_MODES
    with pytest.raises(ValueError, match="Unknown sampling"):
        BackboneConditioner(64, 30, sampling="deepest_semantic_layer")


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [7, 14, 20, 80])
def test_effective_action_head_matches_published_formula(a):
    head = Fixed16PiActionDiT(action_dim=a, state_dim=a)
    assert _n(head) == HEAD_CONST + 2049 * a + 1024 * a
    assert sum(_n(b) for b in head.blocks) == ALL_BLOCKS
    assert _n(head.timestep_encoder) == TIMESTEP_ENCODER


def test_action_head_has_no_raw_context_path():
    """Instruction must reach control only through the evaluated backbone."""
    head = Fixed16PiActionDiT(action_dim=ACTION_DIM, state_dim=STATE_DIM)
    assert not [k for k, _ in head.named_modules() if "text" in k or "context" in k]


# ---------------------------------------------------------------------------
# Flow matching: v* = a - eps, the OPPOSITE sign from the native OpenWAM scheduler
# ---------------------------------------------------------------------------


def test_flow_matching_sign_convention():
    head = Fixed16PiActionDiT(action_dim=ACTION_DIM, state_dim=0)
    fixed_t = 0.3
    head.sample_time = lambda b, device, dtype: torch.full((b,), fixed_t, device=device, dtype=dtype)

    actions = torch.randn(4, HORIZON, ACTION_DIM)
    noisy, velocity, buckets = head.sample_flow_batch(actions)

    # a_t = (1-t)eps + t*a and v = a - eps together imply a_t + (1-t)v == a.
    assert torch.allclose(noisy + (1 - fixed_t) * velocity, actions, atol=1e-5)
    assert buckets.tolist() == [int(fixed_t * head.num_timestep_buckets)] * 4

    # The opposite convention (eps - a) would fail this.
    assert not torch.allclose(noisy + (1 - fixed_t) * (-velocity), actions, atol=1e-3)


def test_euler_integration_uses_num_inference_steps():
    head = Fixed16PiActionDiT(action_dim=ACTION_DIM, state_dim=0, action_horizon=HORIZON)
    calls = {"n": 0}

    def constant_velocity(noisy_actions, buckets, conditions, state=None, **kwargs):
        calls["n"] += 1
        return torch.ones_like(noisy_actions)

    head.predict_velocity = constant_velocity
    conditions = [torch.zeros(1, TOKENS_PER_TAP, DIM) for _ in range(NUM_CROSS)]
    torch.manual_seed(0)
    actions = head.predict_action(conditions)
    assert calls["n"] == head.num_inference_steps == 10
    torch.manual_seed(0)
    start = torch.randn(1, HORIZON, ACTION_DIM)
    # 4 steps of dt=1/4 with unit velocity displace the sample by exactly 1.
    assert torch.allclose(actions - start, torch.ones_like(start), atol=1e-5)


def test_masked_mse_ignores_padded_action_dims():
    predicted = torch.zeros(2, HORIZON, ACTION_DIM)
    target = torch.ones(2, HORIZON, ACTION_DIM)
    pad = torch.zeros(2, HORIZON, ACTION_DIM, dtype=torch.bool)
    pad[..., 3:] = True  # only the first 3 dims are real
    assert torch.isclose(Fixed16PiActionDiT.masked_mse(predicted, target, pad), torch.tensor(1.0))
    target[..., 3:] = 1000.0  # padding must not move the loss
    assert torch.isclose(Fixed16PiActionDiT.masked_mse(predicted, target, pad), torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Architecture level
# ---------------------------------------------------------------------------


class _DistinctBlockBackbone(_MockVideoBackbone):
    """Mock whose per-block hidden states differ and that counts block runs."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.blocks_run: list[int] = []

    def run_block(self, block_id, state):
        self.blocks_run.append(block_id)
        state.hidden_states = state.hidden_states + float(block_id + 1)
        return state


def _build_arch(**overrides):
    from openwam.model import build_architecture

    cfg = {
        "video_dim": D_B,
        "num_dit_layers": N_B,
        "action_dim": ACTION_DIM,
        "state_dim": STATE_DIM,
        "use_proprioception": True,
        "action_horizon": HORIZON,
    }
    cfg.update(overrides)
    arch = build_architecture("dual_system_fixed16_pi", cfg)
    arch.video_backbone = _DistinctBlockBackbone(dim=D_B, num_layers=N_B, num_heads=4)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


@pytest.fixture(scope="module")
def arch():
    return _build_arch()


def test_registry_resolves_the_variant():
    from types import SimpleNamespace

    from openwam.model import resolve_architecture_config

    resolved = resolve_architecture_config(
        SimpleNamespace(
            architecture={"framework": "dual_system", "variant": "fixed16_pi_layerwise"},
            action_backbone={},
        )
    )
    assert resolved.registry_name == "dual_system_fixed16_pi"


def test_architecture_wiring(arch):
    assert arch.backbone_conditioner.tap_indices == [0, 4, 8, 12, 17, 21, 25, 29]
    assert arch.action_backbone.num_layers == NUM_BLOCKS
    # The conditioner rides along for dtype/device moves and deploy saves, but is
    # listed separately so its parameters never fold into the Action Expert.
    assert "backbone_conditioner" in arch.backbones
    report = arch.parameter_report()
    assert report["action_head"] == HEAD_CONST + 2049 * ACTION_DIM + 1024 * STATE_DIM
    assert report["backbone_projector"] == 1024 * D_B + 1024 - 0  # LayerNorm is parameter-free
    assert report["token_resampler"] == RESAMPLER


def test_no_proprio_context_bypass(arch):
    """State must not be injected into the video backbone's text context."""
    assert not hasattr(arch, "proprio_encoder")
    assert arch.action_backbone.uses_proprioception


def test_training_step_runs_backbone_once_and_has_no_video_loss(arch):
    inputs = _make_fake_loss_inputs(B=2, action_dim=ACTION_DIM, T_action=HORIZON, video_dim=D_B)
    inputs["proprio"] = torch.randn(2, STATE_DIM)
    action_is_pad = torch.zeros(2, HORIZON, ACTION_DIM, dtype=torch.bool)
    action_is_pad[..., 4:] = True

    arch.video_backbone.blocks_run.clear()
    out = arch.compute_loss(
        actions=torch.randn(2, HORIZON, ACTION_DIM),
        lambda_video=0.0,
        lambda_action=1.0,
        action_is_pad=action_is_pad,
        **inputs,
    )
    assert torch.isfinite(out["loss"])
    assert out["loss_video"].item() == 0.0
    assert len(arch.video_backbone.blocks_run) == N_B, "exactly one pass over the video DiT"

    out["loss"].backward()
    assert all(p.grad is not None for p in arch.backbone_conditioner.parameters())
    arch.zero_grad(set_to_none=True)


def test_inference_runs_backbone_once_for_all_denoising_steps(arch):
    inputs = _make_fake_loss_inputs(B=1, action_dim=ACTION_DIM, video_dim=D_B)
    inputs.pop("max_timestep_boundary")
    inputs.pop("min_timestep_boundary")

    arch.video_backbone.blocks_run.clear()
    result = arch.predict_action_chunk(proprio=torch.randn(1, STATE_DIM), **inputs)

    assert len(arch.video_backbone.blocks_run) == N_B, "the backbone must not re-run per denoising step"
    assert arch.action_backbone.num_inference_steps == 10
    # Squeezed like BaseWAMArchitecture.generate: one request -> (T, action_dim).
    assert result["actions"].shape == (HORIZON, ACTION_DIM)
    assert result["video"] is None
    assert set(result["latency"]) == {
        "backbone_feature_extraction_s",
        "action_denoising_s",
        "end_to_end_s",
    }


@pytest.mark.parametrize(
    "config_horizon,supplied",
    [
        (HORIZON, HORIZON),  # dataloader already matches
        (HORIZON, HORIZON * 3),  # dataloader ships a longer window
    ],
)
def test_action_horizon_config_wins_over_the_dataloader(config_horizon, supplied):
    """Training and inference must use the configured chunk length, not whatever
    ``num_frames - 1`` happens to be."""
    trained = _build_arch(action_horizon=config_horizon)
    ab = trained.action_backbone
    seen = {}
    original = ab.predict_velocity

    def spy(noisy_actions, buckets, conditions, state=None, **kwargs):
        seen["horizon"] = noisy_actions.shape[1]
        return original(noisy_actions, buckets, conditions, state)

    ab.predict_velocity = spy
    inputs = _make_fake_loss_inputs(B=2, action_dim=ACTION_DIM, video_dim=D_B)
    inputs["proprio"] = torch.randn(2, STATE_DIM)
    trained.compute_loss(
        actions=torch.randn(2, supplied, ACTION_DIM),
        lambda_video=0.0,
        lambda_action=1.0,
        action_is_pad=torch.zeros(2, supplied, ACTION_DIM, dtype=torch.bool),
        **inputs,
    )
    ab.predict_velocity = original

    with torch.no_grad():
        generated = ab.predict_action(
            [torch.randn(1, TOKENS_PER_TAP, DIM) for _ in range(NUM_CROSS)],
            state=torch.randn(1, 1, STATE_DIM),
        )
    assert seen["horizon"] == config_horizon
    assert generated.shape[1] == config_horizon


def test_leading_action_steps_are_kept_when_clipping():
    """The observation is frame 0, so the chunk to predict is t=1..H, not the tail."""
    trained = _build_arch(action_horizon=HORIZON)
    supplied = torch.arange(HORIZON * 2, dtype=torch.float32).view(1, HORIZON * 2, 1)
    supplied = supplied.expand(1, HORIZON * 2, ACTION_DIM).contiguous()
    clipped, _ = trained._clip_to_action_horizon(supplied, None)
    assert clipped.shape[1] == HORIZON
    assert torch.equal(clipped, supplied[:, :HORIZON]), "leading steps must be kept"


def test_too_short_an_action_window_raises():
    trained = _build_arch(action_horizon=HORIZON * 2)
    inputs = _make_fake_loss_inputs(B=1, action_dim=ACTION_DIM, video_dim=D_B)
    inputs["proprio"] = torch.randn(1, STATE_DIM)
    with pytest.raises(ValueError, match="only supplied"):
        trained.compute_loss(
            actions=torch.randn(1, HORIZON, ACTION_DIM),
            lambda_video=0.0,
            lambda_action=1.0,
            **inputs,
        )


def test_action_horizon_defaults_to_the_dataloader_window():
    """Default 32 matches the shipped `num_frames: 33` (action horizon = num_frames - 1).

    ``_build_arch`` pins a small horizon for speed, so the default is checked on
    the head itself and on an architecture cfg that omits the key.
    """
    from openwam.model import build_architecture
    from openwam.model.action_backbone.fixed16_pi_action_dit import Fixed16PiActionDiT

    assert Fixed16PiActionDiT(action_dim=7).action_horizon == 32
    defaulted = build_architecture(
        "dual_system_fixed16_pi",
        {"video_dim": D_B, "num_dit_layers": N_B, "action_dim": ACTION_DIM},
    )
    assert defaulted.action_backbone.action_horizon == 32


def test_generate_without_an_observation_frame_is_refused(arch):
    """`generate` is the deploy entry point; the observation frame is mandatory.

    The full engine-shaped call is exercised on the VLM side
    (``test_generate_adapts_the_engine_call_onto_the_cached_condition_loop``) —
    the adapter itself lives on the shared base, so it is covered once. What is
    worth pinning here is that the video path cannot be driven without pixels
    either: it has no other visual input to fall back on.
    """
    with pytest.raises(ValueError, match="first_frame_image"):
        arch.generate(schedule=None, prompt="pick up the cube")


def test_last_hidden_ablation_taps_only_the_final_block():
    ablation = _build_arch(condition_sampling="last_hidden")
    assert ablation.backbone_conditioner.tap_indices == [N_B - 1] * NUM_CROSS
    assert ablation.backbone_conditioner.required_block_indices == [N_B - 1]


# ---------------------------------------------------------------------------
# condition_pathway: report (default) vs starvla
# ---------------------------------------------------------------------------


def test_pathway_table_covers_both_and_report_is_the_default():
    from openwam.model.architectures.fixed16_pi_base import CONDITION_PATHWAYS

    assert set(CONDITION_PATHWAYS) == {"report", "starvla"}
    report = CONDITION_PATHWAYS["report"]
    assert report["use_resampler"] and report["add_depth_embedding"]
    assert report["depth_norm_mode"] == "parameter_free"
    assert report["position_embedding_scope"] == "sequence"
    starvla = CONDITION_PATHWAYS["starvla"]
    assert not starvla["use_resampler"] and not starvla["add_depth_embedding"]
    assert starvla["depth_norm_mode"] == "none"
    assert starvla["position_embedding_scope"] == "action"


def test_starvla_pathway_conditioner_is_only_the_projector():
    """One shared Linear(D_B, 1024) and nothing else, as WanPI has."""
    conditioner = BackboneConditioner(
        backbone_hidden_dim=3072, num_backbone_blocks=N_B, use_resampler=False,
        add_depth_embedding=False, depth_norm_mode="none",
    )
    assert conditioner.resampler is None
    assert conditioner.depth_embedding is None
    assert list(dict(conditioner.named_parameters())) == ["projector.weight", "projector.bias"]
    assert _n(conditioner.projector) == 1024 * 3072 + 1024


@pytest.mark.parametrize("seq_len", [40, 391])
def test_starvla_conditions_keep_the_backbone_sequence(seq_len):
    """Width is what must match across backbones; token count need not.

    The Action Expert's K/V projections are Linear(1024, 1024) whatever arrives,
    so nothing about the head depends on the condition length.
    """
    conditioner = BackboneConditioner(
        backbone_hidden_dim=2048, num_backbone_blocks=28, use_resampler=False,
        add_depth_embedding=False, depth_norm_mode="none",
    )
    conditions = conditioner([torch.randn(3, seq_len, 2048) for _ in range(28)])
    assert len(conditions) == NUM_CROSS
    assert all(c.shape == (3, seq_len, DIM) for c in conditions)


def test_padding_mask_is_absorbed_by_the_resampler_but_forwarded_without_it():
    torch.manual_seed(0)
    hidden = [torch.randn(1, 12, 64) for _ in range(8)]
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[:, 6:] = False

    with_resampler = BackboneConditioner(64, 8).eval()
    with torch.no_grad():
        masked = with_resampler(hidden, key_padding_mask=mask)[0]
        truncated = with_resampler([h[:, :6] for h in hidden])[0]
    assert torch.allclose(masked, truncated, atol=1e-5), "resampler must consume the mask"
    assert with_resampler.last_key_padding_mask is None

    without = BackboneConditioner(
        64, 8, use_resampler=False, add_depth_embedding=False, depth_norm_mode="none"
    ).eval()
    with torch.no_grad():
        out = without(hidden, key_padding_mask=mask)
    assert out[0].shape[1] == 12, "conditions must keep the padded sequence"
    assert without.last_key_padding_mask is mask, "the mask has to reach the Action Expert"
    with pytest.raises(ValueError, match="does not match the condition sequence"):
        without(hidden, key_padding_mask=torch.ones(1, 99, dtype=torch.bool))


def test_action_expert_masks_the_padded_condition_tail():
    """The padded tail must be unable to influence the prediction."""
    torch.manual_seed(0)
    head = Fixed16PiActionDiT(action_dim=ACTION_DIM, state_dim=0).eval()
    conditions = [torch.randn(1, 12, DIM) for _ in range(NUM_CROSS)]
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[:, 6:] = False
    noisy, buckets = torch.randn(1, HORIZON, ACTION_DIM), torch.tensor([10])
    with torch.no_grad():
        base = head.predict_velocity(noisy, buckets, conditions, condition_padding_mask=mask)
        poisoned = [c.clone() for c in conditions]
        for c in poisoned:
            c[:, 6:] = 1e4
        after = head.predict_velocity(noisy, buckets, poisoned, condition_padding_mask=mask)
    assert torch.allclose(base, after, atol=1e-5)


def test_position_embedding_scope_changes_where_positions_land():
    conditions = [torch.zeros(1, 64, DIM) for _ in range(NUM_CROSS)]
    noisy, buckets = torch.randn(1, HORIZON, ACTION_DIM), torch.tensor([10])
    outs = {}
    for scope in ("sequence", "action"):
        torch.manual_seed(0)
        head = Fixed16PiActionDiT(
            action_dim=ACTION_DIM, state_dim=0, position_embedding_scope=scope
        ).eval()
        with torch.no_grad():
            outs[scope] = head.predict_velocity(noisy, buckets, conditions)
    assert not torch.allclose(outs["sequence"], outs["action"])
    with pytest.raises(ValueError, match="position_embedding_scope"):
        Fixed16PiActionDiT(action_dim=ACTION_DIM, state_dim=0, position_embedding_scope="nowhere")


# ---------------------------------------------------------------------------
# Deploy path: normalization must be applied exactly once, in each direction
# ---------------------------------------------------------------------------


def _q99_normalizer(dim, lo=-0.5, hi=0.5):
    import numpy as np

    from openwam.dataloader.transforms.normalize import Normalizer

    return Normalizer(
        mode="q99",
        stats={"q01": np.full(dim, lo, np.float32), "q99": np.full(dim, hi, np.float32)},
    )


def test_deploy_proprio_is_normalized_once_not_twice():
    """``generate`` must not re-normalize what the engine already normalized.

    ``JointInferenceEngine`` normalizes proprio at engine.py:330 and then calls
    ``architecture.generate``. Normalizing again there is silent and wrong: the
    transform is affine, not idempotent, so the state token the policy conditions
    on lands in the wrong place and saturates near the edges of the range.
    """
    import numpy as np

    from openwam.model.architectures.base import BaseWAMArchitecture

    holder = type("H", (), {})()
    holder.normalizer = _q99_normalizer(20)
    normalize = BaseWAMArchitecture.normalize_deploy_proprio

    raw = np.full(20, 0.25, np.float32)
    once = normalize(holder, raw)
    twice = normalize(holder, once)
    assert not torch.allclose(once, twice), "the fixture is wrong if this is idempotent"
    assert torch.allclose(once, torch.full((20,), 0.5))
    assert torch.allclose(twice, torch.full((20,), 1.0))

    # The real guard: generate() hands proprio straight to predict_action_chunk.
    import inspect

    from openwam.model.architectures.fixed16_pi_base import Fixed16PiArchitectureBase

    body = inspect.getsource(Fixed16PiArchitectureBase.generate)
    # Comments explain why the call is absent, so strip them before looking.
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    assert "normalize_deploy_proprio" not in code, (
        "generate() must not normalize proprio -- the engine already did"
    )


def test_generated_actions_are_unnormalized_for_any_batch_size():
    """A batched request must not come back in normalized model space."""
    import numpy as np

    from openwam.model.architectures.fixed16_pi_base import Fixed16PiArchitectureBase

    normalizer = _q99_normalizer(ACTION_DIM)
    for batch in (1, 3):
        holder = type("H", (), {})()
        holder.normalizer = normalizer
        raw = np.full((batch, HORIZON, ACTION_DIM), 0.5, np.float32)

        out = raw.squeeze(0) if batch == 1 else raw
        out = holder.normalizer.unnormalize(out)
        expected_rank = 2 if batch == 1 else 3
        assert out.ndim == expected_rank
        # q99 over [-0.5, 0.5] maps 0.5 in model space back to 0.25 in real units.
        assert np.allclose(out, 0.25), f"batch={batch} was not unnormalized"

    import inspect

    body = inspect.getsource(Fixed16PiArchitectureBase._predict_action_chunk)
    squeeze_at = body.index("out = out.squeeze(0)")
    unnorm_at = body.index("normalizer.unnormalize(out)")
    assert unnorm_at > squeeze_at
    between = body[squeeze_at:unnorm_at]
    assert "if out.shape[0] == 1" not in between, (
        "unnormalize must sit outside the batch-1 branch"
    )


def test_ffn_keeps_the_historical_state_dict_keys():
    """The FFN output projection must stay at ``ff.2``.

    A parameter-free module inserted into the Sequential (a Dropout, say) still
    occupies an index and would move the projection to ``ff.3``, making every
    existing checkpoint fail to load with a missing/unexpected key pair that
    names no parameter anyone changed. That happened once; this pins it.
    """
    head = Fixed16PiActionDiT(action_dim=ACTION_DIM, state_dim=0)
    assert [type(m).__name__ for m in head.blocks[0].ff] == ["Linear", "GELU", "Linear"]
    assert [n for n, _ in head.named_parameters() if n.startswith("blocks.0.ff")] == [
        "blocks.0.ff.0.weight",
        "blocks.0.ff.0.bias",
        "blocks.0.ff.2.weight",
        "blocks.0.ff.2.bias",
    ]
    assert [layer.out_features for layer in head.blocks[0].ff_linears] == [4096, DIM]
