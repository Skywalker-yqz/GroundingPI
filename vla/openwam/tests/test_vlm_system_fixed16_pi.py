"""Tests for the VLM-only Fixed-16 π-style architecture.

The load-bearing assertion in this file is
:func:`test_action_expert_is_byte_identical_to_the_video_path`: the whole
protocol rests on a Qwen run and a Wan run sharing one Action Expert, so it is
checked operationally (same weights + same conditions → same velocity), not just
by counting parameters.

Everything runs on CPU against a stub VLM backbone; no checkpoint is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from openwam.model.action_backbone.backbone_conditioner import normalized_depth_indices
from tests.test_fixed16_pi_action_dit import HEAD_CONST, NUM_CROSS, TOKENS_PER_TAP
from tests.test_fixed16_pi_action_dit import _build_arch as _build_video_arch

N_LAYERS, HIDDEN, SEQ = 36, 64, 12
ACTION_DIM = STATE_DIM = 7
HORIZON = 5
DIM = 1024


class _StubVlm(nn.Module):
    """Ducks ``VlmBackbone`` with exactly what the architecture touches."""

    def __init__(self, n_layers: int = N_LAYERS, hidden: int = HIDDEN, seq: int = SEQ):
        super().__init__()
        self._n, self._h, self._seq = n_layers, hidden, seq
        self.proj = nn.Linear(1, hidden)
        self.forwards = 0
        self.requested: list | None = None

    @property
    def hidden_size(self) -> int:
        return self._h

    @property
    def num_layers(self) -> int:
        return self._n

    def prepare_vlm_inputs(self, prompts, images):
        b = len(prompts)
        mask = torch.ones(b, self._seq, dtype=torch.long)
        mask[-1, self._seq - 4 :] = 0  # padded tail so the mask path is exercised
        return {"input_ids": torch.zeros(b, self._seq, dtype=torch.long), "attention_mask": mask}

    def batch_vlm_inputs(self, x):
        return x if isinstance(x, dict) else x[0]

    def extract_layerwise_features(self, vlm_inputs, block_indices):
        self.forwards += 1
        self.requested = sorted(block_indices)
        base = self.proj(vlm_inputs["input_ids"].float().unsqueeze(-1))
        return {i: base + float(i) for i in self.requested}, vlm_inputs["attention_mask"].to(torch.bool)

    def set_dtype_device(self, dtype, device):
        self.to(dtype=dtype, device=device)


def _build_vlm_arch(**overrides):
    from openwam.model import build_architecture

    cfg = {
        "vlm_dim": HIDDEN,
        "num_vlm_layers": N_LAYERS,
        "action_dim": ACTION_DIM,
        "state_dim": STATE_DIM,
        "use_proprioception": True,
        "action_horizon": HORIZON,
    }
    cfg.update(overrides)
    arch = build_architecture("vlm_system_fixed16_pi", cfg)
    arch.vlm_backbone = _StubVlm()
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _samples(n: int = 2):
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    return [
        {
            "video": [img, img],
            "prompt": f"put the cube in the bowl {i}",
            "action": np.random.uniform(-1, 1, (HORIZON, ACTION_DIM)).astype(np.float32),
            "action_mask": np.ones((HORIZON, ACTION_DIM), dtype=bool),
            "proprio": np.zeros((STATE_DIM,), dtype=np.float32),
        }
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def arch():
    return _build_vlm_arch()


# ---------------------------------------------------------------------------
# Registry / construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "framework,variant,expected",
    [
        ("vlm_system", "fixed16_pi_layerwise", "vlm_system_fixed16_pi"),
        # Regression guard for the shared-base refactor.
        ("dual_system", "fixed16_pi_layerwise", "dual_system_fixed16_pi"),
    ],
)
def test_registry_resolution(framework, variant, expected):
    from types import SimpleNamespace

    from openwam.model import resolve_architecture_config

    resolved = resolve_architecture_config(
        SimpleNamespace(architecture={"framework": framework, "variant": variant}, action_backbone={})
    )
    assert resolved.registry_name == expected


def test_no_video_backbone_is_built(arch):
    assert arch.video_backbone is None
    assert "video_backbone" not in arch.backbones
    assert sorted(arch.backbones) == ["action_backbone", "backbone_conditioner", "vlm_backbone"]


def test_attaching_a_video_backbone_is_rejected():
    """A stray `model/video_backbone=...` override would load billions of dead params."""
    from openwam.model.architectures.vlm_system.fixed16_pi import VlmSystemFixed16PiArchitecture

    class _WithInjectedVideoBackbone(VlmSystemFixed16PiArchitecture):
        def _init_video_backbone(self, cfg):
            self.video_backbone = object()  # pretend Hydra composed one in

    with pytest.raises(ValueError, match="must not be given a video backbone"):
        _WithInjectedVideoBackbone(cfg={"vlm_dim": HIDDEN, "num_vlm_layers": N_LAYERS, "action_dim": ACTION_DIM})


def test_hydra_config_composes_without_a_video_backbone():
    """`model=vlm_system` must not pull in a video_backbone group.

    ``resolve_architecture_config`` injects ``params["video_backbone"]`` whenever
    ``model.video_backbone`` exists, which would make the architecture raise.
    """
    import os

    from hydra import compose, initialize_config_dir

    from openwam.model import resolve_architecture_config

    configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["model=vlm_system"])

    assert "video_backbone" not in cfg.model
    assert cfg.model.architecture.framework == "vlm_system"
    # The VLM co-trains with the action expert, so it must NOT be frozen — and
    # the detach has to agree, or the first forward refuses the run.
    assert not list(getattr(cfg.model, "freeze", []) or [])
    assert cfg.model.action_backbone.detach_backbone_features is False

    resolved = resolve_architecture_config(cfg.model)
    assert resolved.registry_name == "vlm_system_fixed16_pi"
    assert "video_backbone" not in resolved.params
    assert resolved.params["vlm_backbone"]["checkpoint_path"]
    # action_backbone group keys are flattened into the architecture params.
    assert resolved.params["action_horizon"] == 32
    assert resolved.params["condition_sampling"] == "normalized_depth"


def test_taps_follow_the_vlm_depth(arch):
    assert arch.backbone_conditioner.tap_indices == normalized_depth_indices(N_LAYERS)
    assert len(arch.backbone_conditioner.tap_indices) == NUM_CROSS


# ---------------------------------------------------------------------------
# The headline claim
# ---------------------------------------------------------------------------


def test_action_expert_is_byte_identical_to_the_video_path(arch):
    """Same Action Expert, different backbone — checked operationally."""
    video_arch = _build_video_arch()
    vlm_head, video_head = arch.action_backbone, video_arch.action_backbone

    vlm_shapes = {k: tuple(v.shape) for k, v in vlm_head.state_dict().items()}
    video_shapes = {k: tuple(v.shape) for k, v in video_head.state_dict().items()}
    assert vlm_shapes == video_shapes
    expected = HEAD_CONST + 2049 * ACTION_DIM + 1024 * STATE_DIM
    assert sum(p.numel() for p in vlm_head.parameters()) == expected
    assert sum(p.numel() for p in video_head.parameters()) == expected

    # Same weights + same conditions must give the same velocity, or the two
    # runs are not measuring the same thing.
    vlm_head.load_state_dict(video_head.state_dict())
    conditions = [torch.randn(1, TOKENS_PER_TAP, DIM) for _ in range(NUM_CROSS)]
    state = torch.randn(1, 1, STATE_DIM)
    noisy = torch.randn(1, HORIZON, ACTION_DIM)
    buckets = torch.tensor([123])
    with torch.no_grad():
        from_vlm = vlm_head.predict_velocity(noisy, buckets, conditions, state)
        from_video = video_head.predict_velocity(noisy, buckets, conditions, state)
    assert torch.equal(from_vlm, from_video)


def test_parameter_report_names_the_vlm_backbone(arch):
    report = arch.parameter_report()
    assert report["action_head"] == HEAD_CONST + 2049 * ACTION_DIM + 1024 * STATE_DIM
    assert report["backbone_projector"] == 1024 * HIDDEN + 1024
    assert "vlm_backbone" in report and "video_backbone" not in report


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_prepare_inputs_carries_actions_and_no_video_keys(arch):
    inputs = arch.prepare_inputs(_samples())
    # trainer:572 raises when `actions` is missing.
    assert inputs["actions"].shape == (2, HORIZON, ACTION_DIM)
    assert inputs["action_is_pad"].shape == (2, HORIZON, ACTION_DIM)
    assert inputs["proprio"].shape == (2, STATE_DIM)
    assert set(inputs["vlm_inputs"]) == {"input_ids", "attention_mask"}
    for banned in ("input_latents", "latents", "max_timestep_boundary", "video_is_pad"):
        assert banned not in inputs


def test_preprocess_is_refused(arch):
    with pytest.raises(NotImplementedError, match="no video backbone"):
        arch.preprocess(frames=None, text=None)


def test_generate_adapts_the_engine_call_onto_the_cached_condition_loop(arch):
    """`JointInferenceEngine` only knows `generate`, so deploy runs through it.

    The engine's video arguments are dropped rather than honoured — there is no
    video stream and the Euler step count is pinned by the protocol.
    """
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    arch.vlm_backbone.forwards = 0
    result = arch.generate(
        schedule=[1.0, 0.5],
        prompt="pick up the cube",
        first_frame_image=[img],
        proprio=np.zeros((STATE_DIM,), dtype=np.float32),
        num_inference_steps=50,  # ignored: the protocol pins it
        decode_video=True,  # ignored: nothing to decode
    )
    assert result["video"] is None
    assert result["actions"].shape == (HORIZON, ACTION_DIM)
    assert set(result["latency"]) == {"backbone_feature_extraction_s", "action_denoising_s", "end_to_end_s"}
    assert arch.vlm_backbone.forwards == 1, "one backbone forward per observation, whatever the step count"


def test_generate_without_an_observation_frame_is_refused(arch):
    with pytest.raises(ValueError, match="first_frame_image"):
        arch.generate(schedule=None, prompt="pick up the cube")


# ---------------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------------


def test_training_step_runs_the_vlm_once_and_has_no_video_loss(arch):
    inputs = arch.prepare_inputs(_samples())
    arch.vlm_backbone.forwards = 0
    out = arch.compute_loss(lambda_video=0.0, lambda_action=1.0, **inputs)

    assert torch.isfinite(out["loss"])
    assert out["loss_video"].item() == 0.0
    assert arch.vlm_backbone.forwards == 1
    assert arch.vlm_backbone.requested == arch.backbone_conditioner.required_block_indices

    out["loss"].backward()
    assert all(p.grad is not None for p in arch.backbone_conditioner.parameters())
    assert all(p.grad is not None for p in arch.action_backbone.parameters())
    arch.zero_grad(set_to_none=True)


def test_protocol_a_detaches_the_backbone_and_protocol_b_does_not():
    frozen = _build_vlm_arch(detach_backbone_features=True)
    # Protocol A is both halves: the detach here and the `freeze:` entry the model
    # config carries. Setting only one is refused (see the test below), so the
    # freeze the trainer would apply has to be applied here too. The stub keeps
    # its weights at the top level rather than under `vlm_model`, so the whole
    # backbone is the equivalent path.
    frozen.freeze_modules(["vlm_backbone"])
    frozen.compute_loss(lambda_video=0.0, lambda_action=1.0, **frozen.prepare_inputs(_samples()))["loss"].backward()
    assert frozen.vlm_backbone.proj.weight.grad is None, "Protocol A must not backprop into the backbone"

    adapted = _build_vlm_arch(detach_backbone_features=False)
    adapted.compute_loss(lambda_video=0.0, lambda_action=1.0, **adapted.prepare_inputs(_samples()))["loss"].backward()
    assert adapted.vlm_backbone.proj.weight.grad is not None, "Protocol B must reach the backbone"


def test_half_set_protocol_a_is_refused():
    """detach without freeze is invisible otherwise: no gradient, but the optimizer
    still carries the backbone and the parameter report still calls it trainable."""
    half_set = _build_vlm_arch(detach_backbone_features=True)  # no freeze_modules call
    with pytest.raises(ValueError, match="still has trainable parameters"):
        half_set.compute_loss(lambda_video=0.0, lambda_action=1.0, **half_set.prepare_inputs(_samples()))


def test_padding_mask_reaches_the_resampler():
    """Text padding must not leak into the conditions the Action Expert reads."""
    arch = _build_vlm_arch()
    arch.eval()
    inputs = arch.prepare_inputs(_samples(2))
    vlm_inputs = inputs["vlm_inputs"]
    padded = vlm_inputs["attention_mask"] == 0
    assert padded.any(), "the stub should produce a padded tail"

    with torch.no_grad():
        clean = arch.encode_conditions(vlm_inputs=vlm_inputs)
        # Garbage in the padded positions must be invisible downstream.
        noisy_ids = vlm_inputs["input_ids"].clone().float()
        noisy_ids[padded] = 1e3
        polluted = arch.encode_conditions(
            vlm_inputs={"input_ids": noisy_ids.long(), "attention_mask": vlm_inputs["attention_mask"]}
        )

    for a, b in zip(clean, polluted, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_inference_runs_the_vlm_once_for_all_denoising_steps(arch):
    inputs = arch.prepare_inputs(_samples(1))
    arch.vlm_backbone.forwards = 0
    result = arch.predict_action_chunk(proprio=inputs["proprio"], vlm_inputs=inputs["vlm_inputs"])

    assert arch.vlm_backbone.forwards == 1
    assert arch.action_backbone.num_inference_steps == 10
    # Squeezed like BaseWAMArchitecture.generate: one request -> (T, action_dim).
    assert result["actions"].shape == (HORIZON, ACTION_DIM)
    assert result["video"] is None
    # Latency keys must match the video path exactly so benchmark tooling has no branch.
    assert set(result["latency"]) == {
        "backbone_feature_extraction_s",
        "action_denoising_s",
        "end_to_end_s",
    }


# ---------------------------------------------------------------------------
# Co-training: the unfrozen VLM's weights must survive a save/load round trip
# ---------------------------------------------------------------------------


def test_cotrained_vlm_weights_are_written_to_the_checkpoint(tmp_path):
    """Excluding a co-trained VLM would silently discard its training."""
    from safetensors import safe_open

    arch = _build_vlm_arch()
    assert arch.vlm_is_trainable(), "the stub VLM should be trainable by default"

    path = tmp_path / "ckpt.safetensors"
    arch.save_checkpoint(str(path))
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
    assert any(k.startswith("vlm_backbone.") for k in keys)
    assert any(k.startswith("action_backbone.") for k in keys)


def test_frozen_vlm_is_still_excluded(tmp_path):
    """A frozen VLM stays out of the checkpoint file."""
    from safetensors import safe_open

    arch = _build_vlm_arch()
    arch.vlm_backbone.requires_grad_(False)
    assert not arch.vlm_is_trainable()

    path = tmp_path / "frozen.safetensors"
    arch.save_checkpoint(str(path))
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
    assert not any(k.startswith("vlm_backbone.") for k in keys)


def test_tied_weights_do_not_break_saving(tmp_path):
    """Qwen ties lm_head to embed_tokens; safetensors rejects shared storage."""
    from safetensors import safe_open

    arch = _build_vlm_arch()
    # Mimic the tie: two parameters backed by the same storage.
    shared = nn.Parameter(torch.randn(4, 4))
    arch.vlm_backbone.embed = shared
    arch.vlm_backbone.head = shared
    assert arch.vlm_is_trainable()

    path = tmp_path / "tied.safetensors"
    arch.save_checkpoint(str(path))  # would raise without dedupe
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
    tied = {k for k in keys if k.endswith((".embed", ".head"))}
    assert len(tied) == 1, f"exactly one copy of the tied tensor should be stored, got {tied}"

    # The dropped key must be tolerated on load (the tie is rebuilt at construction).
    arch.load_checkpoint(str(path), strict=True)


def test_vlm_gets_its_own_optimizer_lr_group():
    """A co-trained VLM must be adjustable on the same terms as a video DiT."""
    from types import SimpleNamespace

    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    arch = _build_vlm_arch()
    holder = SimpleNamespace(architecture=arch)

    flat = build_trainable_parameters(holder)
    assert isinstance(flat, list) and all(torch.is_tensor(p) for p in flat)

    groups = build_trainable_parameters(holder, vlm_lr=1e-5)
    vlm_group = [g for g in groups if g.get("lr") == 1e-5]
    assert len(vlm_group) == 1
    vlm_param_ids = {id(p) for p in vlm_group[0]["params"]}
    assert vlm_param_ids == {id(p) for p in arch.vlm_backbone.parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# Observation frames
# ---------------------------------------------------------------------------


def test_vlm_path_rejects_multi_frame_observations():
    with pytest.raises(ValueError, match="num_observation_frames=1 only"):
        _build_vlm_arch(num_observation_frames=5)


def test_both_paths_default_to_a_single_observation_frame(arch):
    assert arch.num_observation_frames == 1
    assert _build_video_arch().num_observation_frames == 1


def test_video_path_feeds_the_vae_the_same_image_the_vlm_gets():
    """The direct evidence that the two backbones see one identical observation."""
    from openwam.model.architectures.utils.common import extract_first_image

    video_arch = _build_video_arch()
    seen = {}
    original = video_arch.preprocess

    def spy(**kwargs):
        seen["frames"] = kwargs["frames"]
        return original(**kwargs)

    video_arch.preprocess = spy
    batch = _samples(2)
    video_arch.prepare_inputs(batch)

    assert [len(f) for f in seen["frames"]] == [1, 1], "only the observation frame may reach the VAE"
    for sent, sample in zip(seen["frames"], batch, strict=True):
        assert sent[0] is extract_first_image(sample)
    # The caller's samples must not be mutated by the truncation.
    assert [len(s["video"]) for s in batch] == [2, 2]


def test_video_mask_is_truncated_alongside_the_clip():
    """A full-length mask against a 1-frame clip breaks the latent-mask downsample."""
    video_arch = _build_video_arch()
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    batch = [
        {
            "video": [img] * 9,
            "video_mask": np.ones((9,), dtype=bool),
            "prompt": "x",
            "action": np.zeros((HORIZON, ACTION_DIM), dtype=np.float32),
            "action_mask": np.ones((HORIZON, ACTION_DIM), dtype=bool),
        }
    ]
    truncated = video_arch._truncate_to_observation(batch)
    assert len(truncated[0]["video"]) == 1
    assert truncated[0]["video_mask"].shape == (1,)
    # End to end: prepare_inputs must not raise on the mask/latent mismatch.
    video_arch.prepare_inputs(batch)


def test_observation_frame_count_rejects_values_wan_would_round():
    """2/3/4 would be silently padded to 5 by Wan; that must fail loudly."""
    with pytest.raises(ValueError, match=r"n % 4 == 1"):
        _build_video_arch(num_observation_frames=2)


# ---------------------------------------------------------------------------
# Layer-wise extraction on the real Qwen3VLBackbone (fake inner HF model)
# ---------------------------------------------------------------------------


def _fake_qwen_backbone(n_layers: int = 36, hidden: int = 6):
    """Real Qwen3VLBackbone methods over a fake inner model — no transformers load.

    Uses the ``object.__new__`` trick to skip the heavy constructor.
    """
    from openwam.model.vlm_backbone.qwen3_vl_backbone import Qwen3VLBackbone

    class _FakeQwen(nn.Module):
        def __init__(self):
            super().__init__()
            text_cfg = type("T", (), {"hidden_size": hidden, "num_hidden_layers": n_layers})()
            self.config = type("Cfg", (), {"text_config": text_cfg})()
            self.param = nn.Parameter(torch.zeros(1))
            self.model = self

        def forward(self, **kwargs):
            batch = kwargs["input_ids"].shape[0]
            seq = kwargs["input_ids"].shape[1]
            # hidden_states[k] is filled with the constant k, so an off-by-one
            # in the tap mapping is directly visible in the values.
            hs = tuple(torch.full((batch, seq, hidden), float(k)) for k in range(n_layers + 1))
            return type("Out", (), {"last_hidden_state": hs[-1], "hidden_states": hs})()

    backbone = object.__new__(Qwen3VLBackbone)
    nn.Module.__init__(backbone)
    backbone.dtype = torch.float32
    backbone._checkpoint_path = "fake"
    backbone.processor = None
    backbone.vlm_model = _FakeQwen()
    backbone.add_module("vlm_model", backbone.vlm_model)
    return backbone


def _fake_inputs(batch: int = 2, seq: int = 5):
    mask = torch.ones(batch, seq, dtype=torch.long)
    mask[-1, seq - 2 :] = 0
    return {"input_ids": torch.zeros(batch, seq, dtype=torch.long), "attention_mask": mask}


def test_block_i_maps_to_hidden_state_i_plus_one():
    """hidden_states[0] is the embedding output and must never be tapped."""
    backbone = _fake_qwen_backbone()
    assert backbone.num_layers == 36
    taps_idx = normalized_depth_indices(backbone.num_layers)
    taps, mask = backbone.extract_layerwise_features(_fake_inputs(), taps_idx)

    assert sorted(taps) == taps_idx
    for i in taps_idx:
        assert taps[i].mean().item() == pytest.approx(i + 1), f"block {i} should read hidden_states[{i + 1}]"
    assert mask.dtype == torch.bool and mask.shape == (2, 5)


def test_extract_features_is_unchanged_by_the_shared_helper():
    """The VLM backbone contract must survive the _to_model_inputs refactor."""
    backbone = _fake_qwen_backbone()
    out = backbone.extract_features(_fake_inputs())
    assert out.shape == (2, 5, 6)
    assert out.mean().item() == pytest.approx(float(backbone.num_layers))


def test_layerwise_extraction_rejects_out_of_range_blocks():
    backbone = _fake_qwen_backbone()
    with pytest.raises(ValueError, match=r"block_indices must lie in \[0, 35\]"):
        backbone.extract_layerwise_features(_fake_inputs(), [0, 99])


def test_layerwise_extraction_detects_a_changed_hidden_state_count():
    """A silent upstream change to the tuple length would shift every tap."""
    backbone = _fake_qwen_backbone()
    original = backbone.vlm_model.forward

    def truncated(**kwargs):
        out = original(**kwargs)
        return type("Out", (), {"last_hidden_state": out.hidden_states[-1], "hidden_states": out.hidden_states[:-1]})()

    backbone.vlm_model.forward = truncated
    with pytest.raises(RuntimeError, match="tap-to-block mapping is no longer valid"):
        backbone.extract_layerwise_features(_fake_inputs(), [0, 5])


# ---------------------------------------------------------------------------
# The other VLM families: Rex-Omni (Qwen2.5-VL), PaliGemma, LocateAnything
# ---------------------------------------------------------------------------


def _fake_hf_backbone(cls, n_layers: int, hidden: int, *, nested_text_config: bool = True, **attrs):
    """A real wrapper class over a fake inner model — no transformers load.

    Uses the ``object.__new__`` trick to skip the heavy constructor, so the shared
    HFVlmBackbone machinery is exercised for real while the checkpoint is not.
    """

    class _FakeInner(nn.Module):
        def __init__(self):
            super().__init__()
            text_cfg = type("T", (), {"hidden_size": hidden, "num_hidden_layers": n_layers})()
            self.config = (
                type("Cfg", (), {"text_config": text_cfg})()
                if nested_text_config
                # Some releases keep the geometry at the top level instead.
                else type("Cfg", (), {"hidden_size": hidden, "num_hidden_layers": n_layers})()
            )
            self.param = nn.Parameter(torch.zeros(1))
            self.model = self

        def forward(self, **kwargs):
            batch, seq = kwargs["input_ids"].shape
            hs = tuple(torch.full((batch, seq, hidden), float(k)) for k in range(n_layers + 1))
            return type("Out", (), {"last_hidden_state": hs[-1], "hidden_states": hs})()

    backbone = object.__new__(cls)
    nn.Module.__init__(backbone)
    backbone.dtype = torch.float32
    backbone._checkpoint_path = "fake"
    backbone.processor = None
    backbone._max_length = 512
    backbone.vlm_model = _FakeInner()
    backbone.add_module("vlm_model", backbone.vlm_model)
    for key, value in attrs.items():
        setattr(backbone, key, value)
    return backbone


def _fake_batch(batch: int = 2, seq: int = 5):
    mask = torch.ones(batch, seq, dtype=torch.long)
    mask[-1, seq - 2 :] = 0
    return {"input_ids": torch.zeros(batch, seq, dtype=torch.long), "attention_mask": mask}


@pytest.mark.parametrize(
    "module_path,class_name,n_layers,hidden",
    [
        ("openwam.model.vlm_backbone.qwen2_5_vl_backbone", "Qwen2_5VLBackbone", 36, 2048),  # Rex-Omni-3B
        ("openwam.model.vlm_backbone.paligemma_backbone", "PaliGemmaBackbone", 18, 2048),
        ("openwam.model.vlm_backbone.locate_anything_backbone", "LocateAnythingBackbone", 36, 2048),
        ("openwam.model.vlm_backbone.rynnbrain_backbone", "RynnBrainBackbone", 28, 2048),
    ],
)
def test_every_vlm_family_reports_geometry_and_taps_the_right_blocks(module_path, class_name, n_layers, hidden):
    """One shared contract: geometry from the checkpoint, block i -> hidden[i+1]."""
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)
    backbone = _fake_hf_backbone(cls, n_layers, hidden)

    assert backbone.num_layers == n_layers
    assert backbone.hidden_size == hidden

    taps_idx = normalized_depth_indices(backbone.num_layers)
    taps, mask = backbone.extract_layerwise_features(_fake_batch(), taps_idx)
    assert sorted(taps) == taps_idx
    for i in taps_idx:
        assert taps[i].mean().item() == pytest.approx(i + 1), f"block {i} must read hidden_states[{i + 1}]"
    assert mask.dtype == torch.bool

    # last_hidden_state stays the plain extract_features contract.
    assert backbone.extract_features(_fake_batch()).mean().item() == pytest.approx(float(n_layers))


def test_flat_config_geometry_is_read_from_the_top_level():
    """Qwen2.5-VL keeps num_hidden_layers/hidden_size outside text_config on some releases."""
    from openwam.model.vlm_backbone.qwen2_5_vl_backbone import Qwen2_5VLBackbone

    backbone = _fake_hf_backbone(Qwen2_5VLBackbone, 36, 2048, nested_text_config=False)
    assert backbone.num_layers == 36
    assert backbone.hidden_size == 2048


def test_paligemma_does_not_wrap_the_prompt_in_a_chat_template():
    """PaliGemma's processor prepends image tokens itself; templating would corrupt it."""
    from openwam.model.vlm_backbone.paligemma_backbone import PaliGemmaBackbone

    backbone = _fake_hf_backbone(PaliGemmaBackbone, 18, 2048)
    assert backbone.format_prompt("pick up the cube") == "pick up the cube"
    assert "token_type_ids" in PaliGemmaBackbone.EXTRA_TENSOR_KEYS


def test_locate_anything_requires_explicit_remote_code_opt_in():
    """Loading it executes Python from the checkpoint dir; that must be deliberate."""
    from openwam.model.vlm_backbone.locate_anything_backbone import LocateAnythingBackbone

    with pytest.raises(ValueError, match="allow_remote_code"):
        LocateAnythingBackbone(checkpoint_path="somewhere", allow_remote_code=False)


def test_locate_anything_uses_the_top_level_forward():
    """It has no `.model` inner module — run_backbone must not reach for one."""
    from openwam.model.vlm_backbone.locate_anything_backbone import LocateAnythingBackbone

    backbone = _fake_hf_backbone(LocateAnythingBackbone, 36, 2048)
    called = {}
    real_forward = backbone.vlm_model.forward

    def spy(**kwargs):
        called["hit"] = True
        return real_forward(**kwargs)

    backbone.vlm_model.forward = spy
    backbone.extract_features(_fake_batch())
    assert called.get("hit"), "the top-level forward should have been called"


def test_locate_anything_marks_the_prefix_boundary_with_a_position_id_drop():
    """Its block mask reads the prompt length off a drop in the position ids.

    ``find_prefix_seq_length_by_pe`` returns -1 when the ids never drop, and at -1
    ``create_block_diff_mask_by_pe_4d`` degenerates to block-diagonal: each token
    sees only its own block, not even the image. The drop has to land on each
    sample's real length, which needs one extra column for the unpadded sample —
    trimmed back off the outputs so the caller's key-padding mask still fits.
    """
    from openwam.model.vlm_backbone.locate_anything_backbone import LocateAnythingBackbone

    backbone = _fake_hf_backbone(LocateAnythingBackbone, 36, 2048)
    seen = {}
    real_forward = backbone.vlm_model.forward

    def spy(**kwargs):
        seen.update(kwargs)
        return real_forward(**kwargs)

    backbone.vlm_model.forward = spy
    batch = _fake_batch(batch=2, seq=5)  # row 1 has 2 padded positions, row 0 none
    taps, mask = backbone.extract_layerwise_features(batch, [0, 35])

    positions = seen["position_ids"]
    assert positions.shape == (2, 6), "one column must be appended so the unpadded row can drop"
    assert seen["input_ids"].shape == (2, 6)
    assert seen["attention_mask"][:, -1].tolist() == [0, 0], "the appended column is not real input"

    def first_drop(row):
        return next((i + 1 for i in range(len(row) - 1) if row[i + 1] < row[i]), -1)

    # Row 0 is full-length (5) and only drops thanks to the appended column;
    # row 1's three real tokens must drop at 3, not at the padded width.
    assert [first_drop(row.tolist()) for row in positions] == [5, 3]

    assert taps[0].shape == (2, 5, 2048), "outputs must come back at the caller's length"
    assert mask.shape == (2, 5)


def test_control_kwargs_are_filtered_to_the_forward_signature():
    """A family whose forward lacks past_key_values must not raise TypeError."""
    from openwam.model.vlm_backbone.qwen2_5_vl_backbone import Qwen2_5VLBackbone

    backbone = _fake_hf_backbone(Qwen2_5VLBackbone, 36, 2048)

    class _PickyInner(nn.Module):
        """Accepts only inputs + output_hidden_states — no **kwargs escape hatch."""

        def __init__(self, inner):
            super().__init__()
            self.config = inner.config
            self.model = self

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            batch, seq = input_ids.shape
            hs = tuple(torch.full((batch, seq, 2048), float(k)) for k in range(37))
            return type("Out", (), {"last_hidden_state": hs[-1], "hidden_states": hs})()

    backbone.vlm_model = _PickyInner(backbone.vlm_model)
    taps, _ = backbone.extract_layerwise_features(_fake_batch(), [0, 35])
    assert taps[0].mean().item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "group,expected_name",
    [
        ("qwen3_vl_4b", "qwen3_vl"),
        ("rex_omni_3b", "qwen2_5_vl"),
        ("paligemma_3b", "paligemma"),
        ("locate_anything_3b", "locate_anything"),
    ],
)
def test_every_vlm_config_group_resolves_to_a_registered_backbone(group, expected_name):
    import os

    from hydra import compose, initialize_config_dir

    from openwam.model import resolve_architecture_config
    from openwam.model.vlm_backbone import _VLM_BACKBONE_REGISTRY

    configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(
            config_name="train", overrides=["model=vlm_system", f"model/vlm_backbone={group}"]
        )
    params = resolve_architecture_config(cfg.model).params["vlm_backbone"]
    name = params.get("name") or "qwen3_vl"
    assert name == expected_name
    assert name in _VLM_BACKBONE_REGISTRY
    assert params["checkpoint_path"]
    if expected_name == "locate_anything":
        # The remote-code opt-in must be present in the config, not defaulted.
        assert params["allow_remote_code"] is True


def test_token_aligned_processor_extras_survive_list_collation():
    """A key the model's forward needs must not be dropped when batching samples.

    transformers 5.x emits mm_token_type_ids for the Qwen family (and
    token_type_ids for PaliGemma); both are token-aligned, so collation has to
    pad them like input_ids rather than drop or plain-concat them.
    """
    from openwam.model.vlm_backbone.paligemma_backbone import PaliGemmaBackbone
    from openwam.model.vlm_backbone.qwen3_vl_backbone import Qwen3VLBackbone

    for cls, key in ((Qwen3VLBackbone, "mm_token_type_ids"), (PaliGemmaBackbone, "token_type_ids")):
        backbone = _fake_hf_backbone(cls, 8, 16)
        backbone.processor = type("P", (), {"tokenizer": type("T", (), {"pad_token_id": 0})()})()
        samples = [
            {
                "input_ids": torch.zeros(1, n, dtype=torch.long),
                "attention_mask": torch.ones(1, n, dtype=torch.long),
                key: torch.full((1, n), 7, dtype=torch.long),
            }
            for n in (5, 3)  # ragged on purpose
        ]
        batched = backbone.batch_vlm_inputs(samples)
        assert key in batched, f"{cls.__name__} dropped {key}"
        assert batched[key].shape == batched["input_ids"].shape, f"{key} must be padded like input_ids"
        # The short sample's tail is padding, the real positions keep their value.
        assert batched[key][1, :3].tolist() == [7, 7, 7]
        assert batched[key][1, 3:].tolist() == [0, 0]


def test_non_tensor_processor_outputs_are_converted_not_dropped():
    """A grid array the model's forward requires must survive prepare_vlm_inputs.

    LocateAnything's processor returns image_grid_hws as a numpy array; filtering
    on isinstance(v, torch.Tensor) dropped it and the vision tower then failed on
    a None deep inside its forward.
    """
    import numpy as np

    from openwam.model.vlm_backbone.locate_anything_backbone import LocateAnythingBackbone

    backbone = _fake_hf_backbone(LocateAnythingBackbone, 36, 2048)

    class _Proc:
        def __call__(self, text=None, images=None, **kwargs):
            n = len(text)
            return {
                "input_ids": torch.zeros(n, 7, dtype=torch.long),
                "attention_mask": torch.ones(n, 7, dtype=torch.long),
                "pixel_values": torch.zeros(n * 4, 3, 14, 14),
                "image_grid_hws": np.array([[2, 2]] * n),  # ndarray, not a Tensor
                "some_metadata": "not an array",  # must stay out
            }

    backbone.processor = _Proc()
    backbone.format_prompt = lambda p: p
    out = backbone.prepare_vlm_inputs(["a", "b"], [None, None])

    assert "image_grid_hws" in out, "the grid array was dropped"
    assert isinstance(out["image_grid_hws"], torch.Tensor)
    assert out["image_grid_hws"].shape == (2, 2)
    assert "some_metadata" not in out, "non-numeric metadata must not be forwarded"
    # And it must survive list collation too.
    assert "image_grid_hws" in LocateAnythingBackbone.EXTRA_TENSOR_KEYS
