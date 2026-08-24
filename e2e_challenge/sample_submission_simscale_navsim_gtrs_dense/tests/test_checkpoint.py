# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import navsim_gtrs_dense_challenge.policy as policy_module
import numpy as np
import pytest
import torch
from navsim_gtrs_dense_challenge.policy import (
    CHECKPOINT_PREFIX,
    VOCABULARY_KEY,
    GTRSDensePolicy,
    InferenceInput,
    normalized_checkpoint,
)
from navsim_gtrs_dense_challenge.preprocessing import CAMERA_IDS
from navsim_gtrs_dense_challenge.simscale_gtrs_dense.config import GTRSDenseConfig
from navsim_gtrs_dense_challenge.simscale_gtrs_dense.model import GTRSDenseModel
from torch import nn

REAL_CHECKPOINT_SIZE = 269_095_388
REAL_CHECKPOINT_SHA256 = (
    "8dad0395332ccd844785cbfc7c9e24cb3f8d8dbf5cb9ca7f8f8dc75478fcf409"
)
EXPERT_CHECKPOINT_SIZE = 269_095_388
EXPERT_CHECKPOINT_SHA256 = (
    "2496b82f5f256d7de09fca656c7634967b8660eb12e5c10386a587283629a7ff"
)
VOV_REWARD_CHECKPOINT_SIZE = 332_348_155
VOV_REWARD_CHECKPOINT_SHA256 = (
    "7567d269bd8d0757cf906c30612bf1ad167ac7310e8af0ead74dc7798fe54c99"
)
VOV_EXPERT_CHECKPOINT_SIZE = 332_348_155
VOV_EXPERT_CHECKPOINT_SHA256 = (
    "badcf3e7c3e2ecc1d7ecb9fc744c78420c368f96e47b89d1681ade7833cd5e57"
)
NAVHARD_VOCABULARY_PATH = (
    Path(__file__).resolve().parents[1] / "assets/gtrs_dense/navhard_8192.npy"
)
RELEASE_VOCABULARY_PATH = (
    Path(__file__).resolve().parents[1] / "assets/gtrs_dense/navsim_16384.npy"
)


def _inference_vocabulary() -> torch.Tensor:
    return torch.zeros(8192, 40, 3)


def _request() -> InferenceInput:
    return InferenceInput(
        images={
            camera_id: np.zeros((1, 1, 3), dtype=np.uint8) for camera_id in CAMERA_IDS
        },
        command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
        velocity_xy=np.zeros(2, dtype=np.float32),
        acceleration_xy=np.zeros(2, dtype=np.float32),
    )


def _fake_policy(trajectory: torch.Tensor) -> GTRSDensePolicy:
    class FakeModel:
        def __call__(
            self, features: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            assert set(features) == {"camera_feature", "status_feature"}
            assert torch.is_inference_mode_enabled()
            return {"trajectory": trajectory}

    policy = object.__new__(GTRSDensePolicy)
    policy.device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.model = FakeModel()
    return policy


def test_policy_exposes_only_fp32_inference() -> None:
    assert "use_autocast" not in inspect.signature(GTRSDensePolicy).parameters


def _real_checkpoint_path() -> Path:
    checkpoint_path = os.environ.get("GTRS_REAL_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("set GTRS_REAL_CHECKPOINT to run released checkpoint tests")
    return Path(checkpoint_path)


def _expert_checkpoint_path() -> Path:
    checkpoint_path = os.environ.get("GTRS_EXPERT_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("set GTRS_EXPERT_CHECKPOINT to run expert tests")
    return Path(checkpoint_path)


def _incompatible_checkpoint_path() -> Path:
    checkpoint_path = os.environ.get("GTRS_INCOMPATIBLE_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("set GTRS_INCOMPATIBLE_CHECKPOINT to run rejection tests")
    return Path(checkpoint_path)


def test_normalized_checkpoint_strips_exact_prefix_and_merges_vocab() -> None:
    checkpoint_vocabulary = torch.zeros(16384, 40, 3)
    inference_vocabulary = _inference_vocabulary()
    payload = {
        "state_dict": {
            f"{CHECKPOINT_PREFIX}layer.weight": torch.ones(1),
            VOCABULARY_KEY: checkpoint_vocabulary,
        }
    }

    normalized = normalized_checkpoint(payload, inference_vocabulary)

    assert set(normalized) == {"layer.weight", "_trajectory_head.vocab"}
    assert normalized["_trajectory_head.vocab"] is inference_vocabulary


def test_load_navhard_vocabulary_requires_release_identity() -> None:
    vocabulary = policy_module.load_navhard_vocabulary(
        Path(__file__).resolve().parents[1] / "assets/gtrs_dense/navhard_8192.npy"
    )

    assert vocabulary.shape == (8192, 40, 3)
    assert vocabulary.dtype == torch.float32
    assert torch.isfinite(vocabulary).all()


def test_load_release_vocabulary_requires_release_identity() -> None:
    vocabulary = policy_module.load_navhard_vocabulary(RELEASE_VOCABULARY_PATH)

    assert vocabulary.shape == (16384, 40, 3)
    assert vocabulary.dtype == torch.float32
    assert torch.isfinite(vocabulary).all()


def test_load_official_vocabulary_rejects_unknown_size(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.npy"
    path.write_bytes(b"not-an-official-vocabulary")

    with pytest.raises(ValueError, match="official vocabulary size"):
        policy_module.load_navhard_vocabulary(path)


def test_load_official_vocabulary_rejects_wrong_digest(tmp_path: Path) -> None:
    path = tmp_path / "wrong-digest.npy"
    path.write_bytes(bytes(3_932_288))

    with pytest.raises(ValueError, match="official vocabulary SHA256"):
        policy_module.load_navhard_vocabulary(path)


def test_normalized_checkpoint_replaces_persisted_vocabulary() -> None:
    checkpoint_vocab = torch.ones(16384, 40, 3)
    inference_vocab = torch.zeros(8192, 40, 3)
    payload = {
        "state_dict": {
            f"{CHECKPOINT_PREFIX}layer.weight": torch.ones(1),
            VOCABULARY_KEY: checkpoint_vocab,
        }
    }

    normalized = normalized_checkpoint(payload, inference_vocab)

    assert normalized["_trajectory_head.vocab"] is inference_vocab
    assert normalized["_trajectory_head.vocab"].shape == (8192, 40, 3)


def test_normalized_checkpoint_accepts_release_vocabulary() -> None:
    checkpoint_vocab = torch.ones(16384, 40, 3)
    inference_vocab = torch.zeros(16384, 40, 3)
    payload = {
        "state_dict": {
            f"{CHECKPOINT_PREFIX}layer.weight": torch.ones(1),
            VOCABULARY_KEY: checkpoint_vocab,
        }
    }

    normalized = normalized_checkpoint(payload, inference_vocab)

    assert normalized["_trajectory_head.vocab"] is inference_vocab
    assert normalized["_trajectory_head.vocab"].shape == (16384, 40, 3)
    torch.testing.assert_close(checkpoint_vocab, torch.ones_like(checkpoint_vocab))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"state_dict": {}, "epoch": 1},
        {"state_dict": []},
    ],
)
def test_normalized_checkpoint_requires_strict_wrapper(payload: Any) -> None:
    with pytest.raises(ValueError):
        normalized_checkpoint(payload, _inference_vocabulary())


def test_normalized_checkpoint_rejects_wrong_prefix() -> None:
    with pytest.raises(ValueError, match="unexpected checkpoint key: weight"):
        normalized_checkpoint(
            {
                "state_dict": {
                    "weight": torch.ones(1),
                    VOCABULARY_KEY: torch.zeros(16384, 40, 3),
                }
            },
            _inference_vocabulary(),
        )


def test_normalized_checkpoint_rejects_non_tensor() -> None:
    with pytest.raises(ValueError, match="must be a tensor"):
        normalized_checkpoint(
            {
                "state_dict": {
                    f"{CHECKPOINT_PREFIX}metadata": "bad",
                    VOCABULARY_KEY: torch.zeros(16384, 40, 3),
                }
            },
            _inference_vocabulary(),
        )


@pytest.mark.parametrize(
    "state_dict",
    [
        {f"{CHECKPOINT_PREFIX}weight": torch.ones(1)},
        {VOCABULARY_KEY: torch.zeros(4, 8, 3)},
        {VOCABULARY_KEY: torch.zeros(4, 40, 2)},
        {VOCABULARY_KEY: torch.zeros(4, 40, 3, dtype=torch.int64)},
    ],
)
def test_normalized_checkpoint_rejects_invalid_vocabulary(
    state_dict: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        normalized_checkpoint(
            {"state_dict": state_dict},
            _inference_vocabulary(),
        )


def test_policy_strictly_loads_synthetic_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyModel(nn.Module):
        def __init__(self, vocab: torch.Tensor) -> None:
            super().__init__()
            self.layer = nn.Linear(2, 1, bias=False)
            self._trajectory_head = nn.Module()
            self._trajectory_head.vocab = nn.Parameter(vocab, requires_grad=False)

        def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            batch = features["status_feature"].shape[0]
            return {"trajectory": self._trajectory_head.vocab[:batch]}

    checkpoint_vocabulary = torch.zeros(16384, 40, 3)
    source = TinyModel(checkpoint_vocabulary)

    def create_tiny_model(
        config: object,
        *,
        vocab: torch.Tensor,
        scorer_mode: str = "release",
        ep_exponent: float = 1.0,
        speed_top_k: int = 0,
        speed_weight: float = 0.0,
        speed_proxy: str = "longitudinal",
        curvature_weight: float = 0.0,
        heading_change_weight: float = 0.0,
    ) -> TinyModel:
        del (
            config,
            scorer_mode,
            ep_exponent,
            speed_top_k,
            speed_weight,
            speed_proxy,
            curvature_weight,
            heading_change_weight,
        )
        return TinyModel(vocab)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": {
                f"{CHECKPOINT_PREFIX}{key}": value.clone()
                for key, value in source.state_dict().items()
            }
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        policy_module,
        "GTRSDenseModel",
        create_tiny_model,
    )

    policy = GTRSDensePolicy(
        checkpoint_path,
        NAVHARD_VOCABULARY_PATH,
        device="cpu",
        warm_up=False,
    )

    assert policy.state_dict_count == 2
    torch.testing.assert_close(policy.model.layer.weight, source.layer.weight)
    assert policy.model._trajectory_head.vocab.shape == (8192, 40, 3)


def test_policy_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA device"):
        GTRSDensePolicy(
            "unused.ckpt",
            "unused.npy",
            device="cuda",
            warm_up=False,
        )


def test_inference_input_rejects_missing_camera() -> None:
    with pytest.raises(ValueError, match="missing required cameras: CAM_R0"):
        InferenceInput(
            images={
                "CAM_L0": np.zeros((1, 1, 3), dtype=np.uint8),
                "CAM_F0": np.zeros((1, 1, 3), dtype=np.uint8),
            },
            command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
            velocity_xy=np.zeros(2, dtype=np.float32),
            acceleration_xy=np.zeros(2, dtype=np.float32),
        )


def test_predict_batch_returns_empty_without_initialized_model() -> None:
    policy = object.__new__(GTRSDensePolicy)

    assert policy.predict_batch([]) == []


def test_predict_batch_returns_independent_40_pose_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = torch.arange(240, dtype=torch.float32).reshape(2, 40, 3)
    policy = _fake_policy(trajectory)
    monkeypatch.setattr(
        policy_module,
        "preprocess_images",
        lambda images, dtype: torch.zeros((3, 2, 2), dtype=dtype),
    )
    monkeypatch.setattr(
        policy_module,
        "build_status_feature",
        lambda **kwargs: torch.zeros(8, dtype=torch.float32),
    )

    predictions = policy.predict_batch([_request(), _request()])

    assert len(predictions) == 2
    np.testing.assert_array_equal(predictions[0].trajectory, trajectory[0].numpy())
    assert not np.shares_memory(predictions[0].trajectory, predictions[1].trajectory)


def test_predict_batch_rejects_wrong_trajectory_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _fake_policy(torch.zeros((1, 8, 3)))
    monkeypatch.setattr(
        policy_module,
        "preprocess_images",
        lambda images, dtype: torch.zeros((3, 2, 2), dtype=dtype),
    )
    monkeypatch.setattr(
        policy_module,
        "build_status_feature",
        lambda **kwargs: torch.zeros(8, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match=r"\(1, 40, 3\)"):
        policy.predict_batch([_request()])


def test_predict_batch_rejects_non_finite_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = torch.zeros((1, 40, 3))
    trajectory[0, 0, 0] = torch.nan
    policy = _fake_policy(trajectory)
    monkeypatch.setattr(
        policy_module,
        "preprocess_images",
        lambda images, dtype: torch.zeros((3, 2, 2), dtype=dtype),
    )
    monkeypatch.setattr(
        policy_module,
        "build_status_feature",
        lambda **kwargs: torch.zeros(8, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="non-finite"):
        policy.predict_batch([_request()])


def test_real_checkpoint_artifact_contract() -> None:
    checkpoint_path = _real_checkpoint_path()

    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size == REAL_CHECKPOINT_SIZE
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == REAL_CHECKPOINT_SHA256


def test_real_checkpoint_exactly_matches_resnet_model_state() -> None:
    payload = torch.load(_real_checkpoint_path(), map_location="cpu", weights_only=True)
    vocab = policy_module.load_navhard_vocabulary(NAVHARD_VOCABULARY_PATH)
    state_dict = normalized_checkpoint(payload, vocab)
    model = GTRSDenseModel(
        GTRSDenseConfig(vocab_size=vocab.shape[0]),
        vocab=vocab,
    )
    model_state = model.state_dict()

    assert len(state_dict) == 765
    assert set(state_dict) == set(model_state)
    assert {key: tuple(value.shape) for key, value in state_dict.items()} == {
        key: tuple(value.shape) for key, value in model_state.items()
    }


def test_expert_checkpoint_artifact_contract() -> None:
    checkpoint_path = _expert_checkpoint_path()

    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size == EXPERT_CHECKPOINT_SIZE
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == EXPERT_CHECKPOINT_SHA256


def test_expert_checkpoint_exactly_matches_resnet_model_state() -> None:
    payload = torch.load(
        _expert_checkpoint_path(), map_location="cpu", weights_only=True
    )
    vocab = policy_module.load_navhard_vocabulary(NAVHARD_VOCABULARY_PATH)
    state_dict = normalized_checkpoint(payload, vocab)
    model = GTRSDenseModel(
        GTRSDenseConfig(vocab_size=vocab.shape[0]),
        vocab=vocab,
    )
    model_state = model.state_dict()

    assert len(state_dict) == 765
    assert set(state_dict) == set(model_state)
    assert {key: tuple(value.shape) for key, value in state_dict.items()} == {
        key: tuple(value.shape) for key, value in model_state.items()
    }
    model.load_state_dict(state_dict, strict=True)


def test_real_checkpoint_reduced_vocab_structural_cpu_smoke() -> None:
    policy = GTRSDensePolicy(
        _real_checkpoint_path(),
        NAVHARD_VOCABULARY_PATH,
        device="cpu",
        warm_up=False,
    )

    assert policy.state_dict_count == 765
    assert policy.model._trajectory_head.vocab.shape == (8192, 40, 3)

    # This is a reduced-vocabulary structural smoke, not full production inference.
    # Full-vocabulary self-attention is quadratic and unsuitable for a CPU test.
    policy.model._trajectory_head.vocab = nn.Parameter(
        policy.model._trajectory_head.vocab[:8].clone(),
        requires_grad=False,
    )
    with torch.inference_mode():
        output = policy.model(
            {
                "camera_feature": torch.zeros(1, 3, 32, 32),
                "status_feature": torch.zeros(1, 8),
            }
        )

    assert output["trajectory"].shape == (1, 40, 3)
    assert torch.isfinite(output["trajectory"]).all()
    assert torch.isfinite(output["scores"]).all()
    for name, value in output.items():
        if value.is_floating_point():
            assert torch.isfinite(value).all(), name


def test_policy_passes_experimental_scorer_mode_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class TinyPolicyModel(nn.Module):
        def __init__(
            self,
            config: object,
            *,
            vocab: torch.Tensor,
            scorer_mode: str,
            ep_exponent: float,
            speed_top_k: int,
            speed_weight: float,
            speed_proxy: str,
            curvature_weight: float,
            heading_change_weight: float,
        ):
            super().__init__()
            captured["backbone_type"] = config.backbone_type
            captured["scorer_mode"] = scorer_mode
            captured["ep_exponent"] = ep_exponent
            captured["speed_top_k"] = speed_top_k
            captured["speed_weight"] = speed_weight
            captured["speed_proxy"] = speed_proxy
            captured["curvature_weight"] = curvature_weight
            captured["heading_change_weight"] = heading_change_weight
            self.layer = nn.Linear(1, 1)

        def load_state_dict(self, state_dict: object, *, strict: bool) -> object:
            return super().load_state_dict({}, strict=False)

    monkeypatch.setattr(policy_module, "GTRSDenseModel", TinyPolicyModel)
    monkeypatch.setattr(
        policy_module,
        "normalized_checkpoint",
        lambda payload, vocabulary: {},
    )
    monkeypatch.setattr(
        policy_module,
        "load_navhard_vocabulary",
        lambda path: torch.zeros(8192, 40, 3),
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})

    GTRSDensePolicy(
        "unused.ckpt",
        "unused.npy",
        device="cpu",
        scorer_mode="nc_dac_ep",
        ep_exponent=10.0,
        speed_top_k=32,
        speed_weight=0.1,
        speed_proxy="path_length",
        curvature_weight=0.05,
        heading_change_weight=0.05,
        warm_up=False,
        backbone_type="vov",
    )

    assert captured["backbone_type"] == "vov"
    assert captured["scorer_mode"] == "nc_dac_ep"
    assert captured["ep_exponent"] == 10.0
    assert captured["speed_top_k"] == 32
    assert captured["speed_weight"] == pytest.approx(0.1)
    assert captured["speed_proxy"] == "path_length"
    assert captured["curvature_weight"] == pytest.approx(0.05)
    assert captured["heading_change_weight"] == pytest.approx(0.05)


def test_vov_reward_checkpoint_artifact_contract() -> None:
    checkpoint_path = _incompatible_checkpoint_path()

    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size == VOV_REWARD_CHECKPOINT_SIZE
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == VOV_REWARD_CHECKPOINT_SHA256


def test_incompatible_vov_checkpoint_is_rejected_by_resnet_policy() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        GTRSDensePolicy(
            _incompatible_checkpoint_path(),
            NAVHARD_VOCABULARY_PATH,
            device="cpu",
            warm_up=False,
            backbone_type="resnet",
        )

    error = str(exc_info.value)
    assert "Missing key(s) in state_dict" in error
    assert '"_backbone.image_encoder.conv1.weight"' in error
    assert "Unexpected key(s) in state_dict" in error
    assert '"_backbone.image_encoder.stem.stem_1/conv.weight"' in error
