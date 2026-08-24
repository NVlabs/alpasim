# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import navsim_transfuser_challenge.policy as policy_module
import numpy as np
import pytest
import torch
from navsim_transfuser_challenge.policy import (
    CHECKPOINT_PREFIX,
    InferenceInput,
    LtfPolicy,
    load_checkpoint,
    normalized_state_dict,
)
from navsim_transfuser_challenge.preprocessing import CAMERA_IDS
from torch import nn

REAL_CHECKPOINT_SIZE = 224560669
REAL_CHECKPOINT_SHA256 = (
    "9c1a17651bb2cd8e2edf006ea45634432c38554a8f44e0714f64d11ea31f2c69"
)


def _prefixed_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{CHECKPOINT_PREFIX}{key}": value.clone()
        for key, value in model.state_dict().items()
    }


def _request() -> InferenceInput:
    return InferenceInput(
        images={
            camera_id: np.zeros((1, 1, 3), dtype=np.uint8) for camera_id in CAMERA_IDS
        },
        command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
        velocity_xy=np.zeros(2, dtype=np.float32),
        acceleration_xy=np.zeros(2, dtype=np.float32),
    )


def _fake_policy(trajectory: torch.Tensor) -> LtfPolicy:
    class FakeModel:
        def __call__(
            self, features: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            assert set(features) == {"camera_feature", "status_feature"}
            assert torch.is_inference_mode_enabled()
            return {"trajectory": trajectory}

    policy = object.__new__(LtfPolicy)
    policy.device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.use_autocast = False
    policy.model = FakeModel()
    return policy


def _real_checkpoint_path() -> Path:
    checkpoint_path = os.environ.get("LTF_REAL_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("set LTF_REAL_CHECKPOINT to run released checkpoint tests")
    return Path(checkpoint_path)


def _real_request() -> InferenceInput:
    return InferenceInput(
        images={
            camera_id: np.zeros((1080, 1920, 3), dtype=np.uint8)
            for camera_id in CAMERA_IDS
        },
        command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
        velocity_xy=np.zeros(2, dtype=np.float32),
        acceleration_xy=np.zeros(2, dtype=np.float32),
    )


def test_normalized_state_dict_strips_exact_prefix() -> None:
    tensor = torch.tensor([1.0])

    normalized = normalized_state_dict(
        {"state_dict": {f"{CHECKPOINT_PREFIX}weight": tensor}}
    )

    assert normalized == {"weight": tensor}


def test_normalized_state_dict_rejects_wrong_prefix() -> None:
    with pytest.raises(ValueError, match="^unexpected checkpoint key: weight$"):
        normalized_state_dict({"state_dict": {"weight": torch.tensor([1.0])}})


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"state_dict": {}, "epoch": 1},
        {"state_dict": []},
    ],
)
def test_normalized_state_dict_requires_strict_wrapper(payload: Any) -> None:
    with pytest.raises(ValueError):
        normalized_state_dict(payload)


def test_load_checkpoint_strict_success_returns_key_count(tmp_path: Any) -> None:
    source = nn.Linear(2, 1, bias=False)
    target = nn.Linear(2, 1, bias=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": _prefixed_state_dict(source)}, checkpoint_path)

    count = load_checkpoint(target, checkpoint_path)

    assert count == 1
    torch.testing.assert_close(target.weight, source.weight)


def test_load_checkpoint_reports_missing_bias(tmp_path: Any) -> None:
    model = nn.Linear(2, 1)
    checkpoint_path = tmp_path / "missing-bias.pt"
    torch.save(
        {
            "state_dict": {
                f"{CHECKPOINT_PREFIX}weight": model.weight.detach().clone(),
            }
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="Missing key.*bias"):
        load_checkpoint(model, checkpoint_path)


def test_load_checkpoint_reports_shape_mismatch(tmp_path: Any) -> None:
    model = nn.Linear(2, 1)
    state_dict = _prefixed_state_dict(model)
    state_dict[f"{CHECKPOINT_PREFIX}weight"] = torch.zeros(2, 2)
    checkpoint_path = tmp_path / "shape-mismatch.pt"
    torch.save({"state_dict": state_dict}, checkpoint_path)

    with pytest.raises(RuntimeError, match="size mismatch.*weight"):
        load_checkpoint(model, checkpoint_path)


def test_inference_input_rejects_missing_camera() -> None:
    with pytest.raises(ValueError, match="^missing required cameras: CAM_R0$"):
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
    policy = object.__new__(LtfPolicy)

    assert policy.predict_batch([]) == []


def test_predict_batch_returns_independent_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = torch.arange(48, dtype=torch.float32).reshape(2, 8, 3)
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
    policy = _fake_policy(torch.zeros((1, 7, 3)))
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

    with pytest.raises(ValueError, match="model trajectory must have shape"):
        policy.predict_batch([_request()])


def test_predict_batch_rejects_non_finite_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = torch.zeros((1, 8, 3))
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


@pytest.mark.parametrize("batch_size", [1, 2])
def test_real_checkpoint_predicts_finite_trajectories(batch_size: int) -> None:
    checkpoint_path = _real_checkpoint_path()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for released checkpoint forward tests")

    policy = LtfPolicy(checkpoint_path, device="cuda", warm_up=False)
    assert policy.state_dict_count == 669

    predictions = policy.predict_batch([_real_request() for _ in range(batch_size)])

    assert len(predictions) == batch_size
    for prediction in predictions:
        assert prediction.trajectory.shape == (8, 3)
        assert np.isfinite(prediction.trajectory).all()
