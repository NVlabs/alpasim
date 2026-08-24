# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import navsim_diffusiondrive_challenge.policy as policy_module
import numpy as np
import pytest
import torch
from navsim_diffusiondrive_challenge.policy import (
    ANCHOR_KEY,
    CHECKPOINT_PREFIX,
    EXPECTED_STATE_DICT_COUNT,
    DiffusionDrivePolicy,
    InferenceInput,
    normalized_checkpoint,
)
from navsim_diffusiondrive_challenge.preprocessing import CAMERA_IDS
from torch import nn

REAL_CHECKPOINT_SIZE = 243_596_717
REAL_CHECKPOINT_SHA256 = (
    "8fdbdb3fdfa7b496e7d7a438efbb5c2022377e59cbfd7095270d89623c5d963f"
)


class TinyReleaseModel(nn.Module):
    """Small model with the release checkpoint's exact tensor count."""

    def __init__(self, anchor: np.ndarray) -> None:
        super().__init__()
        self._trajectory_head = nn.Module()
        self._trajectory_head.plan_anchor = nn.Parameter(
            torch.from_numpy(anchor.copy()),
            requires_grad=False,
        )
        self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(381)])
        self.extra = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        features: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = features["status_feature"].shape[0]
        return torch.zeros((batch_size, 8, 3), device=features["status_feature"].device)


def _anchor() -> torch.Tensor:
    return torch.zeros((20, 8, 2), dtype=torch.float32)


def _release_state_dict(anchor: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    state_dict = {
        f"{CHECKPOINT_PREFIX}scalar_{index}": torch.zeros(())
        for index in range(EXPECTED_STATE_DICT_COUNT - 1)
    }
    state_dict[ANCHOR_KEY] = _anchor() if anchor is None else anchor
    return state_dict


def _request(
    *,
    noise_seed: int | None = None,
    noise_index: int | None = None,
) -> InferenceInput:
    return InferenceInput(
        images={
            camera_id: np.zeros((1, 1, 3), dtype=np.uint8) for camera_id in CAMERA_IDS
        },
        command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
        velocity_xy=np.zeros(2, dtype=np.float32),
        acceleration_xy=np.zeros(2, dtype=np.float32),
        noise_seed=noise_seed,
        noise_index=noise_index,
    )


def _fake_policy(trajectory: torch.Tensor) -> DiffusionDrivePolicy:
    class FakeModel:
        def __call__(
            self,
            features: dict[str, torch.Tensor],
            *,
            noise: torch.Tensor,
        ) -> torch.Tensor:
            assert set(features) == {"camera_feature", "status_feature"}
            assert noise.shape == (features["status_feature"].shape[0], 20, 8, 2)
            assert torch.is_inference_mode_enabled()
            return trajectory

    policy = object.__new__(DiffusionDrivePolicy)
    policy.device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.use_autocast = False
    policy.model = FakeModel()
    policy.generator = torch.Generator(device="cpu").manual_seed(7)
    return policy


def _real_checkpoint_path() -> Path:
    checkpoint_path = os.environ.get("DIFFUSIONDRIVE_REAL_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip(
            "set DIFFUSIONDRIVE_REAL_CHECKPOINT to run released checkpoint tests"
        )
    return Path(checkpoint_path)


def test_normalized_checkpoint_strips_prefix_and_copies_anchor() -> None:
    anchor = _anchor()

    normalized, actual_anchor = normalized_checkpoint(
        {"state_dict": _release_state_dict(anchor)}
    )

    assert len(normalized) == EXPECTED_STATE_DICT_COUNT
    assert "_trajectory_head.plan_anchor" in normalized
    np.testing.assert_array_equal(actual_anchor, anchor.numpy())
    anchor[0, 0, 0] = 1
    assert actual_anchor[0, 0, 0] == 0


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"state_dict": {}, "epoch": 1}, {"state_dict": []}],
)
def test_normalized_checkpoint_requires_strict_wrapper(payload: Any) -> None:
    with pytest.raises(ValueError):
        normalized_checkpoint(payload)


def test_normalized_checkpoint_rejects_wrong_prefix() -> None:
    state_dict = _release_state_dict()
    state_dict["weight"] = state_dict.pop(f"{CHECKPOINT_PREFIX}scalar_0")
    with pytest.raises(ValueError, match="unexpected checkpoint key: weight"):
        normalized_checkpoint({"state_dict": state_dict})


def test_normalized_checkpoint_rejects_non_tensor_value() -> None:
    state_dict: dict[str, object] = _release_state_dict()
    state_dict[f"{CHECKPOINT_PREFIX}scalar_0"] = "bad"
    with pytest.raises(ValueError, match="must be a tensor"):
        normalized_checkpoint({"state_dict": state_dict})


@pytest.mark.parametrize(
    ("anchor", "message"),
    [
        (torch.zeros((20, 8, 3)), "shape"),
        (torch.zeros((20, 8, 2), dtype=torch.int64), "floating-point"),
        (torch.full((20, 8, 2), torch.nan), "non-finite"),
    ],
)
def test_normalized_checkpoint_rejects_invalid_anchor(
    anchor: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalized_checkpoint({"state_dict": _release_state_dict(anchor)})


def test_normalized_checkpoint_rejects_wrong_tensor_count() -> None:
    state_dict = _release_state_dict()
    state_dict.pop(f"{CHECKPOINT_PREFIX}scalar_0")
    with pytest.raises(ValueError, match="764 tensors"):
        normalized_checkpoint({"state_dict": state_dict})


def test_policy_strictly_loads_synthetic_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = TinyReleaseModel(_anchor().numpy())
    assert len(source.state_dict()) == EXPECTED_STATE_DICT_COUNT
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
        "DiffusionDriveModel",
        lambda config, plan_anchor: TinyReleaseModel(plan_anchor),
    )

    policy = DiffusionDrivePolicy(
        checkpoint_path,
        device="cpu",
        use_autocast=False,
        warm_up=False,
    )

    assert policy.state_dict_count == EXPECTED_STATE_DICT_COUNT
    torch.testing.assert_close(policy.model.extra, source.extra)


def test_policy_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device"):
        DiffusionDrivePolicy("unused.ckpt", device="cuda", warm_up=False)


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
    policy = object.__new__(DiffusionDrivePolicy)
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


def test_predict_batch_keeps_diffusion_noise_in_float32_under_autocast_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_noise_dtypes: list[torch.dtype] = []

    class FakeModel:
        def __call__(
            self,
            features: dict[str, torch.Tensor],
            *,
            noise: torch.Tensor,
        ) -> torch.Tensor:
            assert features["camera_feature"].dtype == torch.float16
            observed_noise_dtypes.append(noise.dtype)
            return torch.zeros((1, 8, 3))

    policy = object.__new__(DiffusionDrivePolicy)
    policy.device = torch.device("cpu")
    policy.dtype = torch.float16
    policy.use_autocast = False
    policy.model = FakeModel()
    policy.generator = torch.Generator(device="cpu").manual_seed(7)
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

    policy.predict_batch([_request()])

    assert observed_noise_dtypes == [torch.float32]


def test_session_noise_is_independent_of_batching_and_interleaving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def make_policy() -> tuple[DiffusionDrivePolicy, list[torch.Tensor]]:
        captured: list[torch.Tensor] = []

        class CapturingModel:
            def __call__(
                self,
                features: dict[str, torch.Tensor],
                *,
                noise: torch.Tensor,
            ) -> torch.Tensor:
                captured.append(noise.detach().cpu().clone())
                return torch.zeros((noise.shape[0], 8, 3))

        policy = object.__new__(DiffusionDrivePolicy)
        policy.device = torch.device("cpu")
        policy.dtype = torch.float32
        policy.use_autocast = False
        policy.model = CapturingModel()
        policy.generator = torch.Generator(device="cpu").manual_seed(999)
        return policy, captured

    session_a = _request(noise_seed=11, noise_index=0)
    session_b = _request(noise_seed=22, noise_index=0)
    batched_policy, batched_noise = make_policy()
    reordered_policy, reordered_noise = make_policy()

    batched_policy.predict_batch([session_a, session_b])
    reordered_policy.predict_batch([session_b])
    reordered_policy.predict_batch([session_a])

    torch.testing.assert_close(batched_noise[0][0], reordered_noise[1][0])
    torch.testing.assert_close(batched_noise[0][1], reordered_noise[0][0])


@pytest.mark.parametrize(
    ("trajectory", "message"),
    [
        (torch.zeros((1, 7, 3)), "model trajectory must have shape"),
        (torch.full((1, 8, 3), torch.nan), "non-finite"),
    ],
)
def test_predict_batch_rejects_invalid_trajectory(
    trajectory: torch.Tensor,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    with pytest.raises(ValueError, match=message):
        policy.predict_batch([_request()])


def test_real_checkpoint_artifact_contract() -> None:
    checkpoint_path = _real_checkpoint_path()
    assert checkpoint_path.stat().st_size == REAL_CHECKPOINT_SIZE
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == REAL_CHECKPOINT_SHA256


def test_real_checkpoint_strict_cpu_load() -> None:
    policy = DiffusionDrivePolicy(
        _real_checkpoint_path(),
        device="cpu",
        use_autocast=False,
        warm_up=False,
    )
    assert policy.state_dict_count == EXPECTED_STATE_DICT_COUNT
    assert tuple(policy.model._trajectory_head.plan_anchor.shape) == (20, 8, 2)
