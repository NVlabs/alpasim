# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""The drive response carries the full pose of every waypoint the model plans."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from alpasim_driver.main import EgoDriverService
from alpasim_driver.models.base import ModelPrediction
from alpasim_utils.geometry import Pose, pose_from_grpc, pose_to_grpc_at_time

CURRENT_POSITION = np.array([10.0, 20.0, 30.0])
CURRENT_POSE = pose_to_grpc_at_time(
    Pose(CURRENT_POSITION, np.array([0.0, 0.0, 0.0, 1.0])), 1_000_000
)
TILTED_ROTATIONS = np.stack(
    [
        Pose.from_denormalized_quat(
            np.zeros(3), np.array([0.1, 0.2, 0.3, 1.0]) + 0.01 * step
        ).as_se3()[:3, :3]
        for step in range(2)
    ]
)
CANDIDATE_POSITIONS = np.array(
    [
        [[1.0, 0.0, 0.1], [2.0, 0.0, 0.2]],
        [[1.0, 0.5, -0.1], [2.0, 1.0, -0.2]],
    ]
)


def _prediction() -> ModelPrediction:
    return ModelPrediction(
        candidate_positions=CANDIDATE_POSITIONS,
        candidate_rotations=np.stack([TILTED_ROTATIONS, TILTED_ROTATIONS]),
        selected_index=1,
    )


def _service() -> EgoDriverService:
    service = EgoDriverService.__new__(EgoDriverService)
    service._model = SimpleNamespace(output_frequency_hz=10)
    service._trajectory_optimizer = None
    return service


def _positions(trajectory) -> np.ndarray:
    return np.array(
        [[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z] for p in trajectory.poses]
    )


def _rotations(trajectory) -> np.ndarray:
    return np.stack([pose_from_grpc(p.pose).as_se3()[:3, :3] for p in trajectory.poses])


def test_driven_trajectory_keeps_position_and_rotation_in_all_three_axes() -> None:
    trajectory = _service()._convert_prediction_to_alpasim_trajectory(
        _prediction(), CURRENT_POSE, 1_000_000
    )

    assert [p.timestamp_us for p in trajectory.poses] == [
        1_000_000,
        1_100_000,
        1_200_000,
    ]
    np.testing.assert_allclose(
        _positions(trajectory)[1:], CANDIDATE_POSITIONS[1] + CURRENT_POSITION, atol=1e-6
    )
    np.testing.assert_allclose(_rotations(trajectory)[1:], TILTED_ROTATIONS, atol=1e-6)


def test_every_sampled_candidate_is_reported() -> None:
    sampled = _service()._sampled_alpasim_trajectories(
        _prediction(), CURRENT_POSE, 1_000_000
    )

    assert len(sampled) == len(CANDIDATE_POSITIONS)
    for candidate, trajectory in zip(CANDIDATE_POSITIONS, sampled, strict=True):
        np.testing.assert_allclose(
            _positions(trajectory)[1:], candidate + CURRENT_POSITION, atol=1e-6
        )


def test_model_t0_pose_and_timestamps_override_newer_request_snapshot() -> None:
    model_t0_us = 900_000
    model_position = np.array([7.0, 8.0, 9.0])
    model_pose = Pose(model_position, np.array([0.0, 0.0, 0.0, 1.0]))
    waypoint_timestamps_us = np.array([1_000_000, 1_100_000], dtype=np.uint64)
    prediction = _prediction()
    prediction.model_t0_us = model_t0_us
    prediction.pose_local_to_rig_t0 = model_pose
    prediction.waypoint_timestamps_us = waypoint_timestamps_us

    trajectory = _service()._convert_prediction_to_alpasim_trajectory(
        prediction,
        CURRENT_POSE,
        1_200_000,
    )

    assert [pose.timestamp_us for pose in trajectory.poses] == [
        model_t0_us,
        *waypoint_timestamps_us,
    ]
    np.testing.assert_allclose(_positions(trajectory)[0], model_position, atol=1e-6)
    np.testing.assert_allclose(
        _positions(trajectory)[1:],
        CANDIDATE_POSITIONS[1] + model_position,
        atol=1e-6,
    )
    np.testing.assert_allclose(_rotations(trajectory)[1:], TILTED_ROTATIONS, atol=1e-6)

    sampled = _service()._sampled_alpasim_trajectories(
        prediction,
        CURRENT_POSE,
        1_200_000,
    )
    assert all(
        [pose.timestamp_us for pose in candidate.poses]
        == [model_t0_us, *waypoint_timestamps_us]
        for candidate in sampled
    )
    for candidate in sampled:
        np.testing.assert_allclose(
            _rotations(candidate)[1:], TILTED_ROTATIONS, atol=1e-6
        )
