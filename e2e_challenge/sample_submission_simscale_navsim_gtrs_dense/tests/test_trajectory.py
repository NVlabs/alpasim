# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import math

import numpy as np
import pytest
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Vec3
from navsim_gtrs_dense_challenge.trajectory import (
    build_trajectory_from_plan,
    make_cached_plan,
    quat_from_yaw,
    yaw_from_quat,
)


def _pose_at(
    timestamp_us: int,
    *,
    x: float = 10.0,
    y: float = 20.0,
    z: float = 0.5,
    yaw: float = math.pi / 2,
) -> PoseAtTime:
    return PoseAtTime(
        timestamp_us=timestamp_us,
        pose=Pose(vec=Vec3(x=x, y=y, z=z), quat=quat_from_yaw(yaw)),
    )


def _forward_prediction(headings: np.ndarray | None = None) -> np.ndarray:
    prediction = np.zeros((40, 3), dtype=np.float64)
    prediction[:, 0] = np.arange(1.0, 41.0) * 0.1
    prediction[:, 2] = (
        np.arange(40, dtype=np.float64) * 0.01 if headings is None else headings
    )
    return prediction


def test_make_cached_plan_uses_40_predictions_at_10hz() -> None:
    anchor = _pose_at(1_000_000)
    prediction = _forward_prediction()
    prediction[0, 1] = 2.0

    plan = make_cached_plan(1_000_000, anchor, prediction)

    np.testing.assert_allclose(plan.positions_xy[0], [10.0, 20.0])
    np.testing.assert_allclose(plan.positions_xy[1], [8.0, 20.1], atol=1e-7)
    np.testing.assert_allclose(plan.times_s, np.arange(41) * 0.1)
    assert plan.positions_xy.shape == (41, 2)


def test_make_cached_plan_time_scale_is_exact_noop_at_one() -> None:
    anchor = _pose_at(1_000_000)
    prediction = _forward_prediction()

    default = make_cached_plan(1_000_000, anchor, prediction)
    explicit = make_cached_plan(
        1_000_000,
        anchor,
        prediction,
        trajectory_time_scale=1.0,
    )

    np.testing.assert_array_equal(explicit.times_s, default.times_s)
    np.testing.assert_array_equal(explicit.positions_xy, default.positions_xy)
    np.testing.assert_array_equal(explicit.yaws, default.yaws)


def test_make_cached_plan_time_scale_advances_near_term_and_preserves_endpoint() -> (
    None
):
    anchor = _pose_at(1_000_000)
    prediction = _forward_prediction()

    plan = make_cached_plan(
        1_000_000,
        anchor,
        prediction,
        trajectory_time_scale=1.10,
    )

    assert plan.positions_xy[10, 1] == pytest.approx(21.1)
    assert plan.positions_xy[20, 1] == pytest.approx(22.1)
    assert plan.positions_xy[35, 1] == pytest.approx(23.55)
    assert plan.positions_xy[40, 1] == pytest.approx(24.0)
    assert np.all(np.diff(plan.positions_xy[:, 1]) >= 0.0)
    np.testing.assert_array_equal(plan.times_s, np.arange(41) * 0.1)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0.99, 1.251])
def test_make_cached_plan_rejects_invalid_time_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="trajectory_time_scale"):
        make_cached_plan(
            1_000_000,
            _pose_at(1_000_000),
            _forward_prediction(),
            trajectory_time_scale=scale,
        )


def test_cached_plan_uses_stable_absolute_clock_and_only_shrinks() -> None:
    anchor = _pose_at(1_000_000)
    plan = make_cached_plan(1_000_000, anchor, _forward_prediction())

    first = build_trajectory_from_plan(plan, anchor, 2_000_000, 2_500_000)
    second = build_trajectory_from_plan(plan, anchor, 2_500_000, 3_000_000)

    assert first.poses[0].timestamp_us == 2_000_000
    assert first.poses[0].pose.vec.y == pytest.approx(21.0)
    assert first.poses[-1].timestamp_us == 5_000_000
    assert second.poses[0].timestamp_us == 2_500_000
    first_by_timestamp = {
        pose.timestamp_us: (pose.pose.vec.x, pose.pose.vec.y) for pose in first.poses
    }
    for pose in second.poses:
        np.testing.assert_allclose(
            (pose.pose.vec.x, pose.pose.vec.y),
            first_by_timestamp[pose.timestamp_us],
        )


def test_missing_plan_fallback_is_increasing_and_covers_long_query() -> None:
    trajectory = build_trajectory_from_plan(None, None, 2_000_000, 7_500_000)

    timestamps = np.array([pose.timestamp_us for pose in trajectory.poses])
    assert timestamps[0] == 2_000_000
    assert np.all(np.diff(timestamps) > 0)
    assert timestamps[-1] >= 7_500_000


def test_yaw_interpolation_is_continuous_across_pi() -> None:
    anchor_yaw = 3.05
    anchor = _pose_at(1_000_000, yaw=anchor_yaw)
    unwrapped = np.linspace(3.10, 3.60, 40)
    wrapped = np.angle(np.exp(1j * unwrapped))
    prediction = _forward_prediction(wrapped - anchor_yaw)
    plan = make_cached_plan(1_000_000, anchor, prediction)

    trajectory = build_trajectory_from_plan(
        plan,
        anchor,
        1_000_000,
        5_000_000,
    )

    decoded_yaws = np.unwrap(
        np.array([yaw_from_quat(pose.pose.quat) for pose in trajectory.poses])
    )
    assert np.max(np.abs(np.diff(decoded_yaws))) < 0.1


@pytest.mark.parametrize("shape", [(8, 3), (40, 2), (41, 3), (40, 3, 1)])
def test_make_cached_plan_rejects_wrong_prediction_shape(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError) as error:
        make_cached_plan(1_000_000, _pose_at(1_000_000), np.zeros(shape))
    assert str(error.value) == (f"GTRS prediction must have shape (40, 3); got {shape}")


def test_make_cached_plan_rejects_non_finite_prediction() -> None:
    prediction = _forward_prediction()
    prediction[4, 1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        make_cached_plan(1_000_000, _pose_at(1_000_000), prediction)


def test_plan_ending_before_query_falls_back_and_covers_query() -> None:
    anchor = _pose_at(1_000_000)
    plan = make_cached_plan(1_000_000, anchor, _forward_prediction())

    trajectory = build_trajectory_from_plan(
        plan,
        anchor,
        2_000_000,
        5_100_000,
    )

    timestamps = np.array([pose.timestamp_us for pose in trajectory.poses])
    assert timestamps[0] == 2_000_000
    assert np.all(np.diff(timestamps) > 0)
    assert timestamps[-1] >= 5_100_000
