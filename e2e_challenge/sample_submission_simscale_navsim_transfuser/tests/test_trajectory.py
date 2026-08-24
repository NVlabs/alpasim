# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import math

import numpy as np
import pytest
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Vec3
from navsim_transfuser_challenge.trajectory import (
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
    prediction = np.zeros((8, 3), dtype=np.float64)
    prediction[:, 0] = np.arange(1.0, 9.0)
    prediction[:, 2] = (
        np.arange(8, dtype=np.float64) * 0.1 if headings is None else headings
    )
    return prediction


def test_make_cached_plan_uses_x_forward_y_left_without_flipping_y() -> None:
    anchor = _pose_at(1_000_000)
    prediction = _forward_prediction()
    prediction[0, 1] = 2.0

    plan = make_cached_plan(1_000_000, anchor, prediction)

    assert plan is not None
    np.testing.assert_allclose(plan.positions_xy[0], [10.0, 20.0])
    # At yaw pi/2, rig +x maps to local +y and rig +y maps to local -x.
    np.testing.assert_allclose(plan.positions_xy[1], [8.0, 21.0], atol=1e-7)
    np.testing.assert_allclose(plan.times_s, np.arange(9) * 0.5)


def test_cached_plan_uses_stable_absolute_clock_and_only_shrinks() -> None:
    anchor = _pose_at(1_000_000)
    plan = make_cached_plan(1_000_000, anchor, _forward_prediction())
    assert plan is not None

    first = build_trajectory_from_plan(
        plan,
        anchor,
        2_000_000,
        2_500_000,
    )
    second = build_trajectory_from_plan(
        plan,
        anchor,
        2_500_000,
        3_000_000,
    )

    assert first.poses[0].timestamp_us == 2_000_000
    assert first.poses[0].pose.vec.y == pytest.approx(22.0)
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


def test_missing_plan_fallback_is_strictly_increasing_and_covers_long_query() -> None:
    time_now_us = 2_000_000
    time_query_us = 7_500_000

    trajectory = build_trajectory_from_plan(
        None,
        None,
        time_now_us,
        time_query_us,
    )

    timestamps = np.array([pose.timestamp_us for pose in trajectory.poses])
    assert timestamps[0] == time_now_us
    assert np.all(np.diff(timestamps) > 0)
    assert timestamps[-1] >= time_query_us


def test_yaw_interpolation_is_continuous_across_pi() -> None:
    anchor_yaw = 3.05
    anchor = _pose_at(1_000_000, yaw=anchor_yaw)
    wrapped_absolute_yaws = np.array(
        [3.10, 3.13, -3.12, -3.09, -3.06, -3.03, -3.00, -2.97],
        dtype=np.float64,
    )
    prediction = _forward_prediction(wrapped_absolute_yaws - anchor_yaw)
    plan = make_cached_plan(1_000_000, anchor, prediction)
    assert plan is not None
    assert plan.yaws[2] > 3.0
    assert plan.yaws[3] < -3.0

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


@pytest.mark.parametrize("shape", [(8, 2), (7, 3), (8, 3, 1)])
def test_make_cached_plan_rejects_wrong_prediction_shape(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError) as error:
        make_cached_plan(1_000_000, _pose_at(1_000_000), np.zeros(shape))
    assert str(error.value) == f"LTF prediction must have shape (8, 3); got {shape}"


def test_plan_ending_before_query_falls_back_and_covers_query() -> None:
    anchor = _pose_at(1_000_000)
    plan = make_cached_plan(1_000_000, anchor, _forward_prediction())
    assert plan is not None
    time_now_us = 2_000_000
    time_query_us = 5_100_000

    trajectory = build_trajectory_from_plan(
        plan,
        anchor,
        time_now_us,
        time_query_us,
    )

    timestamps = np.array([pose.timestamp_us for pose in trajectory.poses])
    assert timestamps[0] == time_now_us
    assert np.all(np.diff(timestamps) > 0)
    assert timestamps[-1] >= time_query_us
