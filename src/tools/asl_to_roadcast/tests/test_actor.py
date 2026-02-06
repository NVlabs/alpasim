# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_utils.qvec import QVec
from asl_to_roadcast.actor import Actor
from maglev.av.av_py_proto_pb.matrix_types_pb2 import Vector3f


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "id": "zero_acceleration",
            "accel": 0.0,
        },
        {
            "id": "positive_acceleration",
            "accel": 2.0,
        },
    ],
    ids=lambda tc: tc["id"],
)
def test_Actor_capture_pose(test_case):
    actor = Actor(extent=Vector3f(x=4.0, y=3.0, z=2.0), id=1, is_ego=True)
    DT_S = 0.1
    VELOCITY_0 = 10.0
    for i in range(10):
        timestamp_s = i * DT_S
        timestamp_us = int(timestamp_s * 1e6)
        pose = QVec(
            vec3=np.array(
                [
                    VELOCITY_0 * timestamp_s
                    + 0.5 * test_case["accel"] * timestamp_s**2,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            quat=np.array(
                [0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        )
        state_ready = actor.capture_pose(timestamp_us, pose)
        if i < 2:
            assert not state_ready
        elif i >= 2:
            assert state_ready
            assert actor.linear_velocity_mps_at_time_us[
                timestamp_us
            ].x == pytest.approx(
                VELOCITY_0 + test_case["accel"] * (timestamp_s - DT_S / 2.0), abs=0.1
            )
            assert actor.linear_acceleration_mps2_at_time_us[
                timestamp_us
            ].x == pytest.approx(test_case["accel"], abs=0.1)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "id": "zero_linear_velocity",
            "local_to_actor": {
                1000000001000000: QVec(
                    vec3=np.array([4.0, 0, 0]), quat=np.array([0, 0, 0, 1])
                ),
                1000000002000000: QVec(
                    vec3=np.array([4.0, 0, 0]), quat=np.array([0, 0, 0, 1])
                ),
                1000000003000000: QVec(
                    vec3=np.array([4.0, 0, 0]), quat=np.array([0, 0, 0, 1])
                ),
                1000000004000000: QVec(
                    vec3=np.array([4.0, 0, 0]), quat=np.array([0, 0, 0, 1])
                ),
                1000000005000000: QVec(
                    vec3=np.array([4.0, 0, 0]), quat=np.array([0, 0, 0, 1])
                ),
                1000000006000000: QVec(
                    vec3=np.array([4.0, 0, 0]), quat=np.array([0, 0, 0, 1])
                ),
            },
            "expected_velocity_for_ego": Vector3f(x=0.0, y=0.0, z=0.0),
            "expected_velocity_for_contenders": Vector3f(x=0.0, y=0.0, z=0.0),
        },
        {
            "id": "non_zero_velocity_all_directions",
            "local_to_actor": {
                1000000001000000: QVec(
                    vec3=np.array([0, 0, 0]), quat=np.array([0, 0, 0, 1])
                ),
                1000000002000000: QVec(
                    vec3=np.array([1.0, 2.0, 30]), quat=np.array([0, 0, 0, 1])
                ),
                1000000003000000: QVec(
                    vec3=np.array([3.0, 4.0, 5.0]), quat=np.array([0, 0, 0, 1])
                ),
                1000000004000000: QVec(
                    vec3=np.array([5.0, 6.0, 7.0]), quat=np.array([0, 0, 0, 1])
                ),
                1000000005000000: QVec(
                    vec3=np.array([7.0, 8.0, 9.0]), quat=np.array([0, 0, 0, 1])
                ),
                1000000006000000: QVec(
                    vec3=np.array([9.0, 10.0, 11.0]), quat=np.array([0, 0, 0, 1])
                ),
            },
            "expected_velocity_for_ego": Vector3f(x=2.0, y=2.0, z=2.0),
            "expected_velocity_for_contenders": Vector3f(x=2.0, y=2.0, z=2.0),
        },
    ],
    ids=lambda tc: tc["id"],
)
@pytest.mark.parametrize(
    "is_ego_case",
    [
        {"id": "ego_actor", "is_ego": True},
        {"id": "non_ego_actor", "is_ego": False},
    ],
    ids=lambda tc: tc["id"],
)
def test_calc_linear_velocity(test_case, is_ego_case, compare_vector3f):
    current_ts = 1000000006000000
    prev_ts = 1000000003000000

    actor = Actor(
        extent=Vector3f(x=4.0, y=3.0, z=2.0),
        id=1,
        is_ego=is_ego_case["is_ego"],
    )
    actor.local_to_actor_at_time_us = test_case["local_to_actor"]

    actor._calc_linear_velocity(current_ts, prev_ts)
    calculated_velocity = actor.linear_velocity_mps_at_time_us[current_ts]

    expected = (
        test_case["expected_velocity_for_ego"]
        if is_ego_case["is_ego"]
        else test_case["expected_velocity_for_contenders"]
    )
    compare_vector3f(calculated_velocity, expected)
