# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from unittest.mock import AsyncMock

import numpy as np
import pytest
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_utils.geometry import Pose, Trajectory


@pytest.fixture
def default_args():

    pose = Pose(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    trajectory = Trajectory.from_poses(
        timestamps=np.array([0], dtype=np.uint64),
        poses=[pose],
    )

    args = {
        "session_uuid": "session_uuid",
        "now_us": 1000000,
        "pose_local_to_rig": Pose(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
        "rig_linear_velocity_in_rig": np.array([1.0, 0.0, 0.0]),
        "rig_angular_velocity_in_rig": np.array([0.0, 0.0, 0.0]),
        "rig_reference_trajectory_in_rig": trajectory,
        "future_us": 10000,
        "fallback_pose_local_to_rig_future": Pose(
            np.array([5.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
        "force_gt": False,
    }
    return args


def test_create_run_controller_and_vehicle_request(default_args):
    # Remove fallback_pose_local_to_rig_future and dispatcher as they're not needed for request creation
    args_for_request = default_args.copy()
    del args_for_request["fallback_pose_local_to_rig_future"]

    request = ControllerService.create_run_controller_and_vehicle_request(
        **args_for_request
    )
    # spot check
    assert request.session_uuid == default_args["session_uuid"]
    assert request.state.timestamp_us == default_args["now_us"]


async def test_skip_controller_instance_run_controller_and_vehicle(default_args):
    """
    Check that the ControllerService in skip mode returns the fallback pose
    """
    # Create a mock broadcaster
    mock_broadcaster = AsyncMock(spec=MessageBroadcaster)

    # Create controller service in skip mode
    controller = ControllerService(address="localhost:50051", skip=True)

    # Create a session with the controller
    async with controller.session(
        uuid=default_args["session_uuid"], broadcaster=mock_broadcaster
    ):
        # Remove session_uuid from args since it's now accessed via session_info
        args = default_args.copy()
        del args["session_uuid"]

        propagated_pose_pair = await controller.run_controller_and_vehicle(**args)
        pose = propagated_pose_pair.pose_local_to_rig
        pose_est = propagated_pose_pair.pose_local_to_rig_estimate
        expected_pose = default_args["fallback_pose_local_to_rig_future"]

        assert pose.vec3 == pytest.approx(expected_pose.vec3)
        assert pose.quat == pytest.approx(expected_pose.quat)

        assert pose_est.vec3 == pytest.approx(expected_pose.vec3)
        assert pose_est.quat == pytest.approx(expected_pose.quat)
