# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared fixtures for event-based simulation loop tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from alpasim_grpc.v0.common_pb2 import DynamicState, Vec3
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.config import PhysicsUpdateMode
from alpasim_runtime.delay_buffer import DelayBuffer
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_runtime.services.controller_service import (
    ControllerService,
    PropagatedPosesAtTime,
)
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_runtime.unbound_rollout import UnboundRollout
from alpasim_utils.geometry import DynamicTrajectory, Pose, Trajectory
from alpasim_utils.scenario import AABB, TrafficObjects


def _identity_pose() -> Pose:
    return Pose(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def _make_trajectory(timestamps_us: list[int]) -> Trajectory:
    """Create a simple straight-line trajectory at origin."""
    poses = [
        Pose(
            np.array([float(t) / 1e6, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        for t in timestamps_us
    ]
    return Trajectory.from_poses(
        timestamps=np.array(timestamps_us, dtype=np.uint64),
        poses=poses,
    )


@pytest.fixture
def identity_pose() -> Pose:
    return _identity_pose()


@pytest.fixture
def simple_trajectory() -> Trajectory:
    """A trajectory spanning 0 to 1_000_000 us with 11 points (every 100_000 us)."""
    timestamps = list(range(0, 1_100_000, 100_000))
    return _make_trajectory(timestamps)


@pytest.fixture
def mock_driver() -> AsyncMock:
    driver = AsyncMock(spec=DriverService)
    driver.skip = False
    driver.submit_image = AsyncMock()
    driver.submit_trajectory = AsyncMock()
    driver.submit_route = AsyncMock()
    driver.submit_recording_ground_truth = AsyncMock()
    driver.drive = AsyncMock()
    return driver


@pytest.fixture
def mock_sensorsim() -> AsyncMock:
    sensorsim = AsyncMock(spec=SensorsimService)
    sensorsim.skip = False
    sensorsim.render = AsyncMock()
    return sensorsim


@pytest.fixture
def mock_physics() -> AsyncMock:
    physics = AsyncMock(spec=PhysicsService)
    physics.skip = False
    physics.ground_intersection = AsyncMock()
    return physics


@pytest.fixture
def mock_trafficsim() -> AsyncMock:
    trafficsim = AsyncMock(spec=TrafficService)
    trafficsim.skip = False
    trafficsim.simulate_traffic = AsyncMock()
    return trafficsim


@pytest.fixture
def mock_controller(simple_trajectory: Trajectory) -> AsyncMock:
    controller = AsyncMock(spec=ControllerService)
    controller.skip = False

    # Default return value: list[PropagatedPosesAtTime] at a single timestamp.
    # Conversion to DynamicTrajectory happens in run_controller().
    fallback_pose = simple_trajectory.last_pose
    controller.run_controller_and_vehicle.return_value = [
        PropagatedPosesAtTime(
            timestamp_us=1_000_000,
            pose_local_to_rig=fallback_pose,
            pose_local_to_rig_estimate=fallback_pose,
            dynamic_state=DynamicState(
                linear_velocity=Vec3(x=0, y=0, z=0),
                angular_velocity=Vec3(x=0, y=0, z=0),
            ),
            dynamic_state_estimated=DynamicState(
                linear_velocity=Vec3(x=0, y=0, z=0),
                angular_velocity=Vec3(x=0, y=0, z=0),
            ),
        )
    ]
    return controller


@pytest.fixture
def mock_broadcaster() -> AsyncMock:
    return AsyncMock(spec=MessageBroadcaster)


@pytest.fixture
def mock_unbound(simple_trajectory: Trajectory) -> MagicMock:
    """A mock UnboundRollout with sensible defaults."""
    unbound = MagicMock(spec=UnboundRollout)

    unbound.rollout_uuid = "test-uuid-1234"
    unbound.scene_id = "test-scene"
    unbound.gt_ego_trajectory = simple_trajectory
    unbound.n_sim_steps = 10
    unbound.start_timestamp_us = 0
    unbound.time_start_offset_us = 0
    unbound.control_timestep_us = 100_000
    unbound.control_timestamps_us = list(range(0, 1_100_000, 100_000))
    unbound.pose_reporting_interval_us = 0
    unbound.force_gt_duration_us = 200_000
    unbound.force_gt_period = range(0, 200_001)
    unbound.save_path_root = "/tmp/test"
    unbound.image_format = 2  # JPEG
    unbound.ego_mask_rig_config_id = "default"
    unbound.assert_zero_decision_delay = False
    unbound.planner_delay_us = 0
    unbound.send_recording_ground_truth = False
    unbound.route_generator_type = "NONE"
    unbound.physics_update_mode = PhysicsUpdateMode.NONE

    unbound.transform_ego_coords_ds_to_aabb = _identity_pose()
    unbound.ego_aabb = AABB(x=5.0, y=2.0, z=1.5)
    unbound.hidden_traffic_objs = None

    # Camera configs
    unbound.camera_configs = []

    return unbound


@pytest.fixture
def mock_traffic_objs() -> TrafficObjects:
    """Empty traffic objects."""
    return TrafficObjects()


@pytest.fixture
def service_bundle(
    mock_driver: AsyncMock,
    mock_controller: AsyncMock,
    mock_physics: AsyncMock,
    mock_trafficsim: AsyncMock,
    mock_broadcaster: AsyncMock,
) -> ServiceBundle:
    return ServiceBundle(
        driver=mock_driver,
        controller=mock_controller,
        physics=mock_physics,
        trafficsim=mock_trafficsim,
        broadcaster=mock_broadcaster,
        planner_delay_buffer=DelayBuffer(delay_us=0),
    )


@pytest.fixture
def rollout_state(
    mock_unbound: MagicMock,
    simple_trajectory: Trajectory,
    mock_traffic_objs: TrafficObjects,
) -> RolloutState:
    """A RolloutState with basic defaults for testing events."""
    # Use two control timestamps so the trajectory covers up to 200_000,
    # matching what EventBasedRollout.__post_init__ produces for [t0, t1].
    ego_traj = simple_trajectory.interpolate(
        np.array([0, 200_000], dtype=np.uint64),
    )
    zero_dynamics = np.zeros((len(ego_traj), 12), dtype=np.float64)

    ego_traj_estimate = simple_trajectory.interpolate(
        np.array([0, 200_000], dtype=np.uint64),
    )

    state = RolloutState(
        unbound=mock_unbound,
        ego_trajectory=DynamicTrajectory.from_trajectory_and_dynamics(
            ego_traj, zero_dynamics
        ),
        ego_trajectory_estimate=DynamicTrajectory.from_trajectory_and_dynamics(
            ego_traj_estimate, zero_dynamics
        ),
        traffic_objs=mock_traffic_objs,
    )
    state.step_context = StepContext()
    return state


@pytest.fixture
def runtime_camera() -> RuntimeCamera:
    """A default runtime camera for testing."""
    return RuntimeCamera(
        logical_id="cam_front",
        render_resolution_hw=(720, 1280),
        clock=Clock(
            interval_us=100_000,
            duration_us=33_000,
            start_us=0,
        ),
    )
