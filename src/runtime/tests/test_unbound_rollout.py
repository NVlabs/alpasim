# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.config import (
    PhysicsUpdateMode,
    RuntimeCameraConfig,
    SimulationConfig,
    VehicleConfig,
)
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.unbound_rollout import UnboundRollout
from alpasim_utils.geometry import Pose, Trajectory
from alpasim_utils.scenario import AABB, CameraId, Rig, TrafficObject, TrafficObjects


def _pose_at_x(x: float) -> Pose:
    return Pose(
        np.array([x, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def _trajectory(timestamps_us: list[int]) -> Trajectory:
    return Trajectory.from_poses(
        timestamps=np.array(timestamps_us, dtype=np.uint64),
        poses=[_pose_at_x(float(ts) / 1e6) for ts in timestamps_us],
    )


def _artifact(timestamps_us: list[int] | None = None) -> SimpleNamespace:
    if timestamps_us is None:
        timestamps_us = [0, 100_000, 200_000, 300_000, 400_000, 500_000]
    rig = Rig(
        sequence_id="clipgt-test",
        trajectory=_trajectory(timestamps_us),
        camera_ids=[
            CameraId("camera_front", 0, "clipgt-test", "unique-front"),
            CameraId("camera_left", 0, "clipgt-test", "unique-left"),
        ],
        camera_frame_timestamps_us={
            "unique-front": [200_000, 300_000],
            "unique-left": [150_000, 250_000],
        },
        camera_frame_ranges_us={
            "unique-front": [range(170_000, 200_000), range(270_000, 300_000)],
            "unique-left": [range(120_000, 150_000), range(220_000, 250_000)],
        },
        world_to_nre=np.eye(4),
        vehicle_config=VehicleConfig(),
    )
    traffic_objects = TrafficObjects(
        actor=TrafficObject(
            track_id="actor",
            aabb=AABB(1.0, 1.0, 1.0),
            trajectory=_trajectory(timestamps_us),
            is_static=False,
            label_class="vehicle",
        )
    )
    metadata = SimpleNamespace(
        logger=SimpleNamespace(run_id="run-1"),
        version_string="0.0.0-test",
        uuid="artifact-uuid",
    )
    return SimpleNamespace(
        rig=rig,
        traffic_objects=traffic_objects,
        source="artifact.usdz",
        metadata=metadata,
        map=None,
    )


def _simulation_config(**overrides) -> SimulationConfig:
    cfg = SimulationConfig(
        n_sim_steps=2,
        n_rollouts=1,
        control_timestep_us=100_000,
        force_gt_duration_us=100_000,
        min_traffic_duration_us=0,
        cameras=[
            RuntimeCameraConfig(logical_id="camera_front"),
            RuntimeCameraConfig(logical_id="camera_left"),
        ],
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _sensorsim_renderer() -> SensorsimService:
    return SensorsimService(
        address="localhost:0",
        skip=True,
        camera_catalog=SimpleNamespace(),
    )


def test_create_uses_rig_start_for_context_and_closed_loop_after_force_gt(
    tmp_path,
) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(),
        scene_id="scene",
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.egomotion_context_start_us == 0
    assert rollout.render_start_timestamp_us == 150_000
    assert rollout.first_policy_timestamp_us == 150_000
    assert rollout.closed_loop_start_us == 250_000
    assert rollout.end_timestamp_us == 350_000
    assert rollout.get_log_metadata().start_timestamp_us == 0
    assert rollout.force_gt_period == range(150_000, 250_001)
    assert rollout.first_camera_frame_ranges_us["camera_front"] == range(
        170_000, 200_000
    )
    assert rollout.gt_ego_trajectory.time_range_us.start == 0
    assert rollout.traffic_objs["actor"].trajectory.time_range_us.start == 0


def test_create_clips_n_sim_steps_to_complete_policy_steps(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(n_sim_steps=10),
        scene_id="scene",
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.closed_loop_start_us == 250_000
    assert rollout.n_sim_steps == 3
    assert rollout.end_timestamp_us == 450_000
    assert rollout.get_log_metadata().n_sim_steps == 3


def test_create_allows_force_gt_over_entire_rollout(tmp_path) -> None:
    force_gt_duration_us = 100_000_000_000_000
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(
            force_gt_duration_us=force_gt_duration_us,
            n_sim_steps=2,
        ),
        scene_id="scene",
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.render_start_timestamp_us == 150_000
    assert rollout.first_policy_timestamp_us == 150_000
    assert rollout.closed_loop_start_us == 150_000 + force_gt_duration_us
    assert rollout.end_timestamp_us == 350_000
    assert rollout.closed_loop_start_us > rollout.end_timestamp_us
    assert rollout.n_sim_steps == 2
    assert 250_000 in rollout.force_gt_period
    assert 350_000 in rollout.force_gt_period


def test_create_synchronizes_cameras_for_zero_decision_delay(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(assert_zero_decision_delay=True),
        scene_id="scene",
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.first_camera_frame_ranges_us["camera_front"] == range(
        133_000, 150_000
    )
    assert rollout.first_camera_frame_ranges_us["camera_left"] == range(
        133_000, 150_000
    )
    assert rollout.render_start_timestamp_us == 150_000
    assert rollout.first_policy_timestamp_us == 150_000
    assert rollout.closed_loop_start_us == 250_000


def test_create_allows_physics_with_zero_force_gt_duration(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(
            force_gt_duration_us=0,
            physics_update_mode=PhysicsUpdateMode.EGO_ONLY,
        ),
        scene_id="scene",
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.render_start_timestamp_us == 150_000
    assert rollout.first_policy_timestamp_us == 150_000
    assert rollout.closed_loop_start_us == 150_000
    assert rollout.force_gt_period == range(150_000, 150_001)
    assert rollout.physics_update_mode == PhysicsUpdateMode.EGO_ONLY


def test_create_keeps_synthetic_first_exposure_inside_rollout_window(
    tmp_path,
) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(
            assert_zero_decision_delay=True,
            cameras=[
                RuntimeCameraConfig(
                    logical_id="camera_front",
                    shutter_duration_us=100_000,
                ),
                RuntimeCameraConfig(
                    logical_id="camera_left",
                    shutter_duration_us=100_000,
                ),
            ],
        ),
        scene_id="scene",
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.first_camera_frame_ranges_us["camera_front"] == range(
        50_000, 150_000
    )
    assert rollout.first_camera_frame_ranges_us["camera_left"] == range(50_000, 150_000)
    assert rollout.egomotion_context_start_us == 0
    assert rollout.traffic_objs["actor"].trajectory.time_range_us.start == 0
