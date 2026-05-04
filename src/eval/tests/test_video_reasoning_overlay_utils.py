# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for reasoning overlay video rendering helpers."""

import numpy as np
from alpasim_grpc.v0.logging_pb2 import (
    RolloutMetadata,  # type: ignore[reportMissingImports]
)
from alpasim_utils.geometry import (  # type: ignore[reportMissingImports]
    Pose,
    Trajectory,
)
from conftest import create_test_eval_config

from eval.data import (
    RAABB,
    ActorPolygons,
    Camera,
    Cameras,
    DriverResponseAtTime,
    DriverResponses,
    RenderableTrajectory,
    Routes,
    SimulationResult,
)
from eval.video_reasoning_overlay_utils import _render_single_reasoning_overlay_frame


def _make_renderable_trajectory(
    timestamps_us: np.ndarray, positions: np.ndarray, raabb: RAABB
) -> RenderableTrajectory:
    quaternions = np.tile(
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        (len(timestamps_us), 1),
    )
    return RenderableTrajectory.from_trajectory(
        Trajectory(timestamps_us, positions.astype(np.float32), quaternions),
        raabb,
    )


def test_render_single_reasoning_overlay_frame_uses_renderable_trajectory_api() -> None:
    """Regression test for rendering with RenderableTrajectory composition."""
    raabb = RAABB(size_x=4.5, size_y=2.0, size_z=1.5, corner_radius_m=0.0)
    timestamps_us = np.array([0, 100_000, 200_000], dtype=np.uint64)
    ego_traj = _make_renderable_trajectory(
        timestamps_us,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        raabb,
    )
    selected_traj = _make_renderable_trajectory(
        np.array([100_000, 200_000], dtype=np.uint64),
        np.array([[1.0, 0.0, 0.04], [2.0, 0.0, 0.04]], dtype=np.float32),
        raabb,
    )
    driver_response = DriverResponseAtTime(
        now_time_us=100_000,
        time_query_us=100_000,
        selected_trajectory=selected_traj,
        sampled_trajectories=[],
        reasoning_text="test reasoning",
    )
    driver_responses = DriverResponses(
        ego_coords_rig_to_aabb_center=Pose.identity(),
        ego_trajectory_local=ego_traj,
        timestamps_us=[100_000],
        query_times_us=[100_000],
        per_timestep_driver_responses=[driver_response],
    )
    cameras = Cameras(
        camera_by_logical_id={"test_camera": Camera.create_empty("test_camera")}
    )
    sim_result = SimulationResult(
        session_metadata=RolloutMetadata.SessionMetadata(
            session_uuid="test-session",
            scene_id="test-scene",
            batch_size=1,
            n_sim_steps=3,
            start_timestamp_us=0,
            control_timestep_us=100_000,
        ),
        ego_coords_rig_to_aabb_center=Pose.identity(),
        actor_trajectories={"EGO": ego_traj},
        driver_estimated_trajectory=ego_traj,
        driver_responses=driver_responses,
        ego_recorded_ground_truth_trajectory=ego_traj,
        vec_map=None,
        actor_polygons=ActorPolygons.from_actor_trajectories({"EGO": ego_traj}),
        cameras=cameras,
        routes=Routes(),
    )
    cfg = create_test_eval_config()
    cfg.video.camera_id_to_render = "test_camera"

    frame = _render_single_reasoning_overlay_frame(
        sim_result,
        np.uint64(100_000),
        "test reasoning",
        0.1,
        cfg,
    )

    assert frame.ndim == 3
    assert frame.shape[2] == 3
