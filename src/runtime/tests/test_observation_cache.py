# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import numpy as np
from alpasim_runtime.observation_cache import (
    ObservationCache,
    ObservationFrame,
)
from alpasim_utils.geometry import DynamicTrajectory, Pose, Trajectory
from alpasim_utils.scenario import TrafficObjects
from alpasim_utils.types import ImageWithMetadata


def _make_dynamic_trajectory(timestamp_us: int) -> DynamicTrajectory:
    trajectory = Trajectory.from_poses(
        timestamps=np.array([timestamp_us], dtype=np.uint64),
        poses=[
            Pose(
                np.array([float(timestamp_us) / 1e6, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
        ],
    )
    return DynamicTrajectory.from_trajectory_and_dynamics(
        trajectory,
        np.zeros((1, 12), dtype=np.float64),
    )


def _make_frame(step_id: int) -> ObservationFrame:
    timestamp_us = step_id * 100_000
    return ObservationFrame(
        step_id=step_id,
        input_snapshot_id=f"snapshot-{step_id}",
        time_now_us=timestamp_us,
        time_query_us=timestamp_us + 100_000,
        camera_frame_timestamps_us={"cam_front": timestamp_us},
        rendered_images={
            "cam_front": ImageWithMetadata(
                start_timestamp_us=timestamp_us - 33_000,
                end_timestamp_us=timestamp_us,
                image_bytes=f"img-{step_id}".encode(),
                camera_logical_id="cam_front",
            )
        },
        renderer_data=f"renderer-{step_id}".encode(),
        ego_trajectory=_make_dynamic_trajectory(timestamp_us),
        ego_trajectory_estimate=_make_dynamic_trajectory(timestamp_us),
        traffic_objs=TrafficObjects(),
        ego_pose_history_timestamps_us=[timestamp_us - 100_000, timestamp_us],
        route_waypoints_in_rig=[[0.0, 0.0, 0.0]],
        planner_context={"step_id": step_id},
        active_backend_ids=["default_driver"],
    )


def test_observation_cache_window_and_checkpoint_restore() -> None:
    cache = ObservationCache(max_frames=4)
    for step_id in range(1, 5):
        cache.append(_make_frame(step_id))

    window = cache.get_window("snapshot-4", window_size=2)
    checkpoint = cache.checkpoint()

    assert [frame.input_snapshot_id for frame in window.frames] == [
        "snapshot-3",
        "snapshot-4",
    ]
    assert window.frames[-1].rendered_images["cam_front"].image_bytes == b"img-4"

    cache.append(_make_frame(5))
    cache.restore(checkpoint)

    assert cache.latest() is not None
    assert cache.latest().input_snapshot_id == "snapshot-4"
    assert cache.get("snapshot-4").renderer_data == b"renderer-4"
