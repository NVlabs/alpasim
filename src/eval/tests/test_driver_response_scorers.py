# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for metrics that consume driver responses."""

import numpy as np
import pytest
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_utils.geometry import Pose, Trajectory
from conftest import create_test_eval_config

from eval.data import (
    RAABB,
    ActorPolygons,
    Cameras,
    DriverResponseAtTime,
    DriverResponses,
    RenderableTrajectory,
    Routes,
    SimulationResult,
)
from eval.scorers.minADE import MinADEScorer
from eval.scorers.plan_deviation import PlanDeviationScorer
from eval.scorers.safety import SafetyScorer


def _trajectory(timestamps_us: np.ndarray) -> Trajectory:
    positions = np.stack(
        [
            timestamps_us.astype(np.float32) / 100_000.0,
            np.zeros(len(timestamps_us), dtype=np.float32),
            np.zeros(len(timestamps_us), dtype=np.float32),
        ],
        axis=1,
    )
    quaternions = np.tile(
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        (len(timestamps_us), 1),
    )
    return Trajectory(timestamps_us, positions, quaternions)


def _renderable(timestamps_us: np.ndarray, raabb: RAABB) -> RenderableTrajectory:
    return RenderableTrajectory.from_trajectory(
        _trajectory(timestamps_us),
        raabb,
    )


@pytest.fixture
def off_cadence_simulation_result() -> SimulationResult:
    """Pose timestamps are 10Hz; driver responses are 5Hz policy decisions."""
    raabb = RAABB(size_x=4.5, size_y=2.0, size_z=1.5, corner_radius_m=0.0)
    pose_timestamps = np.arange(0, 900_000, 100_000, dtype=np.uint64)
    response_timestamps = [0, 200_000, 400_000]
    ego_traj = _renderable(pose_timestamps, raabb)

    responses = []
    for idx, ts in enumerate(response_timestamps):
        plan_timestamps = pose_timestamps[pose_timestamps >= ts]
        responses.append(
            DriverResponseAtTime(
                now_time_us=ts,
                time_query_us=ts,
                selected_trajectory=_renderable(plan_timestamps, raabb),
                sampled_trajectories=[_renderable(plan_timestamps, raabb)],
                safety_monitor_safe=(idx != 1),
            )
        )

    driver_responses = DriverResponses(
        ego_coords_rig_to_aabb_center=Pose.identity(),
        ego_trajectory_local=ego_traj,
        timestamps_us=response_timestamps,
        query_times_us=response_timestamps,
        per_timestep_driver_responses=responses,
    )

    return SimulationResult(
        session_metadata=RolloutMetadata.SessionMetadata(
            session_uuid="test-session",
            scene_id="test-scene",
            batch_size=1,
            n_sim_steps=len(pose_timestamps),
            start_timestamp_us=0,
            control_timestep_us=200_000,
        ),
        ego_coords_rig_to_aabb_center=Pose.identity(),
        actor_trajectories={"EGO": ego_traj},
        driver_estimated_trajectory=ego_traj,
        driver_responses=driver_responses,
        ego_recorded_ground_truth_trajectory=ego_traj,
        vec_map=None,
        actor_polygons=ActorPolygons.from_actor_trajectories({"EGO": ego_traj}),
        cameras=Cameras(),
        routes=Routes(),
    )


def test_driver_response_scorers_use_policy_timestamps(
    off_cadence_simulation_result: SimulationResult,
) -> None:
    cfg = create_test_eval_config()
    cfg.scorers.min_ade.time_deltas = [0.1]
    min_ade = MinADEScorer(cfg).calculate(off_cadence_simulation_result)[0]
    plan_deviation = PlanDeviationScorer(cfg).calculate(off_cadence_simulation_result)[
        0
    ]
    safety = SafetyScorer(cfg).calculate(off_cadence_simulation_result)[0]

    assert min_ade.timestamps_us == [0, 200_000, 400_000]
    assert min_ade.values == pytest.approx([0.0, 0.0, 0.0])
    assert plan_deviation.timestamps_us == [200_000, 400_000]
    assert plan_deviation.values == pytest.approx([0.0, 0.0])
    assert safety.timestamps_us == [0, 200_000, 400_000]
    assert safety.values == [False, True, False]
