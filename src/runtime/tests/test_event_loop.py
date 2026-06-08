# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from alpasim_runtime.event_loop import EventBasedRollout, create_event_rollout
from alpasim_runtime.events.base import Event, SimulationEndEvent
from alpasim_runtime.events.policy import PolicyEvent
from alpasim_utils.geometry import DynamicTrajectory, Pose, Trajectory
from alpasim_utils.scenario import TrafficObjects


def test_initial_event_schedule_uses_policy_and_end_timestamps() -> None:
    rollout = cast(Any, object.__new__(EventBasedRollout))
    rollout.unbound = SimpleNamespace(
        egomotion_context_start_us=100,
        render_start_timestamp_us=150,
        first_policy_timestamp_us=150,
        closed_loop_start_us=200,
        end_timestamp_us=300,
        control_timestep_us=100,
        send_recording_ground_truth=False,
    )
    rollout.runtime_cameras = []
    rollout.driver = MagicMock()
    rollout.controller = MagicMock()
    rollout.physics = MagicMock()
    rollout.trafficsim = MagicMock()
    rollout.broadcaster = MagicMock()
    rollout.planner_delay_buffer = MagicMock()
    rollout.route_generator = None
    rollout.renderer_service = MagicMock()

    class _FakeRenderEvent(Event):
        priority = 10

        async def handle(self, *_args, **_kwargs) -> None:  # pragma: no cover
            return

    def _fake_render_event(**kwargs) -> Event:
        return _FakeRenderEvent(timestamp_us=kwargs["scene_start_us"])

    rollout.renderer_service.make_initial_render_event.side_effect = _fake_render_event

    queue = rollout._create_initial_events()
    events = list(queue.queue)

    policy = next(e for e in events if isinstance(e, PolicyEvent))
    end = next(e for e in events if isinstance(e, SimulationEndEvent))
    render = next(e for e in events if isinstance(e, _FakeRenderEvent))

    # ``scene_start_us`` is anchored to the rollout start so the renderer's
    # scheduling/prefetch covers any prerun before the first GT camera frame.
    assert render.timestamp_us == 100
    assert policy.timestamp_us == 150
    assert end.timestamp_us == 300


def test_create_event_rollout_uses_renderer_service() -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_event_based_rollout(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    from unittest.mock import patch

    renderer_service = MagicMock()
    with patch(
        "alpasim_runtime.event_loop.EventBasedRollout", fake_event_based_rollout
    ):
        rollout = create_event_rollout(
            unbound=MagicMock(),
            data_source=MagicMock(),
            driver=MagicMock(),
            renderer_service=renderer_service,
            physics=MagicMock(),
            trafficsim=MagicMock(),
            controller=MagicMock(),
            camera_catalog=MagicMock(),
            eval_config=MagicMock(),
            eval_executor=MagicMock(),
        )

    assert captured_kwargs["renderer_service"] is renderer_service
    assert rollout.renderer_service is renderer_service


def test_initial_ego_context_uses_all_gt_samples_through_first_policy(
    simple_trajectory: Trajectory,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("alpasim_runtime.event_loop.RuntimeEvaluator", MagicMock())
    monkeypatch.setattr(
        "alpasim_runtime.event_loop.RouteGenerator.create",
        MagicMock(return_value=None),
    )

    unbound = SimpleNamespace(
        egomotion_context_start_us=0,
        first_policy_timestamp_us=100_000,
        closed_loop_start_us=300_000,
        gt_ego_trajectory=simple_trajectory,
        traffic_objs=TrafficObjects(),
        planner_delay_us=0,
        vector_map=None,
        route_generator_type="RECORDED",
        route_start_offset_m=0.0,
        rollout_uuid="rollout",
        scene_id="scene",
        save_path_root=str(tmp_path),
    )

    rollout = EventBasedRollout(
        unbound=unbound,
        data_source=MagicMock(),
        driver=MagicMock(),
        renderer_service=MagicMock(),
        physics=MagicMock(),
        trafficsim=MagicMock(),
        controller=MagicMock(),
        camera_catalog=MagicMock(),
        eval_config=MagicMock(),
        eval_executor=MagicMock(),
    )

    assert rollout.ego_trajectory_estimate.timestamps_us.tolist() == [0, 100_000]
    state = rollout._create_rollout_state()
    assert state.last_egopose_update_us is None


@pytest.mark.asyncio
async def test_force_gt_physics_blend_holds_first_frame_then_blends(
    simple_trajectory: Trajectory,
) -> None:
    rollout = cast(Any, object.__new__(EventBasedRollout))
    rollout.unbound = SimpleNamespace(
        gt_ego_trajectory=simple_trajectory,
        egomotion_context_start_us=0,
        render_start_timestamp_us=100_000,
        first_policy_timestamp_us=100_000,
        closed_loop_start_us=300_000,
        end_timestamp_us=300_000,
        force_gt_duration_us=200_000,
        control_timestep_us=100_000,
        first_camera_frame_ranges_us={
            "camera_front": range(70_000, 90_000),
            "camera_cross_left": range(75_000, 100_000),
        },
    )

    async def fake_apply_physics(trajectory: Trajectory) -> Trajectory:
        poses = [
            Pose(
                pose.vec3 + np.array([0.0, 0.0, 10.0], dtype=np.float32),
                pose.quat,
            )
            for pose in (trajectory.get_pose(i) for i in range(len(trajectory)))
        ]
        return Trajectory.from_poses(trajectory.timestamps_us, poses)

    rollout._apply_physics_to_trajectory = AsyncMock(side_effect=fake_apply_physics)

    blended = await rollout._build_force_gt_physics_blend_trajectory()

    assert blended.timestamps_us.tolist() == [0, 100_000, 200_000, 300_000]
    assert blended.positions[:, 2].tolist() == [0.0, 0.0, 5.0, 10.0]


@pytest.mark.asyncio
async def test_force_gt_physics_blend_caps_huge_duration_to_rollout_end(
    simple_trajectory: Trajectory,
) -> None:
    rollout = cast(Any, object.__new__(EventBasedRollout))
    rollout.unbound = SimpleNamespace(
        gt_ego_trajectory=simple_trajectory,
        egomotion_context_start_us=0,
        render_start_timestamp_us=100_000,
        first_policy_timestamp_us=100_000,
        closed_loop_start_us=100_000_000_100_000,
        end_timestamp_us=300_000,
        force_gt_duration_us=100_000_000_000_000,
        control_timestep_us=100_000,
    )

    async def fake_apply_physics(trajectory: Trajectory) -> Trajectory:
        poses = [
            Pose(
                pose.vec3 + np.array([0.0, 0.0, 10.0], dtype=np.float32),
                pose.quat,
            )
            for pose in (trajectory.get_pose(i) for i in range(len(trajectory)))
        ]
        return Trajectory.from_poses(trajectory.timestamps_us, poses)

    rollout._apply_physics_to_trajectory = AsyncMock(side_effect=fake_apply_physics)

    blended = await rollout._build_force_gt_physics_blend_trajectory()

    assert blended.timestamps_us.tolist() == [0, 100_000, 200_000, 300_000]
    assert float(blended.positions[-1, 2]) < 0.001


@pytest.mark.asyncio
async def test_force_gt_physics_blend_zero_duration_is_noop() -> None:
    gt_trajectory = Trajectory.from_poses(
        np.array([0, 50_000, 100_000], dtype=np.uint64),
        [
            Pose(
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            Pose(
                np.array([0.0, 10.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            Pose(
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
        ],
    )
    rollout = cast(Any, object.__new__(EventBasedRollout))
    rollout.unbound = SimpleNamespace(
        gt_ego_trajectory=gt_trajectory,
        egomotion_context_start_us=0,
        render_start_timestamp_us=100_000,
        first_policy_timestamp_us=100_000,
        closed_loop_start_us=100_000,
        end_timestamp_us=200_000,
        force_gt_duration_us=0,
        control_timestep_us=100_000,
    )
    rollout._apply_physics_to_trajectory = AsyncMock()

    blended = await rollout._build_force_gt_physics_blend_trajectory()

    rollout._apply_physics_to_trajectory.assert_not_awaited()
    assert blended.timestamps_us.tolist() == [0, 50_000, 100_000]
    np.testing.assert_array_equal(blended.positions, gt_trajectory.positions)


def test_force_gt_blend_replaces_seeded_ego_poses() -> None:
    rollout = cast(Any, object.__new__(EventBasedRollout))
    seeded = Trajectory.from_poses(
        np.array([0, 100_000, 166_665], dtype=np.uint64),
        [
            Pose(
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            Pose(
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            Pose(
                np.array([2.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
        ],
    )
    blended = Trajectory.from_poses(
        np.array([0, 100_000, 166_665], dtype=np.uint64),
        [
            Pose(
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            Pose(
                np.array([1.0, 0.0, 2.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            Pose(
                np.array([2.0, 0.0, 4.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
        ],
    )
    dynamics = np.zeros((3, 12), dtype=np.float64)
    rollout.ego_trajectory = DynamicTrajectory.from_trajectory_and_dynamics(
        seeded, dynamics
    )
    rollout.ego_trajectory_estimate = rollout.ego_trajectory.clone()
    rollout.force_gt_ego_trajectory = blended

    rollout._apply_force_gt_blend_to_seeded_ego_trajectory()

    np.testing.assert_allclose(rollout.ego_trajectory.positions, blended.positions)
    np.testing.assert_allclose(
        rollout.ego_trajectory_estimate.positions, blended.positions
    )
