# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from alpasim_runtime.config import DriverBackendConfig
from alpasim_runtime.decision import MultiBackendDriverOrchestrator
from alpasim_runtime.event_loop import EventBasedRollout
from alpasim_runtime.events.base import Event, EventQueue, SimulationEndEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle
from alpasim_runtime.events.step import StepEvent
from alpasim_runtime.events.policy import PolicyEvent


def test_initial_event_schedule_matches_control_timestamps() -> None:
    rollout = cast(Any, object.__new__(EventBasedRollout))
    rollout.unbound = SimpleNamespace(
        control_timestamps_us=[100, 200, 300],
        control_timestep_us=100,
        group_render_requests=False,
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
    rollout.sensorsim = MagicMock()

    queue = rollout._create_initial_events()
    events = list(queue.queue)

    policy = next(e for e in events if isinstance(e, PolicyEvent))
    end = next(e for e in events if isinstance(e, SimulationEndEvent))

    assert policy.timestamp_us == 200
    assert end.timestamp_us == 300


class _MarkerEvent(Event):
    """Simple event used to assert loop ordering in tests."""

    priority = 5

    def __init__(self, timestamp_us: int, marker: list[int]):
        super().__init__(timestamp_us)
        self._marker = marker

    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        del rollout_state, queue
        self._marker.append(self.timestamp_us)


@pytest.mark.asyncio
async def test_run_until_step_commit_stops_after_first_step_event(
    rollout_state: RolloutState,
    service_bundle: ServiceBundle,
) -> None:
    rollout = cast(Any, object.__new__(EventBasedRollout))
    marker: list[int] = []

    rollout._initialized = True
    rollout._closed = False
    rollout._state = rollout_state
    rollout._async_stack = None
    rollout._loop_start_time = 0.0
    rollout._rollout_start_time = 0.0
    rollout._event_queue = EventQueue.init_from_sequence(
        [
            _MarkerEvent(timestamp_us=50, marker=marker),
            StepEvent(
                timestamp_us=100,
                control_timestep_us=100,
                services=service_bundle,
            ),
            _MarkerEvent(timestamp_us=150, marker=marker),
        ]
    )
    rollout_state.step_context = None

    result = await rollout.run_until_step_commit()

    assert marker == [50]
    assert result.step_committed is True
    assert result.simulation_finished is False
    assert result.committed_step_timestamp_us == 100

    pending_timestamps = sorted(event.timestamp_us for event in rollout._event_queue.queue)
    assert pending_timestamps == [150, 200]


def test_create_service_bundle_builds_multi_backend_orchestrator_from_config() -> None:
    rollout = cast(Any, object.__new__(EventBasedRollout))
    rollout.unbound = SimpleNamespace(
        driver_backends=[
            DriverBackendConfig(backend_id="ar1_backend", model_type="ar1", priority=5),
            DriverBackendConfig(backend_id="pdm_backend", model_type="pdm", priority=1),
        ]
    )
    rollout.driver = MagicMock()
    rollout.controller = MagicMock()
    rollout.physics = MagicMock()
    rollout.trafficsim = MagicMock()
    rollout.broadcaster = MagicMock()
    rollout.planner_delay_buffer = MagicMock()
    rollout._default_driver_orchestrator = None

    bundle = rollout._create_service_bundle()

    assert isinstance(bundle.driver_orchestrator, MultiBackendDriverOrchestrator)


@pytest.mark.asyncio
async def test_capture_and_restore_runtime_checkpoint_preserves_state_and_queue_async(
    rollout_state: RolloutState,
) -> None:
    rollout = cast(Any, object.__new__(EventBasedRollout))
    handler = MagicMock()
    rollout._initialized = True
    rollout._closed = False
    rollout._default_driver_orchestrator = None
    rollout._build_default_driver_orchestrator = MagicMock(return_value=None)
    rollout._state = rollout_state
    rollout._event_queue = EventQueue.init_from_sequence(
        [
            _MarkerEvent(timestamp_us=50, marker=[]),
            SimulationEndEvent(timestamp_us=300),
        ]
    )
    rollout_state.rendered_images_handler = handler
    rollout_state.last_decision_step_id = 7
    checkpoint = rollout.capture_runtime_checkpoint()

    rollout_state.last_decision_step_id = 99
    rollout._event_queue.pop()

    await rollout.restore_runtime_checkpoint(checkpoint)

    assert rollout.current_state.last_decision_step_id == 7
    assert rollout.current_state.rendered_images_handler is handler
    assert sorted(event.timestamp_us for event in rollout._event_queue.queue) == [50, 300]


@pytest.mark.asyncio
async def test_restore_runtime_checkpoint_restores_backend_state(
    rollout_state: RolloutState,
) -> None:
    rollout = cast(Any, object.__new__(EventBasedRollout))
    orchestrator = SimpleNamespace(
        capture_backend_checkpoint=MagicMock(return_value=({"safe": {"cp": 1}}, [])),
        restore_backend_checkpoint=AsyncMock(),
    )
    rollout._initialized = True
    rollout._closed = False
    rollout._default_driver_orchestrator = orchestrator
    rollout._build_default_driver_orchestrator = MagicMock(return_value=orchestrator)
    rollout._state = rollout_state
    rollout._event_queue = EventQueue.init_from_sequence([SimulationEndEvent(timestamp_us=300)])
    rollout_state.rendered_images_handler = MagicMock()

    checkpoint = rollout.capture_runtime_checkpoint()

    await rollout.restore_runtime_checkpoint(checkpoint)

    orchestrator.restore_backend_checkpoint.assert_awaited_once_with({"safe": {"cp": 1}})
