# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from alpasim_runtime.event_loop import EventBasedRollout
from alpasim_runtime.events.base import SimulationEndEvent
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
