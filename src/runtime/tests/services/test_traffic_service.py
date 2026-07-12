# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import pytest
from alpasim_runtime.services.service_base import SessionInfo
from alpasim_runtime.services.session_configs import TrafficSessionConfig
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_utils.geometry import Trajectory
from alpasim_utils.scenario import AABB


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.entries = []

    async def broadcast(self, entry) -> None:
        self.entries.append(entry)


@pytest.mark.asyncio
async def test_traffic_session_request_uses_scene_id() -> None:
    broadcaster = RecordingBroadcaster()
    service = TrafficService(address="localhost:0", skip=True)

    await service._initialize_session(
        SessionInfo(
            uuid="session-1",
            broadcaster=broadcaster,
            session_config=TrafficSessionConfig(
                traffic_objs={},
                scene_id="clipgt-01d503d4-449b-46fc-8d78-9085e70d3554",
                ego_aabb=AABB(x=4.5, y=2.0, z=1.7),
                gt_ego_aabb_trajectory=Trajectory.create_empty(),
                start_timestamp_us=0,
                force_gt_duration_us=100_000,
                control_timestep_us=100_000,
            ),
        )
    )

    assert len(broadcaster.entries) == 1
    request = broadcaster.entries[0].traffic_session_request
    assert request.scene_id == "clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"
