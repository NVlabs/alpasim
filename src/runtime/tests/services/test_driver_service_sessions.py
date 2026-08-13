# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import pytest
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.service_base import SessionInfo
from alpasim_runtime.services.session_configs import DriverSessionConfig


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.entries = []

    async def broadcast(self, entry) -> None:
        self.entries.append(entry)


@pytest.mark.asyncio
async def test_driver_session_request_uses_configured_seed() -> None:
    broadcaster = RecordingBroadcaster()
    service = DriverService(address="localhost:0", skip=True)

    await service._initialize_session(
        SessionInfo(
            uuid="session-seeded",
            broadcaster=broadcaster,
            session_config=DriverSessionConfig(
                sensorsim_cameras=[],
                scene_id="clipgt-01d503d4-449b-46fc-8d78-9085e70d3554",
                random_seed=777,
            ),
        )
    )

    assert len(broadcaster.entries) == 1
    request = broadcaster.entries[0].driver_session_request
    assert request.random_seed == 777


@pytest.mark.asyncio
async def test_driver_session_request_randomizes_seed_when_unset() -> None:
    broadcaster = RecordingBroadcaster()
    service = DriverService(address="localhost:0", skip=True)

    await service._initialize_session(
        SessionInfo(
            uuid="session-random",
            broadcaster=broadcaster,
            session_config=DriverSessionConfig(sensorsim_cameras=[]),
        )
    )

    assert len(broadcaster.entries) == 1
    request = broadcaster.entries[0].driver_session_request
    assert 0 <= request.random_seed < 2**32
