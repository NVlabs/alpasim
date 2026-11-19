# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from alpasim_grpc.v0.egodriver_pb2 import VectorizedObservation
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.service_base import SessionInfo
from alpasim_runtime.services.session_configs import DriverSessionConfig
from alpasim_utils.geometry import Pose, pose_to_grpc
from alpasim_utils.scenario import AABB

import grpc
import grpc.aio


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.entries = []

    async def broadcast(self, entry) -> None:
        self.entries.append(entry)


@pytest.fixture
def vehicle_geometry() -> tuple[AABB, Pose]:
    return AABB(5.4, 2.1, 1.5), Pose(
        np.array([1.4, 0.0, 0.75], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_driver_session_request_uses_configured_seed(vehicle_geometry) -> None:
    broadcaster = RecordingBroadcaster()
    service = DriverService(address="localhost:0", skip=True)

    await service._initialize_session(
        SessionInfo(
            uuid="session-seeded",
            broadcaster=broadcaster,
            session_config=DriverSessionConfig(
                sensorsim_cameras=[],
                ego_aabb=vehicle_geometry[0],
                rig_to_ego_aabb=vehicle_geometry[1],
                scene_key="clipgt-01d503d4-449b-46fc-8d78-9085e70d3554@t0=AUTO",
                random_seed=777,
            ),
        )
    )

    assert len(broadcaster.entries) == 1
    request = broadcaster.entries[0].driver_session_request
    assert request.random_seed == 777
    assert request.rollout_spec.vehicle.bounding_box == vehicle_geometry[0].to_grpc()
    assert request.rollout_spec.vehicle.rig_to_bounding_box == pose_to_grpc(
        vehicle_geometry[1]
    )


@pytest.mark.asyncio
async def test_driver_session_request_randomizes_seed_when_unset(
    vehicle_geometry,
) -> None:
    broadcaster = RecordingBroadcaster()
    service = DriverService(address="localhost:0", skip=True)

    await service._initialize_session(
        SessionInfo(
            uuid="session-random",
            broadcaster=broadcaster,
            session_config=DriverSessionConfig(
                sensorsim_cameras=[],
                ego_aabb=vehicle_geometry[0],
                rig_to_ego_aabb=vehicle_geometry[1],
                scene_key="clipgt-01d503d4-449b-46fc-8d78-9085e70d3554@t0=AUTO",
            ),
        )
    )

    assert len(broadcaster.entries) == 1
    request = broadcaster.entries[0].driver_session_request
    assert 0 <= request.random_seed < 2**32


def _make_aio_rpc_error(
    code: grpc.StatusCode, details: str = ""
) -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(
        code=code,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details=details,
    )


@pytest.mark.asyncio
async def test_vectorized_observation_rpc_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcaster = RecordingBroadcaster()
    service = DriverService(address="driver:6000", skip=False)
    service.session_info = SessionInfo(uuid="session", broadcaster=broadcaster)
    service.stub = SimpleNamespace(submit_vectorized_observation=object())
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def fail_unimplemented(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))
        raise _make_aio_rpc_error(
            grpc.StatusCode.UNIMPLEMENTED,
            "Method not found!",
        )

    monkeypatch.setattr(
        "alpasim_runtime.services.driver_service.profiled_rpc_call",
        fail_unimplemented,
    )

    with pytest.raises(grpc.aio.AioRpcError):
        await service.submit_vectorized_observation(
            VectorizedObservation(timestamp_us=1)
        )

    assert len(calls) == 1
    assert len(broadcaster.entries) == 1
    entry = broadcaster.entries[0].vectorized_observation
    assert entry.session_uuid == "session"
    assert entry.timestamp_us == 1


@pytest.mark.asyncio
async def test_navigation_instruction_is_logged_with_its_session() -> None:
    from alpasim_grpc.v0.egodriver_pb2 import NavigationInstruction
    from alpasim_runtime.nav_route import NavInstruction, NavStatus

    broadcaster = RecordingBroadcaster()
    service = DriverService(address="localhost:0", skip=True)
    service.session_info = SessionInfo(uuid="session-nav", broadcaster=broadcaster)

    await service.submit_navigation(
        1_500_000,
        NavInstruction(
            maneuver="TURN_RIGHT",
            instruction_text="Turn right onto Oak Ave in 42m",
            remaining_distance_m=42.0,
            status=NavStatus.UPCOMING,
            route_offset_m=0.5,
        ),
    )

    assert len(broadcaster.entries) == 1
    instruction = broadcaster.entries[0].navigation_instruction
    assert instruction.session_uuid == "session-nav"
    assert instruction.timestamp_us == 1_500_000
    assert instruction.instruction_text == "Turn right onto Oak Ave in 42m"
    # Diagnostics ride under debug_info; the driver-facing payload is the text alone.
    assert instruction.debug_info.status == NavigationInstruction.DebugInfo.UPCOMING
    assert instruction.debug_info.remaining_distance_m == pytest.approx(42.0)
    assert instruction.debug_info.route_offset_m == pytest.approx(0.5)
