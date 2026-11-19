# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""The driver keeps the latest navigation instruction per session, and drops it on an empty one."""

from types import SimpleNamespace

from alpasim_driver.main import EgoDriverService
from alpasim_grpc.v0.egodriver_pb2 import NavigationInstruction


def _service_with_session(
    session_uuid: str,
) -> tuple[EgoDriverService, SimpleNamespace]:
    service = EgoDriverService.__new__(EgoDriverService)
    session = SimpleNamespace(navigation=None)
    service._sessions = {session_uuid: session}
    return service, session


def test_navigation_instruction_is_kept_until_replaced() -> None:
    service, session = _service_with_session("s")
    turn = NavigationInstruction(
        session_uuid="s",
        timestamp_us=1,
        instruction_text="Turn right onto Oak Ave in 42m",
    )

    service.submit_navigation(turn, None)

    assert session.navigation is turn


def test_empty_navigation_instruction_clears_the_session() -> None:
    """A step without an instruction must not leave the previous one conditioning the model."""
    service, session = _service_with_session("s")
    service.submit_navigation(
        NavigationInstruction(
            session_uuid="s", timestamp_us=1, instruction_text="Turn left"
        ),
        None,
    )

    service.submit_navigation(
        NavigationInstruction(session_uuid="s", timestamp_us=2), None
    )

    assert session.navigation is None
