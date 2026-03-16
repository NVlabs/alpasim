# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Base classes for service architecture.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Generic, Optional, Type, TypeVar

from alpasim_runtime.broadcaster import MessageBroadcaster

import grpc

logger = logging.getLogger(__name__)

StubType = TypeVar("StubType")


@dataclass
class SessionInfo:
    """Common session information shared by all services.

    ``session_config`` carries a typed, per-service frozen dataclass (e.g.
    ``DriverSessionConfig``, ``TrafficSessionConfig``).  Services should read
    from ``session_config`` when service-specific parameters are needed.
    """

    uuid: str
    broadcaster: MessageBroadcaster
    session_config: object | None = None


class ServiceBase(ABC, Generic[StubType]):
    """
    Base class for all services. All services are treated as having sessions.
    For services that don't need session state, the session methods are no-ops.
    """

    def __init__(
        self,
        address: str,
        skip: bool = False,
    ):
        self.address = address
        self.skip = skip
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[StubType] = None
        self.session_info: Optional[SessionInfo] = None

    @property
    @abstractmethod
    def stub_class(self) -> Type[StubType]:
        """Return the gRPC stub class for this service."""
        pass

    @property
    def name(self) -> str:
        """Return a human-readable name for this service instance."""
        return f"{self.__class__.__name__}(address={self.address})"

    async def _open_connection(self) -> None:
        """Open gRPC connection."""
        if not self.skip:
            self.channel = grpc.aio.insecure_channel(self.address)
            self.stub = self.stub_class(self.channel)

    async def _close_connection(self) -> None:
        """Close gRPC connection."""
        if self.channel is not None:
            await self.channel.close()
            self.channel = None
        self.stub = None

    # -- Explicit lifecycle context manager (Phase 3) --------------------------

    @asynccontextmanager
    async def rollout_session(
        self,
        uuid: str,
        broadcaster: MessageBroadcaster,
        session_config: object | None = None,
    ) -> AsyncIterator[ServiceBase[StubType]]:
        """Full rollout lifecycle: connection + session init/cleanup.

        Opens the connection, creates :class:`SessionInfo`, calls
        ``_initialize_session``, yields, then runs ``_cleanup_session``
        and closes the connection — even when exceptions occur.
        """
        assert self.session_info is None, "Session already set up"
        active_session_info = SessionInfo(
            uuid=uuid,
            broadcaster=broadcaster,
            session_config=session_config,
        )
        self.session_info = active_session_info
        body_exception: BaseException | None = None
        await self._open_connection()
        try:
            await self._initialize_session(session_info=active_session_info)
            yield self
        except BaseException as exc:
            # Re-raise after teardown below.
            body_exception = exc
            raise
        finally:
            teardown_errors: list[tuple[str, Exception]] = []

            # Always attempt cleanup+close, even on init failure.
            try:
                await self._cleanup_session(session_info=active_session_info)
            except Exception as exc:  # noqa: BLE001
                teardown_errors.append(("cleanup_session", exc))
            self.session_info = None

            try:
                await self._close_connection()
            except Exception as exc:  # noqa: BLE001
                teardown_errors.append(("close_connection", exc))

            if teardown_errors:
                if body_exception is not None:
                    for stage, exc in teardown_errors:
                        logger.warning(
                            "Suppressed %s error in %s while handling prior exception",
                            stage,
                            self.name,
                            exc_info=exc,
                        )
                else:
                    raise teardown_errors[0][1]

    def _require_session_info(self) -> SessionInfo:
        """Return active session info or raise a clear lifecycle error."""
        if self.session_info is None:
            raise RuntimeError(f"{self.name} used outside rollout_session")
        return self.session_info

    # -- Session hooks (override in subclasses) --------------------------------

    async def _initialize_session(self, session_info: SessionInfo) -> None:
        """Override in services that need to initialize session in service."""
        pass

    async def _cleanup_session(self, session_info: SessionInfo) -> None:
        """Override in services that need to cleanup session in service."""
        pass
