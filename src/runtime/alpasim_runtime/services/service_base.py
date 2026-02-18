# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Base classes for service architecture.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import Empty, VersionId
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.config import ScenarioConfig

import grpc

logger = logging.getLogger(__name__)

StubType = TypeVar("StubType")

WILDCARD_SCENE_ID = "*"


@dataclass
class SessionInfo:
    """Common session information shared by all services."""

    uuid: str
    broadcaster: MessageBroadcaster
    additional_args: Dict[str, Any]


class ServiceBase(ABC, Generic[StubType]):
    """
    Base class for all services. All services are treated as having sessions.
    For services that don't need session state, the session methods are no-ops.
    """

    def __init__(
        self,
        address: str,
        skip: bool = False,
        connection_timeout_s: int = 30,
        id: int = 0,
    ):
        self.address = address
        self.skip = skip
        self.id = id
        self.connection_timeout_s = connection_timeout_s
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[StubType] = None
        self._available_scenes: Optional[List[str]] = None
        self.session_info: Optional[SessionInfo] = None

    @property
    @abstractmethod
    def stub_class(self) -> Type[StubType]:
        """Return the gRPC stub class for this service."""
        pass

    @property
    def name(self) -> str:
        """Return a human-readable name for this service instance."""
        return f"{self.__class__.__name__}(address={self.address}, id={self.id})"

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

    def session(
        self, uuid: str, broadcaster: MessageBroadcaster, **kwargs: Any
    ) -> "ServiceBase[StubType]":
        """Configure this service for session mode.

        This class can act as a context manager in two ways:
          * If used as context manager before `session()` is called, it will
            act as a context manager for the connection to the service.
          * If used as context manager after `session()` is called, it will
            additionally also call `_initialize_session()` and
            `_cleanup_session()` when entering and exiting the context manager.
            This can be used to initialize the session in the service.

        Child classes should, if needed, override:
          * [Optional] The `session()` method to add typed parameters instead of
            using `**kwargs`.
          * The `_initialize_session()` method to initialize the session in the service.
          * The `_cleanup_session()` method to cleanup the session in the service.
        """
        # Set up session state
        assert self.session_info is None, "Session already set up"
        self.session_info = SessionInfo(
            uuid=uuid, broadcaster=broadcaster, additional_args=kwargs
        )
        return self

    async def _initialize_session(
        self, session_info: SessionInfo, **kwargs: Any
    ) -> None:
        """Override in services that need to initialize session in service."""
        pass

    async def _cleanup_session(self, session_info: SessionInfo, **kwargs: Any) -> None:
        """Override in services that need to cleanup session in service."""
        pass

    async def __aenter__(self) -> "ServiceBase[StubType]":
        """Enter async context manager for connection and optional session management."""
        await self._open_connection()

        # If this is a session context, initialize the session
        if self.session_info:
            await self._initialize_session(session_info=self.session_info)

        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit async context manager, handling cleanup for both modes."""
        # If this was a session context, clean up session first
        if self.session_info:
            await self._cleanup_session(session_info=self.session_info)
            self.session_info = None

        # Always close connection
        await self._close_connection()

    async def get_version(self) -> VersionId:
        """Get version information from the service."""
        if self.skip:
            return VersionId(
                version_id="<skip>",
                git_hash="<skip>",
                grpc_api_version=API_VERSION_MESSAGE,
            )

        async with self:
            return await self.stub.get_version(
                Empty(), wait_for_ready=True, timeout=self.connection_timeout_s
            )

    async def get_available_scenes(self) -> List[str]:
        """Get list of available scenes from the sensorsim service.

        Can be overridden in services that have special requirements (or none).
        """
        if self.skip:
            return [WILDCARD_SCENE_ID]

        if self._available_scenes is None:
            async with self:
                response = await self.stub.get_available_scenes(Empty())
                self._available_scenes = list(response.scene_ids)

        return self._available_scenes

    async def find_scenario_incompatibilities(
        self, scenario: ScenarioConfig
    ) -> List[str]:
        """Check if this service can handle the given scenario."""
        incompatibilities = []

        available_scenes = await self.get_available_scenes()

        # Allow wildcard scene
        if (
            scenario.scene_id not in available_scenes
            and WILDCARD_SCENE_ID not in available_scenes
        ):
            incompatibilities.append(
                f"Scene {scenario.scene_id} not available in {self.__class__.__name__} service. "
                f"Available scenes: {sorted(available_scenes)}"
            )

        return incompatibilities
