# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for explicit ServiceBase lifecycle split (Phase 3)."""

from __future__ import annotations

from typing import Type
from unittest.mock import AsyncMock, MagicMock

import pytest
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.services.service_base import ServiceBase, SessionInfo


class _StubService(ServiceBase[object]):
    """Minimal concrete service for testing lifecycle."""

    stub_class_value: type = object

    @property
    def stub_class(self) -> Type[object]:
        return self.stub_class_value

    def __init__(self, **kwargs):
        super().__init__(address="localhost:0", skip=True, **kwargs)
        self.init_called = False
        self.cleanup_called = False
        self.init_error: Exception | None = None
        self.cleanup_error: Exception | None = None

    async def _initialize_session(self, session_info: SessionInfo):
        self.init_called = True
        if self.init_error:
            raise self.init_error

    async def _cleanup_session(self, session_info: SessionInfo):
        self.cleanup_called = True
        if self.cleanup_error:
            raise self.cleanup_error

    def get_required_session_uuid(self) -> str:
        return self._require_session_info().uuid


@pytest.fixture
def broadcaster():
    return AsyncMock(spec=MessageBroadcaster)


class TestRolloutSession:
    """Tests for rollout_session context manager."""

    @pytest.mark.asyncio
    async def test_rollout_session_runs_full_lifecycle(self, broadcaster):
        svc = _StubService()
        async with svc.rollout_session(
            uuid="u", broadcaster=broadcaster, session_config=None
        ):
            assert svc.init_called
            assert svc.session_info is not None
        # After exit: cleanup called, session cleared, connection closed
        assert svc.cleanup_called
        assert svc.session_info is None
        assert svc.channel is None

    @pytest.mark.asyncio
    async def test_rollout_session_failure_closes_connection(self, broadcaster):
        """If _initialize_session raises, connection should still be closed."""
        svc = _StubService()
        svc.init_error = RuntimeError("init boom")
        with pytest.raises(RuntimeError, match="init boom"):
            async with svc.rollout_session(
                uuid="u", broadcaster=broadcaster, session_config=None
            ):
                pass  # pragma: no cover — should not reach here
        assert svc.channel is None
        assert svc.session_info is None

    @pytest.mark.asyncio
    async def test_rollout_session_body_error_runs_cleanup(self, broadcaster):
        """If an error occurs in the body, cleanup should still run."""
        svc = _StubService()
        with pytest.raises(ValueError, match="body error"):
            async with svc.rollout_session(
                uuid="u", broadcaster=broadcaster, session_config=None
            ):
                raise ValueError("body error")
        assert svc.cleanup_called
        assert svc.channel is None

    @pytest.mark.asyncio
    async def test_rollout_session_passes_config(self, broadcaster):
        """session_config should propagate to session_info."""
        svc = _StubService()
        config = MagicMock()
        async with svc.rollout_session(
            uuid="u", broadcaster=broadcaster, session_config=config
        ):
            assert svc.session_info is not None
            assert svc.session_info.session_config is config

    @pytest.mark.asyncio
    async def test_rollout_session_cleanup_error_raised_on_success(self, broadcaster):
        """Cleanup failures should surface when body succeeded."""
        svc = _StubService()
        svc.cleanup_error = RuntimeError("cleanup boom")

        with pytest.raises(RuntimeError, match="cleanup boom"):
            async with svc.rollout_session(
                uuid="u", broadcaster=broadcaster, session_config=None
            ):
                pass

        assert svc.cleanup_called
        assert svc.session_info is None
        assert svc.channel is None

    @pytest.mark.asyncio
    async def test_rollout_session_preserves_body_error_when_cleanup_fails(
        self, broadcaster
    ):
        """Body exception should not be masked by cleanup failure."""
        svc = _StubService()
        svc.cleanup_error = RuntimeError("cleanup boom")

        with pytest.raises(ValueError, match="body error"):
            async with svc.rollout_session(
                uuid="u", broadcaster=broadcaster, session_config=None
            ):
                raise ValueError("body error")

        assert svc.cleanup_called
        assert svc.session_info is None
        assert svc.channel is None

    def test_require_session_info_raises_outside_session(self):
        svc = _StubService()

        with pytest.raises(RuntimeError, match="used outside rollout_session"):
            svc.get_required_session_uuid()

    @pytest.mark.asyncio
    async def test_require_session_info_returns_active_session(self, broadcaster):
        svc = _StubService()

        async with svc.rollout_session(uuid="u", broadcaster=broadcaster):
            assert svc.get_required_session_uuid() == "u"
