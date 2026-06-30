# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for NRE introspection helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from alpasim_runtime.nre_introspection import (
    IntrospectionNotSupportedError,
    get_loaded_scenes,
)

import grpc
import grpc.aio


def _make_aio_rpc_error(
    code: grpc.StatusCode, details: str = ""
) -> grpc.aio.AioRpcError:
    """Construct an AioRpcError for testing."""
    return grpc.aio.AioRpcError(
        code=code,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details=details,
    )


class TestGetLoadedScenes:
    """Tests for get_loaded_scenes."""

    @pytest.mark.asyncio
    async def test_parses_response_correctly(self):
        """Successful response should be parsed into {scene_id: count} dict."""
        mock_channel = AsyncMock()

        entry_a = MagicMock()
        entry_a.scene_id = "scene-A"
        entry_a.loaded_instance_count = 3
        entry_b = MagicMock()
        entry_b.scene_id = "scene-B"
        entry_b.loaded_instance_count = 1

        mock_response = MagicMock()
        mock_response.scenes = [entry_a, entry_b]

        mock_stub = MagicMock()
        mock_stub.get_loaded_scenes = AsyncMock(return_value=mock_response)

        with (
            patch(
                "alpasim_runtime.nre_introspection.grpc.aio.insecure_channel",
                return_value=mock_channel,
            ),
            patch(
                "alpasim_runtime.nre_introspection.SensorsimServiceStub",
                return_value=mock_stub,
            ),
        ):
            result = await get_loaded_scenes("gpu-0:50052")

        assert result == {"scene-A": 3, "scene-B": 1}
        mock_channel.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_grpc_error(self):
        """Any gRPC error should return None (logged as warning)."""
        mock_channel = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.get_loaded_scenes = AsyncMock(
            side_effect=_make_aio_rpc_error(
                grpc.StatusCode.UNAVAILABLE, "connection refused"
            )
        )

        with (
            patch(
                "alpasim_runtime.nre_introspection.grpc.aio.insecure_channel",
                return_value=mock_channel,
            ),
            patch(
                "alpasim_runtime.nre_introspection.SensorsimServiceStub",
                return_value=mock_stub,
            ),
        ):
            result = await get_loaded_scenes("gpu-0:50052")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_unimplemented_by_default(self):
        """UNIMPLEMENTED returns None when raise_on_unimplemented is False."""
        mock_channel = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.get_loaded_scenes = AsyncMock(
            side_effect=_make_aio_rpc_error(grpc.StatusCode.UNIMPLEMENTED)
        )

        with (
            patch(
                "alpasim_runtime.nre_introspection.grpc.aio.insecure_channel",
                return_value=mock_channel,
            ),
            patch(
                "alpasim_runtime.nre_introspection.SensorsimServiceStub",
                return_value=mock_stub,
            ),
        ):
            result = await get_loaded_scenes("gpu-0:50052")
            assert result is None

    @pytest.mark.asyncio
    async def test_raises_on_unimplemented_when_requested(self):
        """UNIMPLEMENTED raises IntrospectionNotSupportedError at startup."""
        mock_channel = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.get_loaded_scenes = AsyncMock(
            side_effect=_make_aio_rpc_error(grpc.StatusCode.UNIMPLEMENTED)
        )

        with (
            patch(
                "alpasim_runtime.nre_introspection.grpc.aio.insecure_channel",
                return_value=mock_channel,
            ),
            patch(
                "alpasim_runtime.nre_introspection.SensorsimServiceStub",
                return_value=mock_stub,
            ),
        ):
            with pytest.raises(IntrospectionNotSupportedError):
                await get_loaded_scenes("gpu-0:50052", raise_on_unimplemented=True)

    @pytest.mark.asyncio
    async def test_transient_error_returns_none_even_with_raise_flag(self):
        """Transient errors (UNAVAILABLE) still return None even with the flag."""
        mock_channel = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.get_loaded_scenes = AsyncMock(
            side_effect=_make_aio_rpc_error(
                grpc.StatusCode.UNAVAILABLE, "connection refused"
            )
        )

        with (
            patch(
                "alpasim_runtime.nre_introspection.grpc.aio.insecure_channel",
                return_value=mock_channel,
            ),
            patch(
                "alpasim_runtime.nre_introspection.SensorsimServiceStub",
                return_value=mock_stub,
            ),
        ):
            result = await get_loaded_scenes("gpu-0:50052", raise_on_unimplemented=True)
            assert result is None
