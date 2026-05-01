# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from collections.abc import Callable

from alpasim_grpc.v0 import common_pb2, runtime_pb2, runtime_pb2_grpc
from alpasim_runtime.daemon.engine import DaemonEngine, InvalidRequestError
from alpasim_runtime.daemon.scheduler import DaemonUnavailableError

import grpc


class RuntimeDaemonServicer(runtime_pb2_grpc.RuntimeServiceServicer):
    """gRPC servicer that maps RPC methods to DaemonEngine operations.

    Maps domain exceptions to appropriate gRPC status codes:
    ``ValueError``-derived request validation errors -> ``INVALID_ARGUMENT``,
    ``DaemonUnavailableError`` -> ``UNAVAILABLE``.
    """

    def __init__(
        self,
        engine: DaemonEngine,
        on_shutdown_requested: Callable[[], None] | None = None,
    ):
        self._engine = engine
        self._on_shutdown_requested = on_shutdown_requested

    async def simulate(
        self,
        request: runtime_pb2.SimulationRequest,
        context: grpc.aio.ServicerContext,
    ) -> runtime_pb2.SimulationReturn:
        try:
            return await self._engine.simulate(request)
        except InvalidRequestError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except DaemonUnavailableError as exc:
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))

        raise RuntimeError("context.abort did not terminate request")

    async def shut_down(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.Empty:
        if self._on_shutdown_requested is not None:
            self._on_shutdown_requested()
        return common_pb2.Empty()
