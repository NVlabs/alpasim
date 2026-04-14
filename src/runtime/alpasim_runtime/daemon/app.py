# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import signal
from contextlib import suppress

from alpasim_grpc.v0 import runtime_pb2_grpc
from alpasim_grpc.v1 import interactive_runtime_pb2_grpc
from alpasim_runtime.daemon.engine import DaemonEngine
from alpasim_runtime.daemon.interactive_servicer import InteractiveRuntimeServicer
from alpasim_runtime.daemon.servicer import RuntimeDaemonServicer

import grpc


class RuntimeDaemonApp:
    """Long-running gRPC server that exposes the runtime daemon.

    Manages the lifecycle of a gRPC server backed by a ``DaemonEngine``:
    starts the engine, serves RPCs, and waits for either an OS signal
    (SIGINT/SIGTERM) or a programmatic shutdown request before tearing
    everything down gracefully.
    """

    _GRPC_GRACEFUL_SHUTDOWN_S = 10.0

    def __init__(
        self,
        engine: DaemonEngine,
        listen_address: str,
    ) -> None:
        self._engine = engine
        self._listen_address = listen_address
        self._shutdown_requested = asyncio.Event()

    def request_shutdown(self) -> None:
        """Signal the app to begin graceful shutdown."""
        self._shutdown_requested.set()

    async def _wait_for_shutdown_request(self) -> None:
        await self._shutdown_requested.wait()

    async def _wait_for_signal(self) -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop_event.set)

        await stop_event.wait()

    async def _wait_for_shutdown(self) -> None:
        tasks = [
            asyncio.create_task(self._wait_for_signal()),
            asyncio.create_task(self._wait_for_shutdown_request()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()
        for task in pending:
            with suppress(asyncio.CancelledError):
                await task
        for task in done:
            task.result()

    async def run(self) -> None:
        """Start the engine, serve RPCs, and block until shutdown is requested."""
        await self._engine.startup()
        server: grpc.aio.Server | None = None
        try:
            server = grpc.aio.server()
            runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
                RuntimeDaemonServicer(
                    engine=self._engine,
                    on_shutdown_requested=self.request_shutdown,
                ),
                server,
            )
            interactive_runtime_pb2_grpc.add_InteractiveRuntimeServiceServicer_to_server(
                InteractiveRuntimeServicer(
                    manager=self._engine.interactive_session_manager,
                ),
                server,
            )
            server.add_insecure_port(self._listen_address)
            await server.start()

            await self._wait_for_shutdown()
        finally:
            stop_error: Exception | None = None
            if server is not None:
                try:
                    await server.stop(grace=self._GRPC_GRACEFUL_SHUTDOWN_S)
                except Exception as exc:
                    stop_error = exc

            await self._engine.shutdown()
            if stop_error is not None:
                raise stop_error
