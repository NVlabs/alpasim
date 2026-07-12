# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Wrapper for profiling gRPC calls with queue depth and blocking time tracking.

Supports both single-process and multiprocessing modes. For multiprocessing,
call init_shared_rpc_tracking() in the main process before spawning workers,
then call set_shared_rpc_tracking() in each worker with the returned values.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from typing import Any, Callable, MutableMapping

import grpc

from .telemetry_context import try_get_context

logger = logging.getLogger(__name__)

# Type alias for lock-like objects (context managers)
LockLike = AbstractContextManager[bool]

# Tuple type for passing shared state between processes
SharedRpcTracking = tuple[MutableMapping[str, int], LockLike]


# Module-level state - defaults to local dict, can be upgraded to shared
_active_rpc_counts: MutableMapping[str, int] = {}
_rpc_lock: LockLike = threading.Lock()
_manager: SyncManager | None = None


def init_shared_rpc_tracking() -> SharedRpcTracking:
    """
    Initialize shared RPC tracking for multiprocessing.

    Call this in the main process before spawning workers.
    Returns SharedRpcTracking tuple to pass to worker processes.

    Example:
        # In main process
        shared_tracking = init_shared_rpc_tracking()

        # Pass to workers via initializer or args
        process = Process(target=worker, args=(shared_tracking,))
    """
    global _active_rpc_counts, _rpc_lock, _manager
    _manager = Manager()
    _active_rpc_counts = _manager.dict()
    _rpc_lock = _manager.Lock()
    return _active_rpc_counts, _rpc_lock


def set_shared_rpc_tracking(shared: SharedRpcTracking) -> None:
    """
    Set the shared RPC tracking state in a worker process.

    Call this at worker initialization with the tuple from init_shared_rpc_tracking().

    Example:
        def worker(shared_tracking):
            set_shared_rpc_tracking(shared_tracking)
            # ... worker code that uses profiled_rpc_call ...
    """
    global _active_rpc_counts, _rpc_lock
    _active_rpc_counts, _rpc_lock = shared


def _increment_count(service_type: str) -> int:
    """Atomically increment and return the old count."""
    with _rpc_lock:
        count = _active_rpc_counts.get(service_type, 0)
        _active_rpc_counts[service_type] = count + 1
        return count


def _decrement_count(service_type: str) -> None:
    """Atomically decrement the count."""
    with _rpc_lock:
        _active_rpc_counts[service_type] = max(
            0, _active_rpc_counts.get(service_type, 0) - 1
        )


def _is_transient_error(
    exc: grpc.aio.AioRpcError,
    transient_error_details: Sequence[str],
) -> bool:
    """Match transport failures or allowlisted server-side transient details."""
    if exc.code() == grpc.StatusCode.UNAVAILABLE:
        return True
    if exc.code() not in (
        grpc.StatusCode.UNKNOWN,
        grpc.StatusCode.DEADLINE_EXCEEDED,
    ):
        return False
    details = exc.details() or ""
    return any(pattern in details for pattern in transient_error_details)


async def profiled_rpc_call(
    method_name: str,
    service_type: str,
    stub_call: Callable[..., grpc.aio.UnaryUnaryCall[Any, Any]],
    *args: Any,
    retry_delays_s: Sequence[float] = (),
    transient_error_details: Sequence[str] = (),
    **kwargs: Any,
) -> Any:
    """
    Wrapper that captures RPC metrics.

    Args:
        method_name: RPC method name used in metrics and logs.
        service_type: Service name used in metrics and logs.
        stub_call: gRPC callable to invoke.
        *args: Positional arguments forwarded to `stub_call`.
        retry_delays_s: Delays before successive transient-error retries.
        transient_error_details: Server error detail substrings. An error is
            retryable when its details contain any entry in this sequence.
        **kwargs: Keyword arguments forwarded to `stub_call`.

    Usage:
        result = await profiled_rpc_call(
            "drive", "driver", self.stub.Drive, request
        )

    If not inside a TelemetryContext, the call still executes but no metrics
    are recorded.

    Each delay in `retry_delays_s` grants one retry after a transient failure:
    UNAVAILABLE, or UNKNOWN/DEADLINE_EXCEEDED whose details match an entry in
    `transient_error_details` (for known server-side transient failures).
    """
    for delay_s in retry_delays_s:
        try:
            return await _profiled_rpc_call_once(
                method_name,
                service_type,
                stub_call,
                *args,
                **kwargs,
            )
        except grpc.aio.AioRpcError as exc:
            if not _is_transient_error(exc, transient_error_details):
                raise
            logger.warning(
                "%s RPC %s failed with %s; retrying in %.1fs: %s",
                service_type,
                method_name,
                exc.code().name,
                delay_s,
                exc.details(),
            )
            await asyncio.sleep(delay_s)

    return await _profiled_rpc_call_once(
        method_name,
        service_type,
        stub_call,
        *args,
        **kwargs,
    )


async def _profiled_rpc_call_once(
    method_name: str,
    service_type: str,
    stub_call: Callable[..., grpc.aio.UnaryUnaryCall[Any, Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute one profiled RPC attempt."""
    ctx = try_get_context()

    fut = stub_call(*args, **kwargs)

    # Track queue depth at start
    queue_depth_at_start = _increment_count(service_type)

    # Initialize timing variables before try block so they're always defined in finally
    t_start = time.perf_counter()
    t_done: float | None = None

    def on_done(_: Any) -> None:
        nonlocal t_done
        t_done = time.perf_counter()

    # Try/finally must immediately follow increment to ensure we always decrement,
    # even if callback registration or timer setup fails.
    try:
        fut.add_done_callback(on_done)
        result = await fut
        return result
    finally:
        _decrement_count(service_type)
        t_resume = time.perf_counter()

        # Record duration
        duration = t_resume - t_start

        if ctx is not None:
            ctx.record_rpc(
                service=service_type,
                method=method_name,
                queue_depth_at_start=queue_depth_at_start,
                duration_seconds=duration,
                blocking_seconds=(
                    max(0, t_resume - t_done) if t_done is not None else None
                ),
            )
