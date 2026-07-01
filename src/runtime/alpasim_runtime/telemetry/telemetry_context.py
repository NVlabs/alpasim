# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
TelemetryContext for runtime metrics collection using Prometheus.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from types import TracebackType
from typing import Generator, Type

from alpasim_runtime.event_loop_idle_profiler import get_event_loop_idle_stats
from alpasim_runtime.gc_pressure_profiler import get_gc_pressure_stats
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

logger = logging.getLogger(__name__)

WORKER_LABELS = ("worker_id",)

# Histogram bucket definitions (centralized)
HISTOGRAM_BUCKETS = {
    "rpc_duration": (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    "rpc_blocking": (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    "rollout_duration": list(range(1, 300)),
    "step_duration": (0.1, 0.5, 1, 2, 5, 10, 30),
}


@dataclass
class TelemetryContext:
    """
    All Prometheus metrics and state in one place.

    Use as a context manager for automatic setup/shutdown:

        async with TelemetryContext(worker_id) as ctx:
            # ctx.metrics available here
            await run_simulation()
    """

    worker_id: int = 0
    bind_host: str = "0.0.0.0"
    port: int | None = None
    refresh_interval_s: float = 1.0

    # Metrics (initialized in __post_init__)
    registry: CollectorRegistry = field(init=False)
    rpc_duration: Histogram = field(init=False)
    rpc_blocking: Histogram = field(init=False)
    rpc_queue_depth_latest: Gauge = field(init=False)
    rollout_duration: Histogram = field(init=False)
    step_duration: Histogram = field(init=False)

    # Simulation summary metrics
    simulation_elapsed_seconds: Gauge = field(init=False)
    simulation_rollouts_completed: Counter = field(init=False)

    # Event loop gauges
    event_loop_idle_seconds: Gauge = field(init=False)
    event_loop_poll_seconds: Gauge = field(init=False)
    event_loop_work_seconds: Gauge = field(init=False)

    # GC pressure gauges
    gc_total_duration_seconds: Gauge = field(init=False)
    gc_max_duration_seconds: Gauge = field(init=False)
    gc_collection_count: Gauge = field(init=False)

    _httpd: object | None = field(init=False, default=None)
    _http_thread: object | None = field(init=False, default=None)
    _refresh_task: asyncio.Task[None] | None = field(init=False, default=None)
    _simulation_started_at: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.registry = CollectorRegistry()
        worker_labels = {"worker_id": str(self.worker_id)}
        # RPC metrics (tag label allows filtering warmup/special operations)
        self.rpc_duration = Histogram(
            "alpasim_rpc_duration_seconds",
            "RPC call duration",
            ["service", "method", "tag", *WORKER_LABELS],
            buckets=HISTOGRAM_BUCKETS["rpc_duration"],
            registry=self.registry,
        )
        self.rpc_blocking = Histogram(
            "alpasim_rpc_blocking_seconds",
            "Time between gRPC I/O completion and coroutine resumption",
            ["service", "method", "tag", *WORKER_LABELS],
            buckets=HISTOGRAM_BUCKETS["rpc_blocking"],
            registry=self.registry,
        )
        self.rpc_queue_depth_latest = Gauge(
            "alpasim_rpc_queue_depth_at_start_latest",
            "Latest observed queue depth when an RPC was initiated",
            ["service", "tag", *WORKER_LABELS],
            registry=self.registry,
        )

        # Simulation timing
        self.rollout_duration = Histogram(
            "alpasim_rollout_duration_seconds",
            "Total rollout execution time",
            WORKER_LABELS,
            buckets=HISTOGRAM_BUCKETS["rollout_duration"],
            registry=self.registry,
        ).labels(**worker_labels)
        self.step_duration = Histogram(
            "alpasim_step_duration_seconds",
            "Per-step execution time",
            WORKER_LABELS,
            buckets=HISTOGRAM_BUCKETS["step_duration"],
            registry=self.registry,
        ).labels(**worker_labels)

        # Pre-register simulation summary metrics
        self.simulation_elapsed_seconds = Gauge(
            "alpasim_simulation_elapsed_seconds",
            "Simulation worker elapsed time sampled when rollouts complete",
            registry=self.registry,
            labelnames=WORKER_LABELS,
        ).labels(**worker_labels)
        self.simulation_rollouts_completed = Counter(
            "alpasim_simulation_rollouts_completed",
            "Number of completed rollouts",
            registry=self.registry,
            labelnames=WORKER_LABELS,
        ).labels(**worker_labels)

        # Pre-register event loop gauges
        self.event_loop_idle_seconds = Gauge(
            "alpasim_event_loop_idle_seconds_total",
            "Total event loop idle time (blocking waits for I/O)",
            registry=self.registry,
            labelnames=WORKER_LABELS,
        ).labels(**worker_labels)
        self.event_loop_poll_seconds = Gauge(
            "alpasim_event_loop_poll_seconds_total",
            "Total event loop poll time (non-blocking I/O checks)",
            registry=self.registry,
            labelnames=WORKER_LABELS,
        ).labels(**worker_labels)
        self.event_loop_work_seconds = Gauge(
            "alpasim_event_loop_work_seconds_total",
            "Total event loop work time (executing Python code)",
            registry=self.registry,
            labelnames=WORKER_LABELS,
        ).labels(**worker_labels)

        # GC pressure gauges
        self.gc_total_duration_seconds = Gauge(
            "alpasim_gc_total_duration_seconds",
            "Total time spent in garbage collection",
            registry=self.registry,
            labelnames=WORKER_LABELS,
        ).labels(**worker_labels)
        self.gc_max_duration_seconds = Gauge(
            "alpasim_gc_max_duration_seconds",
            "Longest single garbage collection pause",
            registry=self.registry,
            labelnames=WORKER_LABELS,
        ).labels(**worker_labels)
        self.gc_collection_count = Gauge(
            "alpasim_gc_collection_count_total",
            "Total number of GC collections",
            registry=self.registry,
            labelnames=WORKER_LABELS,
        ).labels(**worker_labels)

    def record_rollout_complete(self) -> None:
        """Record one completed rollout in the live simulation summary."""
        self.simulation_rollouts_completed.inc()
        if self._simulation_started_at is not None:
            self.simulation_elapsed_seconds.set(
                perf_counter() - self._simulation_started_at
            )

    def refresh_gauges(self) -> None:
        """Refresh live gauge snapshots for Prometheus scrapes."""
        self._refresh_event_loop_gauges()
        self._refresh_gc_pressure_gauges()

    def _refresh_event_loop_gauges(self) -> None:
        idle_stats = get_event_loop_idle_stats()
        self.event_loop_idle_seconds.set(idle_stats["idle_seconds"])
        self.event_loop_poll_seconds.set(idle_stats["poll_seconds"])
        self.event_loop_work_seconds.set(idle_stats["work_seconds"])

    def _refresh_gc_pressure_gauges(self) -> None:
        gc_stats = get_gc_pressure_stats()
        self.gc_total_duration_seconds.set(gc_stats["total_duration_s"])
        self.gc_max_duration_seconds.set(gc_stats["max_duration_s"])
        self.gc_collection_count.set(gc_stats["collection_count"])

    async def _refresh_gauges_periodically(self) -> None:
        while True:
            self.refresh_gauges()
            await asyncio.sleep(self.refresh_interval_s)

    async def __aenter__(self) -> "TelemetryContext":
        if self.port is None:
            raise ValueError(f"Telemetry port missing for worker {self.worker_id}")
        try:
            self._httpd, self._http_thread = start_http_server(
                self.port,
                addr=self.bind_host,
                registry=self.registry,
            )
            logger.info(
                "Worker %d metrics endpoint listening on %s:%d",
                self.worker_id,
                self.bind_host,
                self.port,
            )
            self._simulation_started_at = perf_counter()
            self.refresh_gauges()
            self._refresh_task = asyncio.create_task(
                self._refresh_gauges_periodically()
            )
        except BaseException:
            if self._httpd is not None:
                self._httpd.shutdown()
                self._httpd.server_close()
            self._httpd = None
            self._http_thread = None
            raise
        _current_context.set(self)
        return self

    async def __aexit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task
            self._refresh_task = None

        try:
            self.refresh_gauges()
        finally:
            if self._httpd is not None:
                self._httpd.shutdown()
                self._httpd.server_close()
                self._httpd = None
                self._http_thread = None
            _current_context.set(None)


# Task-local context using ContextVar (async-safe)
_current_context: ContextVar[TelemetryContext | None] = ContextVar(
    "telemetry_context", default=None
)


def get_context() -> TelemetryContext:
    """Get current telemetry context. Raises if not inside a TelemetryContext."""
    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError(
            "Not inside a TelemetryContext. Use 'async with TelemetryContext(...)'"
        )
    return ctx


def try_get_context() -> TelemetryContext | None:
    """Get current telemetry context, or None if not inside one.

    Use when telemetry is optional, e.g. for functions that might be in tests.
    """
    return _current_context.get()


# Task-local tag for labeling telemetry samples (e.g., "warmup")
_current_tag: ContextVar[str] = ContextVar("telemetry_tag", default="default")


def get_telemetry_tag() -> str:
    """Get current telemetry tag."""
    return _current_tag.get()


@contextlib.contextmanager
def tag_telemetry(tag: str) -> Generator[None, None, None]:
    """
    Context manager to tag telemetry samples with a label.

    Tagged samples are still recorded but can be filtered in analysis.
    Use this for operations like warmup that should be tracked separately.

    Example:
        with tag_telemetry("warmup"):
            await some_warmup_operation()  # Recorded with tag="warmup"
    """
    old_tag = _current_tag.get()
    _current_tag.set(tag)
    try:
        yield
    finally:
        _current_tag.set(old_tag)
