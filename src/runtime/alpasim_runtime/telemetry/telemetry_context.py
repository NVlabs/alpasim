# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
TelemetryContext for runtime metrics collection using Prometheus.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from types import TracebackType
from typing import Callable, Type
from wsgiref.simple_server import WSGIServer

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
    "rollout_duration": (1, 2.5, 5, 10, 25, 50, 100, 250, 500),
    "step_duration": (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    "observation_barrier": (
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
    ),
    "scheduler_wait": (0.01, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
}


@dataclass
class TelemetryContext:
    """
    All Prometheus metrics and state in one place.

    Use as a context manager for automatic setup/shutdown:

        async with TelemetryContext(port=port, worker_id=worker_id) as ctx:
            # ctx.metrics available here
            await run_simulation()
    """

    port: int
    worker_id: int = 0
    bind_host: str = "0.0.0.0"
    refresh_interval_s: float = 1.0
    job_queue_depth_fn: Callable[[], int] | None = None

    # Metrics (initialized in __post_init__)
    registry: CollectorRegistry = field(init=False)
    _rpc_duration: Histogram = field(init=False)
    _rpc_blocking: Histogram = field(init=False)
    _rpc_queue_depth_latest: Gauge = field(init=False)
    _rollout_duration: Histogram = field(init=False)
    _step_duration: Histogram = field(init=False)
    _observation_barrier: Histogram = field(init=False)

    # Simulation summary metrics
    _rollouts: Counter = field(init=False)
    _renderer_active_rollouts: Gauge = field(init=False)
    _renderer_rollouts_started: Counter = field(init=False)
    _renderer_scheduler_wait: Histogram = field(init=False)
    _simulation_elapsed_seconds: Gauge = field(init=False)
    _job_queue_depth: Gauge = field(init=False)

    # Event loop gauges
    _event_loop_idle_seconds: Gauge = field(init=False)
    _event_loop_poll_seconds: Gauge = field(init=False)
    _event_loop_work_seconds: Gauge = field(init=False)

    # GC pressure gauges
    _gc_total_duration_seconds: Gauge = field(init=False)
    _gc_max_duration_seconds: Gauge = field(init=False)
    _gc_collection_count: Gauge = field(init=False)

    _httpd: WSGIServer | None = field(init=False, default=None)
    _http_thread: threading.Thread | None = field(init=False, default=None)
    _refresh_task: asyncio.Task[None] | None = field(init=False, default=None)
    _simulation_started_at: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.registry = CollectorRegistry()
        worker_labels = {"worker_id": str(self.worker_id)}

        def worker_gauge(name: str, description: str) -> Gauge:
            return Gauge(
                name, description, WORKER_LABELS, registry=self.registry
            ).labels(**worker_labels)

        # RPC metrics
        self._rpc_duration = Histogram(
            "alpasim_rpc_duration_seconds",
            "RPC call duration",
            ["service", "method", *WORKER_LABELS],
            buckets=HISTOGRAM_BUCKETS["rpc_duration"],
            registry=self.registry,
        )
        self._rpc_blocking = Histogram(
            "alpasim_rpc_blocking_seconds",
            "Time between gRPC I/O completion and coroutine resumption",
            ["service", "method", *WORKER_LABELS],
            buckets=HISTOGRAM_BUCKETS["rpc_blocking"],
            registry=self.registry,
        )
        self._rpc_queue_depth_latest = Gauge(
            "alpasim_rpc_queue_depth_at_start_latest",
            "Latest observed queue depth when an RPC was initiated",
            ["service", *WORKER_LABELS],
            registry=self.registry,
        )

        # Simulation timing
        self._rollout_duration = Histogram(
            "alpasim_rollout_duration_seconds",
            "Total rollout execution time",
            WORKER_LABELS,
            buckets=HISTOGRAM_BUCKETS["rollout_duration"],
            registry=self.registry,
        ).labels(**worker_labels)
        self._step_duration = Histogram(
            "alpasim_step_duration_seconds",
            "Per-step execution time",
            WORKER_LABELS,
            buckets=HISTOGRAM_BUCKETS["step_duration"],
            registry=self.registry,
        ).labels(**worker_labels)
        self._observation_barrier = Histogram(
            "alpasim_observation_barrier_seconds",
            "Wall time awaiting the pre-drive observation barrier",
            WORKER_LABELS,
            buckets=HISTOGRAM_BUCKETS["observation_barrier"],
            registry=self.registry,
        ).labels(**worker_labels)

        # Pre-register simulation summary metrics
        self._rollouts = Counter(
            "alpasim_rollouts",
            "Number of rollouts reaching a terminal status",
            registry=self.registry,
            labelnames=["status", *WORKER_LABELS],
        )
        self._renderer_active_rollouts = worker_gauge(
            "alpasim_renderer_active_rollouts",
            "Renderer-backed rollouts currently executing",
        )
        self._renderer_rollouts_started = Counter(
            "alpasim_renderer_rollouts_started",
            "Renderer-backed rollouts started by scheduling decision",
            registry=self.registry,
            labelnames=[
                "kind",
                *WORKER_LABELS,
            ],
        )
        self._renderer_scheduler_wait = Histogram(
            "alpasim_renderer_scheduler_wait_seconds",
            "Time a renderer-backed rollout waited for scheduler assignment",
            registry=self.registry,
            labelnames=["kind", *WORKER_LABELS],
            buckets=HISTOGRAM_BUCKETS["scheduler_wait"],
        )
        self._simulation_elapsed_seconds = worker_gauge(
            "alpasim_simulation_elapsed_seconds",
            "Simulation worker elapsed time sampled when rollouts finish",
        )
        self._job_queue_depth = worker_gauge(
            "alpasim_job_queue_depth",
            "Runtime rollout jobs waiting for a worker",
        )

        # Pre-register event loop gauges
        self._event_loop_idle_seconds = worker_gauge(
            "alpasim_event_loop_idle_seconds_total",
            "Total event loop idle time (blocking waits for I/O)",
        )
        self._event_loop_poll_seconds = worker_gauge(
            "alpasim_event_loop_poll_seconds_total",
            "Total event loop poll time (non-blocking I/O checks)",
        )
        self._event_loop_work_seconds = worker_gauge(
            "alpasim_event_loop_work_seconds_total",
            "Total event loop work time (executing Python code)",
        )

        # GC pressure gauges
        self._gc_total_duration_seconds = worker_gauge(
            "alpasim_gc_total_duration_seconds",
            "Total time spent in garbage collection",
        )
        self._gc_max_duration_seconds = worker_gauge(
            "alpasim_gc_max_duration_seconds",
            "Longest single garbage collection pause",
        )
        self._gc_collection_count = worker_gauge(
            "alpasim_gc_collection_count_total",
            "Total number of GC collections",
        )

    def record_rpc(
        self,
        *,
        service: str,
        method: str,
        queue_depth_at_start: int,
        duration_seconds: float,
        blocking_seconds: float | None,
    ) -> None:
        """Record duration and queue depth for one RPC attempt."""
        worker_id = str(self.worker_id)
        self._rpc_queue_depth_latest.labels(service=service, worker_id=worker_id).set(
            queue_depth_at_start
        )
        self._rpc_duration.labels(
            service=service,
            method=method,
            worker_id=worker_id,
        ).observe(duration_seconds)
        if blocking_seconds is not None:
            self._rpc_blocking.labels(
                service=service,
                method=method,
                worker_id=worker_id,
            ).observe(blocking_seconds)

    def record_rollout_duration(self, duration_seconds: float) -> None:
        """Record one rollout's wall-clock duration."""
        self._rollout_duration.observe(duration_seconds)

    def record_step_duration(self, duration_seconds: float) -> None:
        """Record one simulation step's wall-clock duration."""
        self._step_duration.observe(duration_seconds)

    def record_observation_barrier(self, duration_seconds: float) -> None:
        """Record one wait at the pre-drive observation barrier."""
        self._observation_barrier.observe(duration_seconds)

    def record_rollout_finished(self, status: str) -> None:
        """Record one terminal rollout status in the live simulation summary."""
        self._rollouts.labels(
            status=status,
            worker_id=str(self.worker_id),
        ).inc()
        if self._simulation_started_at is not None:
            self._simulation_elapsed_seconds.set(
                perf_counter() - self._simulation_started_at
            )

    def record_renderer_rollout_started(
        self,
        *,
        dispatch_kind: str,
        scheduler_wait_seconds: float,
    ) -> None:
        """Record the scheduling class and begin active-rollout accounting."""
        self._renderer_rollouts_started.labels(
            kind=dispatch_kind,
            worker_id=str(self.worker_id),
        ).inc()
        self._renderer_scheduler_wait.labels(
            kind=dispatch_kind,
            worker_id=str(self.worker_id),
        ).observe(scheduler_wait_seconds)
        self._renderer_active_rollouts.inc()

    def record_renderer_rollout_stopped(self) -> None:
        """Finish active-rollout accounting."""
        self._renderer_active_rollouts.dec()

    def refresh_gauges(self) -> None:
        """Refresh live gauge snapshots for Prometheus scrapes."""
        self._refresh_event_loop_gauges()
        self._refresh_gc_pressure_gauges()
        if self.job_queue_depth_fn is not None:
            self._job_queue_depth.set(self.job_queue_depth_fn())

    def _refresh_event_loop_gauges(self) -> None:
        idle_stats = get_event_loop_idle_stats()
        self._event_loop_idle_seconds.set(idle_stats["idle_seconds"])
        self._event_loop_poll_seconds.set(idle_stats["poll_seconds"])
        self._event_loop_work_seconds.set(idle_stats["work_seconds"])

    def _refresh_gc_pressure_gauges(self) -> None:
        gc_stats = get_gc_pressure_stats()
        self._gc_total_duration_seconds.set(gc_stats["total_duration_s"])
        self._gc_max_duration_seconds.set(gc_stats["max_duration_s"])
        self._gc_collection_count.set(gc_stats["collection_count"])

    async def _refresh_gauges_periodically(self) -> None:
        while True:
            self.refresh_gauges()
            await asyncio.sleep(self.refresh_interval_s)

    async def __aenter__(self) -> "TelemetryContext":
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
