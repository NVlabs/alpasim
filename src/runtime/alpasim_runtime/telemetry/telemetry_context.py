# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
TelemetryContext for runtime metrics collection using Prometheus.
"""

from __future__ import annotations

import contextlib
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Generator, Optional, Type

from alpasim_runtime.event_loop_idle_profiler import get_event_loop_idle_stats
from prometheus_client import CollectorRegistry, Gauge, Histogram, write_to_textfile

from .resources import ResourceSampler

logger = logging.getLogger(__name__)

# Histogram bucket definitions (centralized)
HISTOGRAM_BUCKETS = {
    "rpc_duration": (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
    "rpc_blocking": (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    "rpc_queue_depth": [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 50],
    "rollout_duration": list(range(1, 300)),
    "step_duration": (0.1, 0.5, 1, 2, 5, 10, 30),
}


@dataclass
class TelemetryContext:
    """
    All Prometheus metrics and state in one place.

    Use as a context manager for automatic setup/shutdown:

        async with TelemetryContext(output_dir, worker_id) as ctx:
            # ctx.metrics available here
            await run_simulation()
        # Metrics automatically written on exit

    Resource sampling: Make sure to only sample resources
    (i.e. sample_resources=True) for one process in the simulation.
    """

    output_dir: str
    worker_id: int = 0
    sample_resources: bool = False
    resource_sample_interval: float = 1.0

    # Metrics (initialized in __post_init__)
    registry: CollectorRegistry = field(init=False)
    rpc_duration: Histogram = field(init=False)
    rpc_blocking: Histogram = field(init=False)
    rpc_queue_depth: Histogram = field(init=False)
    rollout_duration: Histogram = field(init=False)
    step_duration: Histogram = field(init=False)

    # Note that because we don't have an active Prometheus server,
    # Gauges are only set once at the end of the simulation.
    # Simulation summary gauges
    simulation_total_seconds: Gauge = field(init=False)
    simulation_rollout_count: Gauge = field(init=False)
    simulation_seconds_per_rollout: Gauge = field(init=False)

    # Event loop gauges
    event_loop_idle_seconds: Gauge = field(init=False)
    event_loop_poll_seconds: Gauge = field(init=False)
    event_loop_work_seconds: Gauge = field(init=False)

    # Resource sampler (owned by context)
    _resource_sampler: Optional[ResourceSampler] = field(init=False, default=None)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.registry = CollectorRegistry()

        # RPC metrics (tag label allows filtering warmup/special operations)
        self.rpc_duration = Histogram(
            "rpc_duration_seconds",
            "RPC call duration",
            ["service", "method", "tag"],
            buckets=HISTOGRAM_BUCKETS["rpc_duration"],
            registry=self.registry,
        )
        self.rpc_blocking = Histogram(
            "rpc_blocking_seconds",
            "Time between gRPC I/O completion and coroutine resumption",
            ["service", "method", "tag"],
            buckets=HISTOGRAM_BUCKETS["rpc_blocking"],
            registry=self.registry,
        )
        self.rpc_queue_depth = Histogram(
            "rpc_queue_depth_at_start",
            "Queue depth when RPC was initiated",
            ["service", "tag"],
            buckets=HISTOGRAM_BUCKETS["rpc_queue_depth"],
            registry=self.registry,
        )

        # Simulation timing
        self.rollout_duration = Histogram(
            "rollout_duration_seconds",
            "Total rollout execution time",
            [],
            buckets=HISTOGRAM_BUCKETS["rollout_duration"],
            registry=self.registry,
        )
        self.step_duration = Histogram(
            "step_duration_seconds",
            "Per-step execution time",
            [],
            buckets=HISTOGRAM_BUCKETS["step_duration"],
            registry=self.registry,
        )

        # Pre-register simulation summary gauges
        self.simulation_total_seconds = Gauge(
            "simulation_total_seconds",
            "Total simulation time",
            registry=self.registry,
        )
        self.simulation_rollout_count = Gauge(
            "simulation_rollout_count",
            "Number of rollouts",
            registry=self.registry,
        )
        self.simulation_seconds_per_rollout = Gauge(
            "simulation_seconds_per_rollout",
            "Average time per rollout",
            registry=self.registry,
        )

        # Pre-register event loop gauges
        self.event_loop_idle_seconds = Gauge(
            "event_loop_idle_seconds_total",
            "Total event loop idle time (blocking waits for I/O)",
            registry=self.registry,
        )
        self.event_loop_poll_seconds = Gauge(
            "event_loop_poll_seconds_total",
            "Total event loop poll time (non-blocking I/O checks)",
            registry=self.registry,
        )
        self.event_loop_work_seconds = Gauge(
            "event_loop_work_seconds_total",
            "Total event loop work time (executing Python code)",
            registry=self.registry,
        )

    def record_simulation_summary(
        self, total_seconds: float, rollout_count: int
    ) -> None:
        """Record simulation summary metrics (called once at end of run)."""
        self.simulation_total_seconds.set(total_seconds)
        self.simulation_rollout_count.set(rollout_count)
        if rollout_count > 0:
            self.simulation_seconds_per_rollout.set(total_seconds / rollout_count)

    def shutdown(self) -> None:
        """Dump all metrics to file."""
        prom_path = Path(self.output_dir) / f"metrics_worker_{self.worker_id}.prom"
        write_to_textfile(str(prom_path), self.registry)
        logger.info(f"Metrics written to {prom_path}")

    async def __aenter__(self) -> "TelemetryContext":
        _current_context.set(self)
        if self.sample_resources:
            self._resource_sampler = ResourceSampler()
            await self._resource_sampler.start(self.resource_sample_interval)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._resource_sampler:
            await self._resource_sampler.stop()
            self._resource_sampler.export_to(self.registry)

        # Record event loop stats
        idle_stats = get_event_loop_idle_stats()
        self.event_loop_idle_seconds.set(idle_stats["idle_seconds"])
        self.event_loop_poll_seconds.set(idle_stats["poll_seconds"])
        self.event_loop_work_seconds.set(idle_stats["work_seconds"])

        self.shutdown()
        _current_context.set(None)


# Task-local context using ContextVar (async-safe)
_current_context: ContextVar[Optional[TelemetryContext]] = ContextVar(
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


def try_get_context() -> Optional[TelemetryContext]:
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
