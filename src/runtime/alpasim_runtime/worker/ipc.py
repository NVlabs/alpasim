# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Inter Process Communication (IPC) message types and helpers for worker pool communication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from multiprocessing import Queue
from queue import Empty as QueueEmpty
from typing import Optional, Union

from alpasim_runtime.config import ScenarioConfig
from alpasim_runtime.telemetry.rpc_wrapper import SharedRpcTracking

from eval.scenario_evaluator import ScenarioEvalResult
from eval.schema import EvalConfig

logger = logging.getLogger(__name__)

# Timeout constants
JOB_POLL_TIMEOUT_S = 10.0
RESULT_POLL_TIMEOUT_S = 30.0


@dataclass
class RolloutJob:
    """Job sent from parent to worker via job_queue."""

    # Unique identifier for tracking this job in results and logs
    job_id: str
    # Scenario configuration defining map, traffic, ego policy, and simulation parameters
    scenario: ScenarioConfig
    # Random seed for deterministic simulation reproducibility
    seed: int


@dataclass
class JobResult:
    """Result sent from worker to parent via result_queue."""

    job_id: str
    success: bool
    error: Optional[str]
    error_traceback: Optional[str]  # Full traceback for debugging
    rollout_uuid: Optional[str]
    # Evaluation metrics from in-runtime evaluation (if enabled).
    # Contains timestep_metrics, aggregated_metrics, and metrics_df.
    # None if evaluation is disabled or failed.
    eval_result: Optional[ScenarioEvalResult] = None


class _ShutdownSentinel:
    """Unique sentinel class for shutdown signal. Distinct from None (timeout returns None)."""

    pass


SHUTDOWN_SENTINEL = _ShutdownSentinel()


@dataclass
class ServiceAllocations:
    """
    Per-service address allocations for a worker.

    Each dict maps address -> number of concurrent slots for that address.
    Empty dict means the service is in skip mode or worker has no allocation.
    """

    driver: dict[str, int] = field(default_factory=dict)
    sensorsim: dict[str, int] = field(default_factory=dict)
    physics: dict[str, int] = field(default_factory=dict)
    trafficsim: dict[str, int] = field(default_factory=dict)
    controller: dict[str, int] = field(default_factory=dict)

    def get_capacity(self) -> int:
        """Return minimum capacity across all services (determines max concurrent rollouts)."""
        capacities = [
            sum(self.driver.values()),
            sum(self.sensorsim.values()),
            sum(self.physics.values()),
            sum(self.trafficsim.values()),
            sum(self.controller.values()),
        ]
        # Filter out zero capacities (skip mode services)
        non_zero = [c for c in capacities if c > 0]
        # Warn if non-zero capacities are mismatched (suboptimal allocation)
        if len(set(non_zero)) > 1:
            logger.warning(
                "Worker has mismatched service capacities: "
                "driver=%d, sensorsim=%d, physics=%d, trafficsim=%d, controller=%d. "
                "This may indicate suboptimal instance allocation.",
                capacities[0],
                capacities[1],
                capacities[2],
                capacities[3],
                capacities[4],
            )
        return min(non_zero) if non_zero else 0


@dataclass
class WorkerArgs:
    """
    Arguments passed to worker process.
    Using a dataclass instead of positional args for maintainability and type safety.
    """

    worker_id: int
    num_workers: int
    job_queue: Queue  # Queue[RolloutJob | _ShutdownSentinel]
    result_queue: Queue  # Queue[JobResult]
    allocations: ServiceAllocations  # Pre-computed service allocations for this worker
    user_config_path: str  # Needed for user config (scenarios, endpoints, etc.)
    usdz_glob: str
    log_dir: str  # Root directory for outputs (asl/, metrics/, txt-logs/)
    eval_config: EvalConfig
    # For orphan detection in subprocess mode. None disables detection (inline mode).
    parent_pid: Optional[int] = None
    # Shared RPC tracking for global queue depth metrics across processes
    shared_rpc_tracking: Optional[SharedRpcTracking] = None


def poll_job_queue(
    job_queue: Queue,
) -> Union[RolloutJob, _ShutdownSentinel, None]:
    """Poll the job queue with timeout. Returns None on timeout."""
    try:
        return job_queue.get(timeout=JOB_POLL_TIMEOUT_S)
    except QueueEmpty:
        return None


def poll_result_queue(
    result_queue: Queue,
) -> Optional[JobResult]:
    """Poll the result queue with timeout. Returns None on timeout."""
    try:
        return result_queue.get(timeout=RESULT_POLL_TIMEOUT_S)
    except QueueEmpty:
        return None
