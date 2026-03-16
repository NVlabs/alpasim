# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Inter Process Communication (IPC) message types and helpers for worker pool communication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import Queue

from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import ServiceAddress
from alpasim_runtime.telemetry.rpc_wrapper import SharedRpcTracking

from eval.scenario_evaluator import ScenarioEvalResult
from eval.schema import EvalConfig

logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoints:
    """Concrete service addresses assigned to a single job."""

    driver: ServiceAddress
    sensorsim: ServiceAddress
    physics: ServiceAddress
    trafficsim: ServiceAddress
    controller: ServiceAddress


@dataclass
class PendingRolloutJob:
    """Job created by parent before service assignment."""

    # Unique identifier for tracking this job in results and logs
    job_id: str
    # Scene ID identifying which scene to simulate
    scene_id: str
    # Index of rollout spec in SimulationRequest.rollout_specs
    rollout_spec_index: int
    # Artifact source path for this job's scene.
    artifact_path: str


@dataclass
class AssignedRolloutJob:
    """Job sent from parent to worker via job_queue."""

    # Request identifier for daemon-mode routing and batch compatibility.
    request_id: str
    # Unique identifier for tracking this job in results and logs
    job_id: str
    # Scene ID identifying which scene to simulate
    scene_id: str
    # Index of rollout spec in SimulationRequest.rollout_specs
    rollout_spec_index: int
    # Artifact source path for this job's scene.
    artifact_path: str
    # Concrete service addresses assigned by the parent dispatch loop.
    endpoints: ServiceEndpoints


@dataclass
class JobResult:
    """Result sent from worker to parent via result_queue."""

    # Request identifier used to route results to waiting callers.
    request_id: str
    job_id: str
    # Index of rollout spec in SimulationRequest.rollout_specs
    rollout_spec_index: int
    success: bool
    error: str | None
    error_traceback: str | None  # Full traceback for debugging
    rollout_uuid: str | None
    # Evaluation metrics from in-runtime evaluation (if enabled).
    # Contains timestep_metrics, aggregated_metrics, and metrics_df.
    # None if evaluation is disabled or failed.
    eval_result: ScenarioEvalResult | None = None


class _ShutdownSentinel:
    """Unique sentinel class for shutdown signal. Distinct from None (timeout returns None)."""

    pass


SHUTDOWN_SENTINEL = _ShutdownSentinel()


@dataclass
class WorkerArgs:
    """Arguments passed to a subprocess worker."""

    worker_id: int
    num_workers: int
    job_queue: Queue  # Queue[AssignedRolloutJob | _ShutdownSentinel]
    result_queue: Queue  # Queue[JobResult]
    num_consumers: int  # Number of concurrent consumer tasks for this worker
    user_config_path: (
        str  # Needed for user config (simulation_config, scenes, endpoints, etc.)
    )
    log_dir: str  # Root directory for outputs (asl/, metrics/, txt-logs/)
    eval_config: EvalConfig
    # Canonical version IDs computed once by the parent process.
    version_ids: RolloutMetadata.VersionIds | None = None
    # For orphan detection in subprocess mode. None disables detection (inline mode).
    parent_pid: int | None = None
    # Shared RPC tracking for global queue depth metrics across processes
    shared_rpc_tracking: SharedRpcTracking | None = None
