# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable
from uuid import uuid4

from alpasim_grpc.v0 import logging_pb2, runtime_pb2
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.daemon.scheduler import DaemonScheduler, DaemonUnavailableError
from alpasim_runtime.runtime_context import (
    build_runtime_context,
    compute_num_consumers_per_worker,
)
from alpasim_runtime.scene_loader import SceneLoader
from alpasim_runtime.worker.ipc import JobResult, PendingRolloutJob
from alpasim_runtime.worker.runtime import WorkerRuntime, start_worker_runtime
from alpasim_utils.scene_data_source import SceneDataSource

from eval.data import AggregationType

logger = logging.getLogger(__name__)

_AGGREGATION_TYPE_TO_PROTO: dict[
    AggregationType, runtime_pb2.TimeAggregation.ValueType
] = {
    AggregationType.MEAN: runtime_pb2.TIME_AGGREGATION_MEAN,
    AggregationType.MEDIAN: runtime_pb2.TIME_AGGREGATION_MEDIAN,
    AggregationType.MAX: runtime_pb2.TIME_AGGREGATION_MAX,
    AggregationType.MIN: runtime_pb2.TIME_AGGREGATION_MIN,
    AggregationType.LAST: runtime_pb2.TIME_AGGREGATION_LAST,
}


def _build_timestep_metrics(
    result: JobResult,
) -> list[runtime_pb2.SimulationReturn.TimestepMetric]:
    """Convert eval timestep metrics to proto TimestepMetric messages."""
    if not result.eval_result:
        return []
    return [
        runtime_pb2.SimulationReturn.TimestepMetric(
            name=m.name,
            timestamps_us=m.timestamps_us,
            values=[float(v) for v in m.values],
            valid=m.valid,
            time_aggregation=_AGGREGATION_TYPE_TO_PROTO[m.time_aggregation],
        )
        for m in result.eval_result.timestep_metrics
    ]


def _build_aggregated_metrics(result: JobResult) -> dict[str, float]:
    """Extract aggregated metrics dict from a job result."""
    if not result.eval_result:
        return {}
    return result.eval_result.aggregated_metrics


def build_simulation_return(
    *,
    request: runtime_pb2.SimulationRequest,
    version_ids: logging_pb2.RolloutMetadata.VersionIds,
    results: list[JobResult],
) -> runtime_pb2.SimulationReturn:
    """Build a SimulationReturn proto by grouping job results back to their rollout specs.

    Each result is matched to its originating RolloutSpec via ``rollout_spec_index``,
    and the response is annotated with the service version IDs from the runtime context.
    Evaluation metrics (per-timestep and aggregated) are included when available.
    """
    results_by_spec_index: dict[int, list[JobResult]] = defaultdict(list)
    for result in results:
        results_by_spec_index[result.rollout_spec_index].append(result)

    rollout_returns: list[runtime_pb2.SimulationReturn.RolloutReturn] = []
    for spec_index, rollout_spec in enumerate(request.rollout_specs):
        for result in results_by_spec_index.get(spec_index, []):
            rollout_returns.append(
                runtime_pb2.SimulationReturn.RolloutReturn(
                    rollout_spec=rollout_spec,
                    success=result.success,
                    rollout_uuid=result.rollout_uuid or "",
                    error=result.error or "",
                    timestep_metrics=_build_timestep_metrics(result),
                    aggregated_metrics=_build_aggregated_metrics(result),
                )
            )

    return runtime_pb2.SimulationReturn(
        runtime_version=version_ids.runtime_version,
        nre_version=version_ids.sensorsim_version,
        physics_version=version_ids.physics_version,
        driver_version=version_ids.egodriver_version,
        traffic_version=version_ids.traffic_version,
        rollout_returns=rollout_returns,
    )


class InvalidRequestError(ValueError):
    """Raised when a simulation request contains invalid parameters."""

    pass


class UnknownSceneError(InvalidRequestError):
    """Raised when a simulation request references a scene_id with no known data source."""

    def __init__(self, scene_id: str):
        super().__init__(f"No data source found for scene_id: {scene_id}")
        self.scene_id = scene_id


def build_pending_jobs_from_request(
    request: runtime_pb2.SimulationRequest,
    get_data_source: Callable[[str], SceneDataSource],
) -> list[PendingRolloutJob]:
    """Expand a SimulationRequest into individual PendingRolloutJob entries.

    Each RolloutSpec is expanded by its ``nr_rollouts`` count.  Specs with
    ``nr_rollouts=0`` are silently dropped with a warning.

    Args:
        request: The simulation request to expand.
        get_data_source: Callable that returns a SceneDataSource for a given scene_id.
            Should raise UnknownSceneError if the scene_id is not found.

    Raises:
        UnknownSceneError: If a spec references an unknown scene_id.
    """
    jobs: list[PendingRolloutJob] = []
    for spec_index, spec in enumerate(request.rollout_specs):
        # This will raise UnknownSceneError if scene_id is not found
        data_source = get_data_source(spec.scenario_id)

        if spec.nr_rollouts == 0:
            logger.warning(
                "Dropping rollout spec with nr_rollouts=0 for scene_id=%s",
                spec.scenario_id,
            )
            continue

        for _ in range(spec.nr_rollouts):
            jobs.append(
                PendingRolloutJob(
                    job_id=uuid4().hex,
                    scene_id=spec.scenario_id,
                    rollout_spec_index=spec_index,
                    data_source=data_source,
                )
            )
    return jobs


class DaemonEngine:
    """Core simulation orchestrator for daemon mode.

    Manages the full lifecycle of a long-running runtime process: startup
    (config parsing, service probing, worker pool creation), simulation
    request dispatch (job building, scheduling, result collection), and
    graceful shutdown.

    The engine is safe to call ``startup`` on multiple times (idempotent)
    and ``shutdown`` will drain pending work before stopping workers.
    """

    def __init__(
        self,
        *,
        user_config: str,
        network_config: str,
        eval_config: str,
        log_dir: str,
        validate_config_scenes: bool = True,
    ) -> None:
        self._user_config_path = user_config
        self._network_config_path = network_config
        self._eval_config_path = eval_config
        self._log_dir = log_dir
        self._validate_config_scenes = validate_config_scenes

        self._version_ids: logging_pb2.RolloutMetadata.VersionIds | None = None
        self._config = None  # Will be set during startup
        self._scene_loader: SceneLoader | None = None
        self._scheduler: DaemonScheduler | None = None
        self._worker_runtime: WorkerRuntime | None = None
        self._started = False

    @property
    def version_ids(self) -> logging_pb2.RolloutMetadata.VersionIds:
        if self._version_ids is None:
            raise RuntimeError("daemon is not started")
        return self._version_ids

    def _get_data_source(self, scene_id: str) -> SceneDataSource:
        """Get or create a data source for the given scene_id.

        Delegates to SceneLoader for lazy loading and caching.

        Args:
            scene_id: Scene identifier to load

        Returns:
            SceneDataSource for the requested scene

        Raises:
            RuntimeError: If SceneLoader not initialized
        """
        if self._scene_loader is None:
            raise RuntimeError("SceneLoader not initialized")
        return self._scene_loader.get_data_source(scene_id)

    async def startup(self) -> None:
        """Initialize the runtime context, start workers, and begin scheduling.

        Builds the RuntimeContext (parses configs, probes service versions,
        validates scenarios, creates scene mapping from trajdata), then creates
        the worker runtime and daemon scheduler.  Idempotent: subsequent calls
        after the first are no-ops.
        """
        if self._started:
            return

        runtime_context = await build_runtime_context(
            user_config_path=self._user_config_path,
            network_config_path=self._network_config_path,
            eval_config_path=self._eval_config_path,
            validate_config_scenes=self._validate_config_scenes,
        )

        # Create SceneLoader from RuntimeContext
        self._scene_loader = SceneLoader(
            dataset=runtime_context.dataset,
            scene_id_to_idx=runtime_context.scene_id_to_idx,
            config=runtime_context.config,
        )
        self._config = runtime_context.config

        num_consumers_per_worker = compute_num_consumers_per_worker(
            max_in_flight=runtime_context.max_in_flight,
            nr_workers=runtime_context.config.user.nr_workers,
        )

        worker_runtime = start_worker_runtime(
            config=runtime_context.config,
            user_config_path=self._user_config_path,
            num_consumers=num_consumers_per_worker,
            log_dir=self._log_dir,
            eval_config=runtime_context.eval_config,
            version_ids=runtime_context.version_ids,
        )

        scheduler = DaemonScheduler(
            pools=runtime_context.pools,
            runtime=worker_runtime,
        )

        self._version_ids = runtime_context.version_ids
        self._worker_runtime = worker_runtime
        self._scheduler = scheduler
        self._started = True

    async def simulate(
        self,
        request: runtime_pb2.SimulationRequest,
    ) -> runtime_pb2.SimulationReturn:
        """Run a simulation request and return results.

        Expands the request into jobs, optionally creates a per-request driver
        pool from ``request.available_drivers``, submits everything to the
        scheduler, and awaits completion.

        Raises:
            DaemonUnavailableError: If the engine has not been started.
            UnknownSceneError: If the request references an unknown scene_id.
        """
        if not self._started:
            raise DaemonUnavailableError("daemon is not started")
        assert self._scheduler is not None

        request_id = uuid4().hex

        # Use instance method for getting data sources
        jobs = build_pending_jobs_from_request(request, self._get_data_source)

        driver_pool: AddressPool | None = None
        if request.available_drivers:
            if request.n_concurrent_per_driver < 1:
                raise InvalidRequestError(
                    "n_concurrent_per_driver must be >= 1 when available_drivers is provided"
                )
            addresses = [f"{d.ip}:{d.port}" for d in request.available_drivers]
            driver_pool = AddressPool(
                addresses, n_concurrent=request.n_concurrent_per_driver, skip=False
            )

        await self._scheduler.submit_request(
            request_id,
            jobs,
            driver_pool=driver_pool,
        )
        results = await self._scheduler.wait_request(request_id)

        return build_simulation_return(
            request=request,
            version_ids=self.version_ids,
            results=results,
        )

    async def shutdown(self) -> None:
        """Gracefully shut down the engine.

        Stops accepting new requests, drains pending (not yet dispatched) jobs
        by failing their associated requests, stops the scheduler's dispatch loop,
        and finally stops all workers.  Idempotent: no-op if not started.
        """
        if not self._started:
            return

        assert self._scheduler is not None
        assert self._worker_runtime is not None

        await self._scheduler.shutdown(reason="daemon shutting down")
        await self._worker_runtime.stop()

        self._scheduler = None
        self._worker_runtime = None
        self._started = False
