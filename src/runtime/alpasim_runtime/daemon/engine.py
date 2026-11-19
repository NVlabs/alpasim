# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from uuid import uuid4

from alpasim_grpc.v0 import logging_pb2, runtime_pb2
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.config import RendererKind
from alpasim_runtime.daemon.scheduler import DaemonScheduler, DaemonUnavailableError
from alpasim_runtime.data_source_loader import DataSourceLoader
from alpasim_runtime.errors import UnknownClipError, UnknownSceneError
from alpasim_runtime.nre_introspection import prefetch_clip as prefetch_nre_clip
from alpasim_runtime.runtime_context import (
    RuntimeContext,
    build_runtime_context,
    compute_num_consumers_per_worker,
)
from alpasim_runtime.scene_registry import SceneRegistry
from alpasim_runtime.worker.ipc import JobResult, PendingRolloutJob
from alpasim_runtime.worker.runtime import WorkerRuntime, start_worker_runtime
from alpasim_utils.types import T0_AUTO, Scene, scene_from_proto, scene_to_proto

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
    AggregationType.P90: runtime_pb2.TIME_AGGREGATION_P90,
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
                    error_code=result.error_code,
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
        video_model_version=version_ids.video_model_version,
        rollout_returns=rollout_returns,
    )


def _build_scene_metadata_info(metadata) -> runtime_pb2.SceneMetadata:
    return runtime_pb2.SceneMetadata(
        version_string=metadata.version_string,
        training_date=metadata.training_date,
        dataset_hash=metadata.dataset_hash,
        camera_ids=metadata.sensors.camera_ids,
        lidar_ids=metadata.sensors.lidar_ids,
        start_time_us=int(metadata.time_range.start),
        end_time_us=int(metadata.time_range.end),
    )


def build_runtime_info(
    runtime_context: RuntimeContext,
    version_ids: logging_pb2.RolloutMetadata.VersionIds,
) -> runtime_pb2.RuntimeInfo:
    """Build a RuntimeInfo proto from the daemon startup context."""
    data_source_info_by_id = {
        info.source.source_id: info
        for info in runtime_context.data_source_loader.data_source_infos
    }
    scenes = []
    for scene in runtime_context.scene_registry.scenes:
        scene_info = scene_to_proto(scene)
        scene_info.metadata.CopyFrom(
            _build_scene_metadata_info(
                data_source_info_by_id[scene.source.source_id].metadata
            )
        )
        scenes.append(scene_info)

    return runtime_pb2.RuntimeInfo(
        max_supported_concurrent_rollouts=runtime_context.max_in_flight,
        nr_workers=runtime_context.config.user.nr_workers,
        renderer_type=runtime_context.config.user.renderer.kind.value,
        scenes=scenes,
        service_capacities=[
            runtime_pb2.ServiceCapacity(
                service_name=service_name,
                total_capacity=pool.total_capacity or 0,
                skipped=pool.skip,
            )
            for service_name, pool in sorted(runtime_context.pools.items())
        ],
        runtime_version=version_ids.runtime_version,
        nre_version=version_ids.sensorsim_version,
        physics_version=version_ids.physics_version,
        driver_version=version_ids.egodriver_version,
        traffic_version=version_ids.traffic_version,
        video_model_version=version_ids.video_model_version,
    )


class InvalidRequestError(ValueError):
    """Raised when a simulation request contains invalid parameters."""

    pass


def build_pending_jobs_from_request(
    request: runtime_pb2.SimulationRequest,
    request_id: str,
    resolve_scene: Callable[[Scene], Scene],
    resolve_fork_scene: Callable[[Scene], Scene] | None = None,
) -> list[PendingRolloutJob]:
    """Expand a SimulationRequest into individual PendingRolloutJob entries.

    Each RolloutSpec is expanded by its ``nr_rollouts`` count.  Specs with
    ``nr_rollouts=0`` are silently dropped with a warning.

    Args:
        request: The simulation request to expand.
        request_id: Identifier stamped onto every job for result routing.
    Raises:
        UnknownSceneError: If a spec references an unconfigured scene address.
    """
    jobs: list[PendingRolloutJob] = []
    for spec_index, spec in enumerate(request.rollout_specs):
        try:
            requested = scene_from_proto(spec.scene)
            if spec.HasField("ego_history"):
                if not spec.scene.HasField("policy_start_us"):
                    raise ValueError("ego_history requires an explicit policy_start_us")
                if resolve_fork_scene is None:
                    raise ValueError("ego_history is not supported by this runtime")
                resolved = resolve_fork_scene(requested)
            else:
                resolved = resolve_scene(requested)
        except KeyError as exc:
            raise UnknownSceneError(
                requested.source.source_id, requested.policy_start_us
            ) from exc
        except ValueError as exc:
            raise InvalidRequestError(f"RolloutSpec index {spec_index}: {exc}") from exc

        if spec.nr_rollouts == 0:
            logger.warning(
                "Dropping rollout spec with nr_rollouts=0 for source_id=%s, "
                "policy_start_us=%s",
                requested.source.source_id,
                requested.policy_start_us,
            )
            continue

        session_uuids = list(spec.session_uuids)
        if session_uuids and len(session_uuids) != spec.nr_rollouts:
            raise ValueError(
                f"RolloutSpec scene={requested.scene_key!r}: "
                f"len(session_uuids)={len(session_uuids)} != "
                f"nr_rollouts={spec.nr_rollouts}"
            )

        for rollout_idx in range(spec.nr_rollouts):
            jobs.append(
                PendingRolloutJob(
                    request_id=request_id,
                    job_id=uuid4().hex,
                    scene=resolved,
                    rollout_spec_index=spec_index,
                    session_uuid=session_uuids[rollout_idx] if session_uuids else "",
                    session_seed=(
                        spec.random_seed + rollout_idx if spec.random_seed else 0
                    ),
                    ego_history_bytes=(
                        spec.ego_history.SerializeToString()
                        if spec.HasField("ego_history")
                        else None
                    ),
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
        self._runtime_context: RuntimeContext | None = None
        self._data_source_loader: DataSourceLoader | None = None
        self._scene_registry: SceneRegistry | None = None
        self._renderer_kind: RendererKind | None = None
        self._scheduler: DaemonScheduler | None = None
        self._worker_runtime: WorkerRuntime | None = None
        self._clip_prefetch_tasks: dict[str, asyncio.Task[None]] = {}
        self._started = False
        self._shutting_down = False

    @property
    def version_ids(self) -> logging_pb2.RolloutMetadata.VersionIds:
        if self._version_ids is None:
            raise RuntimeError("daemon is not started")
        return self._version_ids

    def _has_data_source(self, source_id: str) -> bool:
        """Return whether the runtime can materialize the given source ID."""
        if self._data_source_loader is None:
            raise RuntimeError("DataSourceLoader not initialized")
        return self._data_source_loader.has_data_source(source_id)

    def _resolve_scene(self, scene: Scene) -> Scene:
        if self._scene_registry is None:
            raise RuntimeError("SceneRegistry not initialized")
        return self._scene_registry.resolve(scene)

    def _resolve_fork_scene(self, scene: Scene) -> Scene:
        if self._scene_registry is None:
            raise RuntimeError("SceneRegistry not initialized")
        return self._scene_registry.resolve_fork(scene)

    def _ensure_accepting_requests(self) -> None:
        if not self._started:
            raise DaemonUnavailableError("daemon is not started")
        if self._shutting_down:
            raise DaemonUnavailableError("daemon is shutting down")

    async def startup(self) -> None:
        """Initialize the runtime context, start workers, and begin scheduling.

        Builds the RuntimeContext (parses configs, probes service versions,
        validates scenes, and creates the data-source loader and scene registry), then
        creates the worker runtime and daemon scheduler. Idempotent: subsequent calls
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

        self._data_source_loader = runtime_context.data_source_loader
        self._scene_registry = runtime_context.scene_registry
        self._renderer_kind = runtime_context.config.user.renderer.kind

        num_consumers_per_worker = compute_num_consumers_per_worker(
            max_in_flight=runtime_context.max_in_flight,
            nr_workers=runtime_context.config.user.nr_workers,
        )

        worker_runtime: WorkerRuntime | None = None
        try:
            worker_runtime = start_worker_runtime(
                config=runtime_context.config,
                user_config_path=self._user_config_path,
                num_consumers=num_consumers_per_worker,
                log_dir=self._log_dir,
                eval_config=runtime_context.eval_config,
                version_ids=runtime_context.version_ids,
                ego_mask_overrides=runtime_context.ego_mask_overrides,
                nav_routes=runtime_context.nav_routes,
            )

            clip_affine_config = runtime_context.config.user.clip_affine_dispatch
            scheduler = DaemonScheduler(
                pools=runtime_context.pools,
                runtime=worker_runtime,
                clip_affine_dispatch=clip_affine_config,
                max_rollout_retries=runtime_context.config.user.max_rollout_retries,
                rollouts_dir=os.path.join(self._log_dir, "rollouts"),
            )
        except Exception:
            if worker_runtime is not None:
                await worker_runtime.stop()
            raise

        try:
            if clip_affine_config.enabled:
                await scheduler.warm_start()
        except BaseException:
            await scheduler.shutdown(reason="warm_start failed")
            await worker_runtime.stop()
            raise

        self._version_ids = runtime_context.version_ids
        self._runtime_context = runtime_context
        self._worker_runtime = worker_runtime
        self._scheduler = scheduler
        self._started = True

    async def get_runtime_info(self) -> runtime_pb2.RuntimeInfo:
        """Return static runtime discovery information for server clients."""
        self._ensure_accepting_requests()
        assert self._runtime_context is not None

        return build_runtime_info(
            runtime_context=self._runtime_context,
            version_ids=self.version_ids,
        )

    def _forget_clip_prefetch(
        self,
        clip_id: str,
        task: asyncio.Task[None],
    ) -> None:
        """Retrieve task failures and remove the completed clip prefetch."""
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        if self._clip_prefetch_tasks.get(clip_id) is task:
            del self._clip_prefetch_tasks[clip_id]

    async def prefetch_clip(self, clip_id: str) -> None:
        """Warm one clip in the sole NRE renderer used by a rollout cell."""
        self._ensure_accepting_requests()
        if not self._has_data_source(clip_id):
            raise UnknownClipError(clip_id)
        assert self._runtime_context is not None
        if self._runtime_context.config.user.renderer.kind != RendererKind.sensorsim:
            raise ValueError("clip prefetch requires the sensorsim renderer")

        renderer_pool = self._runtime_context.pools["renderer"]
        if renderer_pool.skip or len(renderer_pool.address_order) != 1:
            raise ValueError(
                "clip prefetch requires exactly one renderer address; "
                f"got {renderer_pool.address_order!r}"
            )
        renderer_address = renderer_pool.address_order[0]

        task = self._clip_prefetch_tasks.get(clip_id)
        if task is None:
            logger.info(
                "Clip prefetch request accepted: clip_id=%s renderer=%s",
                clip_id,
                renderer_address,
            )
            task = asyncio.create_task(
                prefetch_nre_clip(renderer_address, clip_id),
                name=f"clip-prefetch-{clip_id}",
            )
            self._clip_prefetch_tasks[clip_id] = task
            task.add_done_callback(
                lambda done, requested_clip_id=clip_id: self._forget_clip_prefetch(
                    requested_clip_id, done
                )
            )
        else:
            logger.info(
                "Clip prefetch request coalesced: clip_id=%s renderer=%s",
                clip_id,
                renderer_address,
            )
        await asyncio.shield(task)

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
            UnknownSceneError: If the request references an unconfigured address.
        """
        self._ensure_accepting_requests()

        request_id = uuid4().hex

        try:
            jobs = build_pending_jobs_from_request(
                request,
                request_id,
                self._resolve_scene,
                self._resolve_fork_scene,
            )
        except UnknownSceneError as exc:
            raise InvalidRequestError(str(exc)) from exc

        assert self._scheduler is not None

        if self._renderer_kind == RendererKind.video_model and any(
            job.ego_history_bytes is not None for job in jobs
        ):
            raise InvalidRequestError(
                "video_model renderer does not support ego_history; use the "
                "stateless sensorsim renderer"
            )

        if any(job.ego_history_bytes is not None for job in jobs):
            assert self._runtime_context is not None
            if not self._runtime_context.pools["trafficsim"].skip:
                raise InvalidRequestError(
                    "ego_history currently requires log-replay traffic "
                    "(runtime.endpoints.trafficsim.skip=true)"
                )

        if self._renderer_kind == RendererKind.video_model and any(
            job.scene.policy_start_us != T0_AUTO for job in jobs
        ):
            raise InvalidRequestError(
                "video_model renderer does not support scenes with an explicit "
                "policy start"
            )

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

        self._shutting_down = True
        errors: list[Exception] = []
        try:
            prefetch_tasks = tuple(self._clip_prefetch_tasks.values())
            for task in prefetch_tasks:
                task.cancel()
            if prefetch_tasks:
                await asyncio.gather(*prefetch_tasks, return_exceptions=True)
            self._clip_prefetch_tasks.clear()

            try:
                await self._scheduler.shutdown(reason="daemon shutting down")
            except Exception as exc:
                logger.exception("Daemon scheduler shutdown failed")
                errors.append(exc)

            try:
                await self._worker_runtime.stop()
            except Exception as exc:
                logger.exception("Daemon worker runtime stop failed")
                errors.append(exc)
        finally:
            self._scheduler = None
            self._worker_runtime = None
            self._runtime_context = None
            self._data_source_loader = None
            self._scene_registry = None
            self._started = False
            self._shutting_down = False

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("daemon shutdown failed", errors)  # noqa: F821
