# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Worker process entry point and main loop.

Workers are stateless with respect to service management: each job arrives
with pre-assigned service addresses from the parent dispatch loop.  The worker
creates lightweight service objects, runs the rollout, and closes channels.

Supports two execution modes:
- Inline mode (W=1): Runs in parent process (in separate threads)
- Subprocess mode (W>1): Runs in spawned child processes for parallelism
"""

from __future__ import annotations

import asyncio
import functools
import logging
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
from queue import Empty as QueueEmpty

from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.config import RendererKind, UserSimulatorConfig
from alpasim_runtime.event_loop import create_event_rollout
from alpasim_runtime.event_loop_idle_profiler import install_event_loop_idle_profiler
from alpasim_runtime.gc_pressure_profiler import setup_gc_pressure_profiler
from alpasim_runtime.scene_loader import SceneLoader, build_scene_loader
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_runtime.services.video_model_service import VideoModelService
from alpasim_runtime.telemetry.rpc_wrapper import set_shared_rpc_tracking
from alpasim_runtime.telemetry.telemetry_context import (
    TelemetryContext,
    try_get_context,
)
from alpasim_runtime.unbound_rollout import UnboundRollout
from alpasim_runtime.worker.ipc import (
    AssignedRolloutJob,
    JobResult,
    WorkerArgs,
    _ShutdownSentinel,
)
from alpasim_utils.scene_data_source import SceneDataSource
from alpasim_utils.yaml_utils import typed_parse_config

from eval.schema import EvalConfig

_JOB_POLL_TIMEOUT_S = 10.0


def _is_orphaned(parent_pid: int) -> bool:
    """Check if parent process has died (orphan detection)."""
    return os.getppid() != parent_pid


async def run_single_rollout(
    job: AssignedRolloutJob,
    user_config: UserSimulatorConfig,
    data_source: SceneDataSource,
    camera_catalog: CameraCatalog,
    version_ids: RolloutMetadata.VersionIds,
    rollouts_dir: str,
    eval_config: EvalConfig,
    eval_executor: ProcessPoolExecutor,
) -> JobResult:
    """Execute one rollout with the addresses assigned by the parent."""
    ep = job.endpoints
    rollout: UnboundRollout | None = None

    try:
        # Create lightweight service objects (just channel + stub, no pool)
        driver = DriverService(
            ep.driver.address,
            skip=ep.driver.skip,
        )
        physics = PhysicsService(
            ep.physics.address,
            skip=ep.physics.skip,
        )
        traffic = TrafficService(
            ep.trafficsim.address,
            skip=ep.trafficsim.skip,
        )
        controller = ControllerService(
            ep.controller.address,
            skip=ep.controller.skip,
        )

        # Resolve the renderer service from the typed runtime config.
        if user_config.renderer.kind == RendererKind.sensorsim:
            renderer_service = SensorsimService(
                ep.renderer.address,
                skip=ep.renderer.skip,
                camera_catalog=camera_catalog,
            )
        elif user_config.renderer.kind == RendererKind.video_model:
            if user_config.renderer.video_model_config is None:
                raise ValueError(
                    "runtime.renderer.video_model_config is required when "
                    "runtime.renderer.kind=video_model"
                )
            renderer_service = VideoModelService(
                address=ep.renderer.address,
                config=user_config.renderer.video_model_config,
                skip=ep.renderer.skip,
                camera_catalog=camera_catalog,
            )
        else:
            raise ValueError(f"Unknown renderer kind: {user_config.renderer.kind!r}")

        # Offload CPU-bound rollout preparation to thread
        loop = asyncio.get_running_loop()
        rollout = await loop.run_in_executor(
            None,
            functools.partial(
                UnboundRollout.create,
                simulation_config=user_config.simulation_config,
                scene_id=job.scene_id,
                version_ids=version_ids,
                data_source=data_source,
                rollouts_dir=rollouts_dir,
                session_uuid=job.session_uuid,
                renderer_service=renderer_service,
            ),
        )

        eval_result = await create_event_rollout(
            unbound=rollout,
            data_source=data_source,
            driver=driver,
            renderer_service=renderer_service,
            physics=physics,
            trafficsim=traffic,
            controller=controller,
            camera_catalog=camera_catalog,
            eval_config=eval_config,
            eval_executor=eval_executor,
        ).run()

        return JobResult(
            request_id=job.request_id,
            job_id=job.job_id,
            rollout_spec_index=job.rollout_spec_index,
            success=True,
            error=None,
            error_traceback=None,
            rollout_uuid=rollout.rollout_uuid,
            eval_result=eval_result,
        )

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        module_logger = logging.getLogger(__name__)
        module_logger.warning(
            "Rollout FAILED: job=%s scene=%s uuid=%s error=%s\n%s",
            job.job_id,
            rollout.scene_id if rollout else "N/A",
            rollout.rollout_uuid if rollout else "N/A",
            exc,
            tb,
        )
        return JobResult(
            request_id=job.request_id,
            job_id=job.job_id,
            rollout_spec_index=job.rollout_spec_index,
            success=False,
            error=str(exc),
            error_traceback=tb,
            rollout_uuid=rollout.rollout_uuid if rollout else None,
        )


async def run_worker_loop(
    worker_id: int,
    job_queue: Queue,
    result_queue: Queue,
    num_consumers: int,
    user_config: UserSimulatorConfig,
    scene_loader: SceneLoader,
    camera_catalog: CameraCatalog,
    version_ids: RolloutMetadata.VersionIds,
    rollouts_dir: str,
    eval_config: EvalConfig,
    parent_pid: int | None = None,
) -> int:
    """
    Core job processing loop with concurrent consumers.

    Args:
        worker_id: Worker identifier for logging.
        job_queue: Queue to pull AssignedRolloutJob or shutdown sentinel from.
        result_queue: Queue to push JobResult to.
        num_consumers: Number of concurrent consumer tasks.
        user_config: User simulator configuration.
        scene_loader: Worker-local SceneLoader for loading scenes by scene_id.
        camera_catalog: Camera catalog for sensorsim.
        version_ids: Canonical version IDs from the parent process.
        rollouts_dir: Directory for rollout outputs.
        eval_config: Evaluation configuration.
        parent_pid: If None, running inline - skip orphan detection.
                    If set, running in subprocess - exit if parent dies.

    Returns:
        Number of rollouts completed by this worker.
    """
    module_logger = logging.getLogger(__name__)
    module_logger.info(
        "Worker %d ready with num_consumers=%d",
        worker_id,
        num_consumers,
    )

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    rollout_count = 0

    # Install event loop idle profiler
    install_event_loop_idle_profiler(loop)

    # Freeze long-lived objects, install GC profiler, and reset counters so
    # telemetry excludes the startup sweep.
    setup_gc_pressure_profiler()

    # Create a process pool for offloading CPU-bound eval computation.
    # One slot per consumer so no consumer blocks waiting for a pool slot.
    # spawn (not fork) so we don't fork with live gRPC client threads.
    eval_executor = ProcessPoolExecutor(
        max_workers=num_consumers,
        mp_context=mp.get_context("spawn"),
    )

    async def job_consumer() -> None:
        """
        Consume jobs from the shared queue, one at a time.

        Terminates when:
            - A shutdown sentinel is received (re-enqueued for sibling consumers)
            - The parent process dies (orphan detection, subprocess mode only)
        """
        nonlocal rollout_count

        while not shutdown_event.is_set():
            # Orphan detection (subprocess mode only)
            if parent_pid is not None and _is_orphaned(parent_pid):
                module_logger.warning("Parent process died, exiting")
                shutdown_event.set()
                break

            # Pull job with timeout to stay responsive to shutdown signals
            def _poll_job() -> AssignedRolloutJob | _ShutdownSentinel | None:
                try:
                    return job_queue.get(timeout=_JOB_POLL_TIMEOUT_S)
                except QueueEmpty:
                    return None

            job = await loop.run_in_executor(None, _poll_job)

            if job is None:
                # Timeout - retry
                continue

            if isinstance(job, _ShutdownSentinel):
                module_logger.info("Received shutdown signal")
                # Put sentinel back for other consumers/workers
                job_queue.put(job)
                shutdown_event.set()
                break

            # Process the job
            result = await run_single_rollout(
                job=job,
                user_config=user_config,
                data_source=scene_loader.get_data_source(job.scene_id),
                camera_catalog=camera_catalog,
                version_ids=version_ids,
                rollouts_dir=rollouts_dir,
                eval_config=eval_config,
                eval_executor=eval_executor,
            )
            result_queue.put(result)
            rollout_count += 1
            telemetry_ctx = try_get_context()
            if telemetry_ctx is not None:
                telemetry_ctx.record_rollout_complete()

    # Spawn num_consumers consumer tasks -- each handles one job at a time
    try:
        async with asyncio.TaskGroup() as tg:
            for _ in range(num_consumers):
                tg.create_task(job_consumer())
    finally:
        eval_executor.shutdown(wait=True)

    # TaskGroup ensures all consumers complete before exiting
    return rollout_count


def worker_main(args: WorkerArgs) -> None:
    """
    Entrypoint for worker processes to start the asyncio event loop.
    """
    asyncio.run(worker_async_main(args))


async def worker_async_main(args: WorkerArgs) -> None:
    """
    Async worker entry point.

    Handles worker setup (logging to file, metrics) then
    delegates to run_worker_loop for the actual job processing.
    """
    # Initialize shared RPC tracking if provided (multiprocessing mode)
    if args.shared_rpc_tracking is not None:
        set_shared_rpc_tracking(args.shared_rpc_tracking)

    # Load user config (for scenarios, endpoints, etc.)
    user_config = typed_parse_config(args.user_config_path, UserSimulatorConfig)
    scene_loader = build_scene_loader(user_config)

    txt_logs_dir = os.path.join(args.log_dir, "txt-logs")
    rollouts_dir = os.path.join(args.log_dir, "rollouts")
    os.makedirs(txt_logs_dir, exist_ok=True)

    # Configure logging with worker_id in format.
    # Only configure alpasim loggers to avoid breaking third-party library logging.
    log_file = os.path.join(txt_logs_dir, f"runtime_worker_{args.worker_id}.log")
    log_formatter = logging.Formatter(
        f"%(asctime)s.%(msecs)03d [W{args.worker_id}] %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # Configure only alpasim-related loggers, not the root logger
    for logger_name in ["alpasim_runtime", "alpasim_utils", "alpasim_grpc"]:
        pkg_logger = logging.getLogger(logger_name)
        pkg_logger.handlers.clear()
        pkg_logger.setLevel(logging.INFO)
        pkg_logger.addHandler(file_handler)
        pkg_logger.addHandler(console_handler)
        pkg_logger.propagate = False  # Don't propagate to root logger

    module_logger = logging.getLogger(__name__)
    module_logger.info(
        "Worker %d starting (num_workers=%d, num_consumers=%d)",
        args.worker_id,
        args.num_workers,
        args.num_consumers,
    )

    camera_catalog = CameraCatalog(user_config.extra_cameras)

    # TelemetryContext for live Prometheus scraping.
    async with TelemetryContext(
        worker_id=args.worker_id,
        port=args.telemetry_port,
    ):
        await run_worker_loop(
            worker_id=args.worker_id,
            job_queue=args.job_queue,
            result_queue=args.result_queue,
            num_consumers=args.num_consumers,
            user_config=user_config,
            scene_loader=scene_loader,
            camera_catalog=camera_catalog,
            version_ids=args.version_ids,
            rollouts_dir=rollouts_dir,
            eval_config=args.eval_config,
            parent_pid=args.parent_pid,
        )

    module_logger.info("Worker %d exiting", args.worker_id)
