# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Worker process entry point and main loop.

Supports two execution modes:
- Inline mode (W=1): Runs in parent process for debugging
- Subprocess mode (W>1): Runs in spawned child processes for parallelism
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from multiprocessing import Queue

from alpasim_runtime.config import UserSimulatorConfig, typed_parse_config
from alpasim_runtime.dispatcher import Dispatcher
from alpasim_runtime.event_loop_idle_profiler import install_event_loop_idle_profiler
from alpasim_runtime.telemetry.rpc_wrapper import set_shared_rpc_tracking
from alpasim_runtime.telemetry.telemetry_context import TelemetryContext
from alpasim_runtime.worker.ipc import WorkerArgs, _ShutdownSentinel, poll_job_queue


def _is_orphaned(parent_pid: int) -> bool:
    """Check if parent process has died (orphan detection)."""
    return os.getppid() != parent_pid


async def run_worker_loop(
    worker_id: int,
    job_queue: Queue,
    result_queue: Queue,
    dispatcher: Dispatcher,
    parent_pid: int | None = None,
) -> int:
    """
    Core job processing loop. Used by both inline (W=1) and subprocess (W>1) modes.

    Consumes jobs from job_queue, executes them via dispatcher, and pushes results
    to result_queue.

    Args:
        worker_id: Worker identifier for logging.
        job_queue: Queue to pull RolloutJob or shutdown sentinel from.
        result_queue: Queue to push JobResult to.
        dispatcher: Dispatcher instance for executing jobs.
        parent_pid: If None, running inline in parent - skip orphan detection.
                    If set, running in subprocess - exit if parent dies.

    Returns:
        Number of rollouts completed by this worker.
    """
    module_logger = logging.getLogger(__name__)

    max_concurrent = dispatcher.get_pool_capacity()
    module_logger.info(f"Worker {worker_id} ready with capacity={max_concurrent}")

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    rollout_count = 0

    # Install event loop idle profiler
    install_event_loop_idle_profiler(loop)

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

            # Pull job. With timeout to stay responsive to shutdown signals.
            job = await loop.run_in_executor(
                None,
                poll_job_queue,
                job_queue,
            )

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
            result = await dispatcher.run_job(job)
            result_queue.put(result)
            rollout_count += 1

    # Spawn max_concurrent consumer tasks - each handles one job at a time
    async with asyncio.TaskGroup() as tg:
        for _ in range(max_concurrent):
            tg.create_task(job_consumer())

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

    Handles worker setup (logging to file, metrics, dispatcher creation) then
    delegates to run_worker_loop for the actual job processing.
    """
    # Initialize shared RPC tracking if provided (multiprocessing mode)
    if args.shared_rpc_tracking is not None:
        set_shared_rpc_tracking(args.shared_rpc_tracking)

    # Load user config (for scenarios, endpoints, etc.)
    # Network config is not needed - allocations are pre-computed
    user_config = typed_parse_config(args.user_config_path, UserSimulatorConfig)

    txt_logs_dir = os.path.join(args.log_dir, "txt-logs")
    rollouts_dir = os.path.join(args.log_dir, "rollouts")
    telemetry_dir = os.path.join(args.log_dir, "telemetry")
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
        f"Worker {args.worker_id} starting (num_workers={args.num_workers})"
    )
    module_logger.info(f"Allocations: {args.allocations}")

    start_time = time.perf_counter()

    # TelemetryContext for telemetry collection.
    # Worker 0 samples resources (CPU/GPU); other workers only collect RPC/rollout/step timing.
    async with TelemetryContext(
        output_dir=telemetry_dir,
        worker_id=args.worker_id,
        sample_resources=(args.worker_id == 0),
    ) as ctx:
        dispatcher = await Dispatcher.create(
            user_config=user_config,
            allocations=args.allocations,
            usdz_glob=args.usdz_glob,
            rollouts_dir=rollouts_dir,
            eval_config=args.eval_config,
        )

        rollout_count = await run_worker_loop(
            worker_id=args.worker_id,
            job_queue=args.job_queue,
            result_queue=args.result_queue,
            dispatcher=dispatcher,
            parent_pid=args.parent_pid,  # Enable orphan detection
        )

        # Record simulation summary with actual measured values
        total_time = time.perf_counter() - start_time
        ctx.record_simulation_summary(total_time, rollout_count)

    module_logger.info(f"Worker {args.worker_id} exiting")
