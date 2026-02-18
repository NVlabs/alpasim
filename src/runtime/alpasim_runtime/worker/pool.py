# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Worker pool management for parallel rollout execution.

Provides the parent-side orchestration for spawning and managing workers:
- Inline mode (W=1): Runs worker loop directly in parent process
- Subprocess mode (W>1): Spawns worker processes for parallel execution

Both modes use the same queue-based job processing pattern and share the
same worker implementation (worker_async_main).

Execution Flow Diagram
======================

The module supports two execution modes based on the number of workers (W):

                       ┌───────────────────────────────────────────┐
                       │  Parent Process: Enqueue all jobs         │
                       │                                           │
                       │              run_workers()                │
                       └─────────────────────┬─────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
                     ▼                                               ▼
       W=1 (Inline Mode)                               W>1 (Subprocess Mode)
       ─────────────────                               ─────────────────────

      ┌─────────────────┐                       ┌───────────────────────────┐
      │ Send SHUTDOWN   │                       │   Spawn W subprocesses    │
      │ sentinel        │                       │   via multiprocessing     │
      │                 │                       │                           │
      │ _run_inline_    │                       │ _run_subprocess_workers() │
      │ worker()        │                       └─────────────┬─────────────┘
      └────────┬────────┘                                     │
               │                          ┌───────────────────┼───────────────────┐
               │                          │                   │                   │
               │                          ▼                   ▼                   ▼
               │                     ┌─────────┐         ┌─────────┐         ┌─────────┐
               │                     │Worker 0 │         │Worker 1 │         │Worker N │
               │                     ├─────────┤         ├─────────┤         ├─────────┤
               │                     │ worker_ │         │ worker_ │         │ worker_ │
               │                     │ main()  │         │ main()  │         │ main()  │
               │                     └────┬────┘         └────┬────┘         └────┬────┘
               │                          │                   │                   │
               ▼                          ▼                   ▼                   ▼
      ╔═════════════════╗            ╔════════════════════════════════════════════════╗
      ║ worker_async_   ║            ║              worker_async_main()               ║
      ║ main()          ║            ║           (in each subprocess)                 ║
      ╚════════╤════════╝            ╚═══════════════════════╤════════════════════════╝
               │                                             │
               ▼                                             ▼
      ╔═════════════════╗            ╔════════════════════════════════════════════════╗
      ║ run_worker_loop ║            ║              run_worker_loop()                 ║
      ║ (in parent)     ║            ║           (in each subprocess)                 ║
      ║                 ║            ║                                                ║
      ║ • No orphan     ║            ║  • Orphan detection enabled                    ║
      ║   detection     ║            ║  • Shared RPC tracking                         ║
      ╚════════╤════════╝            ╚═══════════════════════╤════════════════════════╝
               │                                             │
               │                          ┌──────────────────┴──────────────────┐
               │                          │                                     │
               │                          ▼                                     │
               │             ┌───────────────────────────┐                      │
               │             │ Parent polls result_queue │                      │
               │             │ & checks worker liveness  │                      │
               │             │                           │                      │
               │             │ _run_subprocess_workers() │                      │
               │             └─────────────┬─────────────┘                      │
               │                           │                                    │
               │                           ▼                                    │
               │             ┌───────────────────────────┐                      │
               │             │ Send SHUTDOWN sentinels   │                      │
               │             │ (one per worker)          │◄─────────────────────┘
               │             │                           │
               │             │ _run_subprocess_workers() │
               │             └─────────────┬─────────────┘
               │                           │
               ▼                           ▼
      ┌─────────────────┐         ┌─────────────────────┐
      │ Drain results   │         │ Join workers,       │
      │ from queue,     │         │ return results      │
      │ return results  │         │                     │
      │                 │         │ _run_subprocess_    │
      │ _run_inline_    │         │ workers()           │
      │ worker()        │         └─────────────────────┘
      └─────────────────┘


Shared Components (both modes):
  • job_queue:      multiprocessing.Queue for RolloutJob distribution
  • result_queue:   multiprocessing.Queue for JobResult collection
  • Dispatcher:     Executes rollouts via gRPC service clients
  ════════════════════════════════════════════════════════════════════
  ║ worker_async_main(): Setup logging, metrics, and Dispatcher      ║
  ║ run_worker_loop():   Core job loop with semaphore-based concurrency ║
  ════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from multiprocessing import Process, Queue

from alpasim_runtime.config import SimulatorConfig
from alpasim_runtime.telemetry.rpc_wrapper import init_shared_rpc_tracking
from alpasim_runtime.worker.allocation import compute_worker_allocations
from alpasim_runtime.worker.ipc import (
    SHUTDOWN_SENTINEL,
    JobResult,
    RolloutJob,
    ServiceAllocations,
    WorkerArgs,
    poll_result_queue,
)
from alpasim_runtime.worker.main import worker_async_main, worker_main

from eval.schema import EvalConfig

logger = logging.getLogger(__name__)


async def run_workers(
    config: SimulatorConfig,
    args: argparse.Namespace,
    jobs: list[RolloutJob],
    log_dir: str,
    eval_config: EvalConfig,
) -> list[JobResult]:
    """
    Execute rollouts using a queue-based pattern.

    Both W=1 (inline) and W>1 (subprocess) modes use the same job processing
    pattern. The only difference is:
    - W=1: run_worker_loop executes directly in parent process
    - W>1: worker_main spawns in child processes which spawn separate asyncio event loops

    Args:
        config: Full simulator configuration.
        args: Command-line arguments.
        jobs: List of rollout jobs to execute.
        log_dir: Root directory for all outputs (asl/, metrics/, txt-logs/).

    Returns:
        List of JobResult objects (one per job).
    """
    nr_workers = config.user.nr_workers

    # Create queues (same for both modes)
    job_queue: Queue = Queue()
    result_queue: Queue = Queue()

    # Enqueue all jobs upfront
    for job in jobs:
        job_queue.put(job)
    logger.info("Enqueued all %d jobs", len(jobs))

    # Compute allocations for all workers
    all_allocations = compute_worker_allocations(config, nr_workers)

    # Log allocation summary
    for worker_idx, alloc in enumerate(all_allocations):
        logger.info(
            "Worker %d allocations: capacity=%d", worker_idx, alloc.get_capacity()
        )
        for svc in ["driver", "sensorsim", "physics", "trafficsim", "controller"]:
            svc_alloc = getattr(alloc, svc)
            if svc_alloc:
                logger.debug("  %s: %s", svc, svc_alloc)

    start_time = time.perf_counter()

    if nr_workers == 1:
        # Inline mode: run worker loop directly in parent process
        results = await _run_inline_worker(
            args=args,
            job_queue=job_queue,
            result_queue=result_queue,
            allocations=all_allocations[0],
            log_dir=log_dir,
            eval_config=eval_config,
        )
    else:
        # Subprocess mode: spawn worker processes
        results = await _run_subprocess_workers(
            args=args,
            jobs=jobs,
            job_queue=job_queue,
            result_queue=result_queue,
            all_allocations=all_allocations,
            log_dir=log_dir,
            eval_config=eval_config,
        )

    total_time = time.perf_counter() - start_time
    logger.info(
        "Simulated %d rollouts in %.2f seconds, i.e. %.2f seconds per rollout",
        len(results),
        total_time,
        total_time / len(results) if results else 0,
    )

    return results


async def _run_inline_worker(
    args: argparse.Namespace,
    job_queue: Queue,
    result_queue: Queue,
    allocations: ServiceAllocations,
    log_dir: str,
    eval_config: EvalConfig,
) -> list[JobResult]:
    """
    Execute jobs inline in the parent process (W=1 mode).

    Uses the same worker_async_main as subprocess workers, just called
    directly instead of in a spawned process. This ensures consistent behavior
    between W=1 and W>1 modes.
    """
    logger.info("Running single worker inline")

    worker_args = WorkerArgs(
        worker_id=0,
        num_workers=1,
        job_queue=job_queue,
        result_queue=result_queue,
        allocations=allocations,
        user_config_path=args.user_config,
        usdz_glob=args.usdz_glob,
        log_dir=log_dir,
        parent_pid=None,  # Disable orphan detection for inline mode
        eval_config=eval_config,
    )

    # Send shutdown sentinel - worker will exit after processing all jobs
    job_queue.put(SHUTDOWN_SENTINEL)

    await worker_async_main(worker_args)

    # Collect results from queue
    results: list[JobResult] = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    return results


async def _run_subprocess_workers(
    args: argparse.Namespace,
    jobs: list[RolloutJob],
    job_queue: Queue,
    result_queue: Queue,
    all_allocations: list[ServiceAllocations],
    log_dir: str,
    eval_config: EvalConfig,
) -> list[JobResult]:
    """
    Execute jobs in spawned worker processes (W>1 mode).

    Spawns multiple worker processes that pull jobs from a shared queue and
    execute them in parallel. Each worker runs its own asyncio event loop
    and manages its own service instances based on pre-computed allocations.

    Args:
        args: Command-line arguments containing user_config and usdz_glob paths.
        jobs: List of rollout jobs to execute (already enqueued in job_queue).
        job_queue: Shared queue containing jobs for workers to consume.
        result_queue: Shared queue where workers post completed JobResults.
        all_allocations: Per-worker service allocations (GPU/CPU assignments).
        log_dir: Root directory for all outputs (asl/, metrics/, txt-logs/).

    Returns:
        List of JobResult objects (one per job, in completion order).

    Raises:
        RuntimeError: If any worker process crashes before all jobs complete.
            Logs orphaned job IDs for debugging.

    Notes:
        - Workers receive parent PID for orphan detection (auto-exit if parent dies).
        - Shared RPC tracking enables global queue depth metrics across processes.
        - Graceful shutdown sends SHUTDOWN_SENTINEL to each worker and waits up to
          30s for exit before forcefully terminating.
    """
    nr_workers = len(all_allocations)
    logger.info("Running in multi-worker mode with %d workers", nr_workers)

    # Get parent PID for orphan detection in workers
    parent_pid = os.getpid()

    # Initialize shared RPC tracking for global queue depth metrics across processes
    shared_rpc_tracking = init_shared_rpc_tracking()

    # Spawn worker processes
    workers: list[Process] = []
    for worker_id in range(nr_workers):
        worker_args = WorkerArgs(
            worker_id=worker_id,
            num_workers=nr_workers,
            job_queue=job_queue,
            result_queue=result_queue,
            allocations=all_allocations[worker_id],
            user_config_path=args.user_config,
            usdz_glob=args.usdz_glob,
            parent_pid=parent_pid,
            log_dir=log_dir,
            shared_rpc_tracking=shared_rpc_tracking,
            eval_config=eval_config,
        )
        p = Process(
            target=worker_main,
            args=(worker_args,),
        )
        p.start()
        workers.append(p)

    # Collect results asynchronously with timeout and liveness checks
    loop = asyncio.get_running_loop()
    results: list[JobResult] = []
    try:
        while len(results) < len(jobs):
            result = await loop.run_in_executor(None, poll_result_queue, result_queue)

            if result is not None:
                results.append(result)
                if len(results) % 10 == 0 or len(results) == len(jobs):
                    logger.info("Completed %d/%d jobs", len(results), len(jobs))
                continue

            # Timeout - check if any workers crashed
            dead_workers = [
                (idx, p)
                for idx, p in enumerate(workers)
                if not p.is_alive() and p.exitcode != 0
            ]
            if dead_workers:
                for idx, proc in dead_workers:
                    logger.error("Worker %d died with exit code %s", idx, proc.exitcode)
                raise RuntimeError(
                    f"{len(dead_workers)} worker(s) crashed. "
                    f"Received {len(results)}/{len(jobs)} results before failure."
                )
            logger.debug("Waiting for results (workers still alive)...")

    # Log orphaned jobs on worker failure
    except RuntimeError as e:
        completed_ids = {r.job_id for r in results}
        orphaned_ids = [j.job_id for j in jobs if j.job_id not in completed_ids]
        if orphaned_ids:
            logger.error(
                "%d jobs were orphaned due to worker failure: %s%s",
                len(orphaned_ids),
                orphaned_ids[:5],
                "..." if len(orphaned_ids) > 5 else "",
            )
        raise RuntimeError(f"Worker pool failed: {e}") from e

    # Always attempt graceful shutdown of workers
    finally:
        # Send shutdown sentinels (one per worker)
        for _ in workers:
            job_queue.put(SHUTDOWN_SENTINEL)

        # Wait for workers to exit
        for worker_idx, proc in enumerate(workers):
            proc.join(timeout=30)
            if proc.is_alive():
                logger.warning(
                    "Worker %d did not exit gracefully, terminating", worker_idx
                )
                proc.terminate()
                proc.join(timeout=5)
            elif proc.exitcode != 0:
                logger.error("Worker %d exited with code %s", worker_idx, proc.exitcode)

    return results
