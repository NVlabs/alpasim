# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Coroutine
from multiprocessing import Process, Queue
from queue import Empty as QueueEmpty
from typing import Any

from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.config import SimulatorConfig
from alpasim_runtime.telemetry.rpc_wrapper import init_shared_rpc_tracking
from alpasim_runtime.worker.ipc import (
    SHUTDOWN_SENTINEL,
    AssignedRolloutJob,
    JobResult,
    WorkerArgs,
)
from alpasim_runtime.worker.main import worker_async_main, worker_main

from eval.schema import EvalConfig

logger = logging.getLogger(__name__)

_RESULT_POLL_TIMEOUT_S = 30.0


class WorkerRuntime:
    """Manages worker lifecycle for both inline (W=1) and subprocess (W>1) modes."""

    def __init__(
        self,
        *,
        job_queue: Queue,
        result_queue: Queue,
        worker_args: list[WorkerArgs],
        inline_main: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._job_queue = job_queue
        self._result_queue = result_queue
        self._inline_main = inline_main
        self._workers: list[Process] = []
        self._inline_task: asyncio.Task[None] | None = None

        if inline_main is not None:
            self._inline_task = asyncio.create_task(inline_main())
        else:
            for args in worker_args:
                proc = Process(target=worker_main, args=(args,))
                proc.start()
                self._workers.append(proc)

    def submit_assigned_job(self, job: AssignedRolloutJob) -> None:
        self._job_queue.put(job)

    def _poll_result_sync(self) -> JobResult | None:
        try:
            return self._result_queue.get(timeout=_RESULT_POLL_TIMEOUT_S)
        except QueueEmpty:
            return None

    async def poll_result(self) -> JobResult | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._poll_result_sync)

    async def stop(self) -> None:
        """Gracefully shut down all workers by sending shutdown sentinels."""
        if self._inline_main is not None:
            await self._stop_inline()
        else:
            await self._stop_subprocess()

    async def _stop_inline(self) -> None:
        if self._inline_task is None:
            return
        self._job_queue.put(SHUTDOWN_SENTINEL)
        await self._inline_task
        self._inline_task = None

    async def _stop_subprocess(self) -> None:
        if not self._workers:
            return

        for _ in self._workers:
            self._job_queue.put(SHUTDOWN_SENTINEL)

        for worker_idx, proc in enumerate(self._workers):
            proc.join(timeout=30)
            if proc.is_alive():
                logger.warning(
                    "Worker %d did not exit gracefully, terminating", worker_idx
                )
                proc.terminate()
                proc.join(timeout=5)
            elif proc.exitcode != 0:
                logger.error("Worker %d exited with code %s", worker_idx, proc.exitcode)
        self._workers.clear()

    def check_for_crashes(self) -> None:
        """Raise RuntimeError if a worker has died unexpectedly.

        Checks both subprocess workers (W>1) and the inline async task (W=1).
        """
        if self._inline_task is not None and self._inline_task.done():
            if self._inline_task.cancelled():
                raise RuntimeError("Inline worker task was cancelled unexpectedly.")
            exc = self._inline_task.exception()
            if exc is not None:
                raise RuntimeError("Inline worker crashed while dispatching.") from exc

        dead_workers = [
            (idx, proc)
            for idx, proc in enumerate(self._workers)
            if not proc.is_alive() and proc.exitcode != 0
        ]
        if not dead_workers:
            return

        for worker_idx, proc in dead_workers:
            logger.error("Worker %d died with exit code %s", worker_idx, proc.exitcode)
        raise RuntimeError(f"{len(dead_workers)} worker(s) crashed while dispatching.")


def start_worker_runtime(
    config: SimulatorConfig,
    *,
    user_config_path: str,
    num_consumers: int,
    log_dir: str,
    eval_config: EvalConfig,
    version_ids: RolloutMetadata.VersionIds,
) -> WorkerRuntime:
    """Create and start a WorkerRuntime for the given configuration.

    For ``nr_workers=1``, runs inline in the current process as an async task.
    For ``nr_workers>1``, spawns subprocess workers with shared RPC tracking.
    """
    nr_workers = config.user.nr_workers
    job_queue: Queue = Queue()
    result_queue: Queue = Queue()

    if nr_workers == 1:
        args = WorkerArgs(
            worker_id=0,
            num_workers=1,
            job_queue=job_queue,
            result_queue=result_queue,
            num_consumers=num_consumers,
            user_config_path=user_config_path,
            log_dir=log_dir,
            eval_config=eval_config,
            version_ids=version_ids,
            parent_pid=None,
        )
        runtime = WorkerRuntime(
            job_queue=job_queue,
            result_queue=result_queue,
            worker_args=[args],
            inline_main=lambda: worker_async_main(args),
        )
    else:
        parent_pid = os.getpid()
        shared_rpc_tracking = init_shared_rpc_tracking()
        all_args = [
            WorkerArgs(
                worker_id=worker_id,
                num_workers=nr_workers,
                job_queue=job_queue,
                result_queue=result_queue,
                num_consumers=num_consumers,
                user_config_path=user_config_path,
                log_dir=log_dir,
                eval_config=eval_config,
                version_ids=version_ids,
                parent_pid=parent_pid,
                shared_rpc_tracking=shared_rpc_tracking,
            )
            for worker_id in range(nr_workers)
        ]
        runtime = WorkerRuntime(
            job_queue=job_queue,
            result_queue=result_queue,
            worker_args=all_args,
        )

    return runtime
