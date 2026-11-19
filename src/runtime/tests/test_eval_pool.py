# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Regression coverage for eval_pool.create_eval_executor.

Locks in the failure seen on AlpaGym jobs 6657760 and 6664486: the eval pool's
named semaphores were deleted from the node-shared ``/dev/shm`` during the long
gap between pool creation and the first evaluation, so the worker that started
then died in ``SemLock._rebuild`` and every evaluation failed with
``BrokenProcessPool``. Starting every worker at creation must make the pool
immune to that — every worker, because one late starter breaks the whole pool.
"""

from __future__ import annotations

import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from alpasim_runtime.worker.eval_pool import create_eval_executor

# The hard-coded shared-memory path is the point: these tests reproduce a
# failure that only happens because the pool's semaphores live in a node-shared
# namespace. Hence the S108 suppression rather than a temp directory.
SEM_GLOB = "/dev/shm/sem.mp-*"  # noqa: S108

# Deleting the names behind multiprocessing's back leaves its own finalizer
# unlinking something that is already gone. That noise is what these tests
# simulate, so it is not a finding.
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning"
)


def _unlink_new_semaphores(before: set[str]) -> None:
    """Delete the semaphore names that appeared since *before* was sampled.

    Stands in for the node-side cleanup that removed them in production. Only
    names this test's pool created are touched, and a name that the real
    cleaner got to first is not an error.
    """
    for path in set(glob.glob(SEM_GLOB)) - before:
        Path(path).unlink(missing_ok=True)


def _pid_after_pause() -> int:
    """Hold a worker slot long enough that one worker cannot serve the batch."""
    time.sleep(0.5)
    return os.getpid()


def test_eval_pool_survives_semaphore_unlink() -> None:
    """The pool keeps working after its /dev/shm semaphore names vanish."""
    before = set(glob.glob(SEM_GLOB))
    executor = create_eval_executor(max_workers=2)
    try:
        _unlink_new_semaphores(before)

        # Without the eager start this raises BrokenProcessPool: the first
        # worker starts here and dies in SemLock._rebuild (FileNotFoundError).
        assert executor.submit(int, 7).result(timeout=60) == 7
    finally:
        executor.shutdown(wait=True)


def test_every_slot_gets_a_worker_when_warmup_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A descheduled warmup still leaves one live worker per slot.

    The executor skips starting a worker whenever another is already idle, so
    spreading the warmup submissions out can leave most slots empty. Those
    slots then start their worker later, after the semaphores are gone — the
    original failure, reached by a slower road.
    """
    unpatched_submit = ProcessPoolExecutor.submit

    def stalling_submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Submit, then stall the caller long enough for a worker to go idle."""
        future = unpatched_submit(self, fn, *args, **kwargs)
        time.sleep(0.2)
        return future

    monkeypatch.setattr(ProcessPoolExecutor, "submit", stalling_submit)
    before = set(glob.glob(SEM_GLOB))
    executor = create_eval_executor(max_workers=3)
    try:
        _unlink_new_semaphores(before)

        # Three concurrent slow tasks need three workers. Any slot whose worker
        # had not started yet has to start one now, and would die doing so.
        futures = [executor.submit(_pid_after_pause) for _ in range(3)]
        assert len({future.result(timeout=60) for future in futures}) == 3
    finally:
        executor.shutdown(wait=True)
