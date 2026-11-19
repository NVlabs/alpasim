# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Construction of the evaluation process pool.

The pool's internal queue locks are named POSIX semaphores in ``/dev/shm``, and
container runtimes commonly bind-mount the host's ``/dev/shm``, so the names
live in a node-shared namespace. ``ProcessPoolExecutor`` starts its workers on
demand, at submission time — which can be an hour after pool creation when
workspace setup is slow. Anything that removes the semaphore files in that
window (host-level cleanup has been observed doing so on shared clusters) makes
the next worker die in ``SemLock._rebuild`` with ``FileNotFoundError``, breaking
the pool and with it every evaluation of the runtime.

Starting the workers at creation closes the window: a process that has opened a
semaphore keeps it alive through its file descriptor, so unlinking the name
afterwards is harmless. Every worker has to start, not most of them — one late
worker breaks the whole pool — and the executor skips a start whenever some
worker is already idle. So the warmup tasks are held in their initializer until
the last worker is up, and only then released.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Backstop for workers left waiting if the parent dies mid-warmup. Only an
# upper bound on worker startup; the gate normally opens in milliseconds.
_GATE_TIMEOUT_S = 300.0


def _warmup() -> None:
    """No-op task whose only purpose is to make the executor start a worker."""


def create_eval_executor(max_workers: int) -> ProcessPoolExecutor:
    """Create the eval pool with every worker started before returning.

    spawn (not fork) so we don't fork with live gRPC client threads. One warmup
    task per slot is submitted because the executor starts workers on demand,
    and each worker blocks on *gate* in its initializer so that none can report
    itself idle — and cost a later slot its worker — before the last one starts.
    """
    ctx = mp.get_context("spawn")
    gate = ctx.Event()
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=gate.wait,
        initargs=(_GATE_TIMEOUT_S,),
    )
    try:
        futures = [executor.submit(_warmup) for _ in range(max_workers)]
        gate.set()
        for future in futures:
            future.result()
    except BaseException:
        # Open the gate first: a worker still blocked in its initializer would
        # hold shutdown(wait=True) until the backstop timeout expires.
        gate.set()
        executor.shutdown(wait=True)
        raise
    return executor
