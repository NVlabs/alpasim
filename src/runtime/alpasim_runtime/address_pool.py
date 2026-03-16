# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Centralized address pool for service slot management.

Runs in the parent process and tracks which service address slots are free vs.
busy. Workers never touch these pools — the parent acquires slots, attaches them
to jobs, and releases them when results arrive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Queue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceAddress:
    """A bookable service slot."""

    address: str
    skip: bool


class AddressPool:
    """
    Tracks available service address slots.

    Each physical address may have N concurrent slots (from n_concurrent_rollouts
    config). The pool hands out individual slots and reclaims them on release.

    Skip pools are non-limiting: they always return a synthetic skip slot on
    acquire and ignore releases.
    """

    def __init__(
        self,
        addresses: list[str],
        n_concurrent: int,
        skip: bool,
    ):
        self.skip = skip
        self._total_capacity: int = 0
        self._queue: Queue[ServiceAddress] = Queue()
        if not skip:
            for addr in addresses:
                for _ in range(n_concurrent):
                    self._queue.put_nowait(ServiceAddress(addr, skip=False))
                    self._total_capacity += 1

    def try_acquire(self) -> ServiceAddress | None:
        """Non-blocking acquire. Returns None if no slots available."""
        if self.skip:
            # Skip pools are non-limiting: no fixed token cap.
            return ServiceAddress("skip", skip=True)
        try:
            return self._queue.get_nowait()
        except QueueEmpty:
            return None

    def release(self, slot: ServiceAddress) -> None:
        """Return a slot to the pool."""
        if self.skip:
            # No-op for synthetic skip slots.
            return
        self._queue.put_nowait(slot)

    @property
    def total_capacity(self) -> int | None:
        """Total number of slots. None for skip pools (non-limiting)."""
        if self.skip:
            return None
        return self._total_capacity


def try_acquire_all(
    pools: dict[str, AddressPool],
) -> dict[str, ServiceAddress] | None:
    """
    Atomically acquire one slot from every pool.

    If any pool has no free slot, releases all already-acquired slots
    and returns None. This guarantees no address leaks on partial failure.
    """
    acquired: dict[str, ServiceAddress] = {}
    for name, pool in pools.items():
        slot = pool.try_acquire()
        if slot is None:
            # Roll back: release everything acquired so far
            for prev_name, prev_slot in acquired.items():
                pools[prev_name].release(prev_slot)
            return None
        acquired[name] = slot
    return acquired


def release_all(
    pools: dict[str, AddressPool],
    acquired: dict[str, ServiceAddress],
) -> None:
    """Release all acquired slots back to their pools."""
    for name, slot in acquired.items():
        pools[name].release(slot)
