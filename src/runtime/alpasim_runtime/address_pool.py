# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Centralized address pool for service slot management.

Runs in the parent process and tracks which service address slots are free vs.
busy. Workers never touch these pools — the parent acquires slots, attaches them
to jobs, and releases them when results arrive.

The pool is purely a token manager: it hands out ``ServiceAddress`` slots and
reclaims them on release.  Scene-affine routing intelligence (which scenes are
cached where) lives in the scheduler, not here.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

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
        self._all_addresses: frozenset[str] = (
            frozenset(addresses) if not skip else frozenset()
        )
        self._slots: deque[ServiceAddress] = deque()
        if not skip:
            for addr in addresses:
                for _ in range(n_concurrent):
                    self._slots.append(ServiceAddress(addr, skip=False))
                    self._total_capacity += 1

    def try_acquire(self) -> ServiceAddress | None:
        """Non-blocking acquire. Returns None if no slots available."""
        if self.skip:
            return ServiceAddress("skip", skip=True)
        if not self._slots:
            return None
        return self._slots.popleft()

    def release(self, slot: ServiceAddress) -> None:
        """Return a slot to the pool."""
        if self.skip:
            return
        self._slots.append(slot)

    def free_addresses(self) -> set[str]:
        """Return unique addresses that currently have at least one free slot."""
        return {slot.address for slot in self._slots}

    def try_acquire_for_address(self, address: str) -> ServiceAddress | None:
        """Acquire a free slot for a specific *address*, or ``None`` if unavailable."""
        if self.skip:
            return ServiceAddress("skip", skip=True)
        for i, slot in enumerate(self._slots):
            if slot.address == address:
                del self._slots[i]
                return slot
        return None

    def all_addresses(self) -> frozenset[str]:
        """All configured addresses, regardless of current slot availability."""
        return self._all_addresses

    @property
    def total_capacity(self) -> int | None:
        """Total number of slots. None for skip pools (non-limiting)."""
        if self.skip:
            return None
        return self._total_capacity


def try_acquire_all(
    pools: dict[str, AddressPool],
    renderer_slot: ServiceAddress | None = None,
) -> dict[str, ServiceAddress] | None:
    """
    Atomically acquire one slot from every pool.

    When *renderer_slot* is provided, it is used directly for the
    ``renderer`` pool instead of acquiring a new slot.  All other pools
    use regular FIFO.

    If any pool has no free slot, releases all already-acquired slots
    (including a pre-acquired *renderer_slot*) and returns ``None``.
    This guarantees no address leaks on partial failure.
    """
    acquired: dict[str, ServiceAddress] = {}
    if renderer_slot is not None:
        acquired["renderer"] = renderer_slot
    for name, pool in pools.items():
        if name in acquired:
            continue
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
