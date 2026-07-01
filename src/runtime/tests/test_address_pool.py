# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Unit tests for AddressPool and related functions.
"""

import pytest
from alpasim_runtime.address_pool import (
    AddressPool,
    ServiceAddress,
    release_all,
    try_acquire_all,
)


class TestServiceAddress:
    """Tests for ServiceAddress dataclass."""

    def test_creation(self):
        addr = ServiceAddress("10.0.0.1:50051", skip=False)
        assert addr.address == "10.0.0.1:50051"
        assert addr.skip is False

    def test_skip_address(self):
        addr = ServiceAddress("skip", skip=True)
        assert addr.address == "skip"
        assert addr.skip is True

    def test_frozen(self):
        """ServiceAddress should be immutable (frozen dataclass)."""
        addr = ServiceAddress("10.0.0.1:50051", skip=False)
        with pytest.raises(AttributeError):
            addr.address = "other"  # type: ignore[misc]

    def test_equality(self):
        a = ServiceAddress("addr1", skip=False)
        b = ServiceAddress("addr1", skip=False)
        assert a == b

    def test_hashable(self):
        """Frozen dataclass should be hashable."""
        addr = ServiceAddress("addr1", skip=False)
        assert hash(addr) is not None
        # Can be used in sets
        s = {addr, ServiceAddress("addr2", skip=False)}
        assert len(s) == 2


class TestAddressPool:
    """Tests for AddressPool."""

    def test_basic_acquire_release(self):
        """Basic acquire and release cycle."""
        pool = AddressPool(["A"], n_concurrent=2, skip=False)

        slot1 = pool.try_acquire()
        slot2 = pool.try_acquire()
        assert slot1 is not None
        assert slot2 is not None
        assert slot1.address == "A"
        assert slot2.address == "A"

        # Pool exhausted
        assert pool.try_acquire() is None

        # Release one, can acquire again
        pool.release(slot1)
        slot3 = pool.try_acquire()
        assert slot3 is not None
        assert slot3.address == "A"

    def test_multiple_addresses(self):
        """Multiple addresses with concurrent slots."""
        pool = AddressPool(["A", "B"], n_concurrent=2, skip=False)

        slots = []
        for _ in range(4):  # 2 addresses * 2 concurrent = 4 total
            slot = pool.try_acquire()
            assert slot is not None
            slots.append(slot)

        # Pool exhausted
        assert pool.try_acquire() is None

        # Addresses should include both A and B
        addresses = {s.address for s in slots}
        assert addresses == {"A", "B"}

    def test_total_capacity(self):
        pool = AddressPool(["A", "B", "C"], n_concurrent=3, skip=False)
        assert pool.total_capacity == 9  # 3 * 3

    def test_total_capacity_skip(self):
        pool = AddressPool(["A"], n_concurrent=2, skip=True)
        assert pool.total_capacity is None

    def test_skip_pool_always_acquires(self):
        """Skip pools should always return a slot."""
        pool = AddressPool([], n_concurrent=0, skip=True)

        # Can acquire indefinitely
        for _ in range(100):
            slot = pool.try_acquire()
            assert slot is not None
            assert slot.skip is True
            assert slot.address == "skip"

    def test_skip_pool_release_is_noop(self):
        """Releasing to a skip pool should be a no-op."""
        pool = AddressPool([], n_concurrent=0, skip=True)
        slot = pool.try_acquire()
        assert slot is not None
        # Should not raise
        pool.release(slot)

    def test_empty_addresses(self):
        """Empty address list with skip=False should have zero capacity."""
        pool = AddressPool([], n_concurrent=4, skip=False)
        assert pool.total_capacity == 0
        assert pool.try_acquire() is None

    def test_zero_concurrent(self):
        """Zero concurrent should have zero capacity."""
        pool = AddressPool(["A", "B"], n_concurrent=0, skip=False)
        assert pool.total_capacity == 0
        assert pool.try_acquire() is None

    def test_preserves_total_slots(self):
        """Total acquired + released should preserve capacity."""
        pool = AddressPool(["A", "B"], n_concurrent=3, skip=False)
        total = pool.total_capacity
        assert total == 6

        # Acquire all
        slots = []
        for _ in range(total):
            slots.append(pool.try_acquire())
        assert pool.try_acquire() is None

        # Release all
        for s in slots:
            pool.release(s)

        # Can acquire all again
        for _ in range(total):
            assert pool.try_acquire() is not None
        assert pool.try_acquire() is None

    def test_free_addresses(self):
        """free_addresses should return addresses with available slots."""
        pool = AddressPool(["A", "B"], n_concurrent=1, skip=False)
        assert pool.free_addresses() == {"A", "B"}

        slot = pool.try_acquire()
        remaining = pool.free_addresses()
        # One address consumed
        assert len(remaining) == 1

        pool.release(slot)
        assert pool.free_addresses() == {"A", "B"}

    def test_try_acquire_for_address(self):
        """Should acquire a specific address's slot."""
        pool = AddressPool(["A", "B"], n_concurrent=1, skip=False)

        slot = pool.try_acquire_for_address("B")
        assert slot is not None
        assert slot.address == "B"

        # B is now exhausted
        assert pool.try_acquire_for_address("B") is None
        # A still available
        assert pool.try_acquire_for_address("A") is not None

    def test_try_acquire_for_address_skip(self):
        """Skip pool should return skip slot for any address."""
        pool = AddressPool([], n_concurrent=0, skip=True)
        slot = pool.try_acquire_for_address("anything")
        assert slot is not None
        assert slot.skip is True

    def test_all_addresses(self):
        pool = AddressPool(["A", "B", "C"], n_concurrent=2, skip=False)
        assert pool.all_addresses() == frozenset({"A", "B", "C"})

    def test_all_addresses_skip(self):
        pool = AddressPool(["A"], n_concurrent=2, skip=True)
        assert pool.all_addresses() == frozenset()


class TestTryAcquireAll:
    """Tests for try_acquire_all function."""

    def test_success(self):
        """Should acquire one slot from each pool."""
        pools = {
            "driver": AddressPool(["A"], n_concurrent=2, skip=False),
            "physics": AddressPool(["B"], n_concurrent=2, skip=False),
        }
        result = try_acquire_all(pools)
        assert result is not None
        assert "driver" in result
        assert "physics" in result
        assert result["driver"].address == "A"
        assert result["physics"].address == "B"

    def test_partial_failure_rollback(self):
        """If any pool has no capacity, all acquired slots are released."""
        pools = {
            "driver": AddressPool(["A"], n_concurrent=1, skip=False),
            "physics": AddressPool(["B"], n_concurrent=1, skip=False),
        }

        # Exhaust physics pool
        physics_slot = pools["physics"].try_acquire()
        assert physics_slot is not None

        # Now try_acquire_all should fail and roll back driver
        result = try_acquire_all(pools)
        assert result is None

        # Driver slot should have been released (rollback)
        driver_slot = pools["driver"].try_acquire()
        assert driver_slot is not None  # still available

    def test_with_skip_pools(self):
        """Skip pools should not block acquisition."""
        pools = {
            "driver": AddressPool(["A"], n_concurrent=1, skip=False),
            "sensorsim": AddressPool([], n_concurrent=0, skip=True),
        }
        result = try_acquire_all(pools)
        assert result is not None
        assert result["driver"].skip is False
        assert result["sensorsim"].skip is True

    def test_all_skip(self):
        """All skip pools should always succeed."""
        pools = {
            "driver": AddressPool([], n_concurrent=0, skip=True),
            "physics": AddressPool([], n_concurrent=0, skip=True),
        }
        result = try_acquire_all(pools)
        assert result is not None

    def test_empty_pools(self):
        """No pools should return empty dict."""
        result = try_acquire_all({})
        assert result == {}

    def test_pre_acquired_renderer_slot(self):
        """try_acquire_all with renderer_slot skips renderer pool."""
        pools = {
            "driver": AddressPool(["D"], n_concurrent=2, skip=False),
            "renderer": AddressPool(["GPU-0", "GPU-1"], n_concurrent=1, skip=False),
        }

        # Pre-acquire a renderer slot (as the scheduler would)
        r_slot = pools["renderer"].try_acquire()
        assert r_slot is not None

        result = try_acquire_all(pools, renderer_slot=r_slot)
        assert result is not None
        assert result["renderer"] is r_slot
        assert result["driver"].address == "D"

    def test_pre_acquired_slot_rolled_back_on_failure(self):
        """If another pool fails, the pre-acquired renderer slot is released."""
        pools = {
            "driver": AddressPool(["D"], n_concurrent=1, skip=False),
            "renderer": AddressPool(["GPU-0"], n_concurrent=1, skip=False),
        }

        # Exhaust driver
        pools["driver"].try_acquire()

        # Pre-acquire renderer
        r_slot = pools["renderer"].try_acquire()
        assert r_slot is not None

        result = try_acquire_all(pools, renderer_slot=r_slot)
        assert result is None

        # renderer slot should have been released back by rollback
        recovered = pools["renderer"].try_acquire()
        assert recovered is not None
        assert recovered.address == "GPU-0"


class TestReleaseAll:
    """Tests for release_all function."""

    def test_releases_slots(self):
        """Should release all acquired slots back to pools."""
        pools = {
            "driver": AddressPool(["A"], n_concurrent=1, skip=False),
            "physics": AddressPool(["B"], n_concurrent=1, skip=False),
        }

        acquired = try_acquire_all(pools)
        assert acquired is not None

        # Both exhausted
        assert pools["driver"].try_acquire() is None
        assert pools["physics"].try_acquire() is None

        release_all(pools, acquired)

        # Both available again
        assert pools["driver"].try_acquire() is not None
        assert pools["physics"].try_acquire() is not None

    def test_releases_skip_slots(self):
        """Releasing skip slots should be a no-op (not raise)."""
        pools = {
            "svc": AddressPool([], n_concurrent=0, skip=True),
        }
        acquired = try_acquire_all(pools)
        assert acquired is not None
        # Should not raise
        release_all(pools, acquired)
