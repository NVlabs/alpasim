# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Tests for the event loop idle profiler.

Verifies that idle time, poll time, and the idle fraction are correctly measured.
"""

import asyncio
from time import perf_counter

import pytest
from alpasim_runtime.event_loop_idle_profiler import (
    get_event_loop_idle_stats,
    install_event_loop_idle_profiler,
    reset_event_loop_idle_stats,
)


@pytest.fixture(autouse=True)
def reset_profiler_state():
    """Reset profiler state before each test."""
    reset_event_loop_idle_stats()
    yield
    reset_event_loop_idle_stats()


class TestIdleTimeMeasurement:
    """Tests for idle time measurement (blocking waits)."""

    async def test_sleep_counts_as_idle_time(self):
        """Sleeping should accumulate idle time."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        sleep_duration = 0.1
        await asyncio.sleep(sleep_duration)

        stats = get_event_loop_idle_stats()

        # Allow 20% tolerance for timing variations
        assert stats["idle_seconds"] > sleep_duration * 0.8
        assert stats["idle_seconds"] < sleep_duration * 1.2

    async def test_multiple_sleeps_accumulate(self):
        """Multiple sleeps should accumulate idle time."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        num_sleeps = 5
        sleep_duration = 0.05

        for _ in range(num_sleeps):
            await asyncio.sleep(sleep_duration)

        stats = get_event_loop_idle_stats()
        expected_total = num_sleeps * sleep_duration

        assert stats["idle_seconds"] > expected_total * 0.8
        assert stats["idle_seconds"] < expected_total * 1.2

    async def test_waiting_on_event_counts_as_idle(self):
        """Waiting on an asyncio.Event should count as idle time."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        event = asyncio.Event()
        wait_duration = 0.1

        async def set_after_delay():
            await asyncio.sleep(wait_duration)
            event.set()

        asyncio.create_task(set_after_delay())
        await event.wait()

        stats = get_event_loop_idle_stats()

        # Should have accumulated idle time while waiting
        assert stats["idle_seconds"] > wait_duration * 0.8


class TestPollTimeMeasurement:
    """Tests for poll time measurement (non-blocking checks)."""

    async def test_cpu_bound_work_generates_polls(self):
        """CPU-bound work with periodic yields should generate poll time."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        # Do some work with zero-delay yields (non-blocking polls)
        iterations = 100
        for _ in range(iterations):
            # asyncio.sleep(0) is a non-blocking yield
            await asyncio.sleep(0)

        stats = get_event_loop_idle_stats()

        # Should have poll time from zero-timeout selects
        # Poll time should be very small but non-zero
        assert stats["poll_seconds"] >= 0
        # Select calls should be at least the number of yields
        assert stats["select_calls"] >= iterations


class TestSelectCallCounting:
    """Tests for select call counting."""

    async def test_select_calls_counted(self):
        """Each sleep should result in select calls."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        initial_stats = get_event_loop_idle_stats()
        initial_calls = initial_stats["select_calls"]

        await asyncio.sleep(0.05)

        stats = get_event_loop_idle_stats()
        # Should have at least one new select call
        assert stats["select_calls"] > initial_calls


class TestIdleFraction:
    """Tests for computing idle fraction from the stats."""

    async def test_mostly_idle_workload(self):
        """A workload that mostly sleeps should have high idle fraction."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        wall_start = perf_counter()

        # Mostly sleeping
        for _ in range(5):
            await asyncio.sleep(0.05)

        wall_elapsed = perf_counter() - wall_start
        stats = get_event_loop_idle_stats()

        # Compute idle fraction
        total_measured = stats["idle_seconds"] + stats["poll_seconds"]
        idle_fraction = (
            stats["idle_seconds"] / total_measured if total_measured > 0 else 0
        )

        # For a mostly-sleeping workload, idle fraction should be very high (> 99%)
        assert (
            idle_fraction > 0.99
        ), f"Expected high idle fraction, got {idle_fraction:.2%}"

        # Idle time should be close to wall time for this simple workload
        assert stats["idle_seconds"] > wall_elapsed * 0.9

    async def test_mixed_workload_idle_fraction(self):
        """A mixed workload should have measurable idle and poll times."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        # Mix of sleeping and yielding
        for _ in range(10):
            await asyncio.sleep(0.01)  # Idle time
            for _ in range(10):
                await asyncio.sleep(0)  # Poll time (zero-timeout)

        stats = get_event_loop_idle_stats()

        # Should have both idle and poll time
        assert stats["idle_seconds"] > 0
        assert stats["poll_seconds"] >= 0

        # Compute fractions
        total_measured = stats["idle_seconds"] + stats["poll_seconds"]
        idle_fraction = (
            stats["idle_seconds"] / total_measured if total_measured > 0 else 0
        )

        # Idle should dominate since sleeps are 10ms each
        assert (
            idle_fraction > 0.9
        ), f"Expected idle to dominate, got {idle_fraction:.2%}"


class TestResetFunctionality:
    """Tests for the reset function."""

    async def test_reset_clears_all_stats(self):
        """Reset should clear all accumulated statistics."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        # Accumulate some stats
        await asyncio.sleep(0.05)

        stats_before = get_event_loop_idle_stats()
        assert stats_before["idle_seconds"] > 0
        assert stats_before["select_calls"] > 0

        # Reset
        reset_event_loop_idle_stats()

        stats_after = get_event_loop_idle_stats()
        assert stats_after["idle_seconds"] == 0.0
        assert stats_after["poll_seconds"] == 0.0
        assert stats_after["select_calls"] == 0

    async def test_profiler_continues_after_reset(self):
        """Profiler should continue measuring after reset."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        await asyncio.sleep(0.05)
        reset_event_loop_idle_stats()

        # Continue measuring
        await asyncio.sleep(0.05)

        stats = get_event_loop_idle_stats()
        assert stats["idle_seconds"] > 0.04
        assert stats["idle_seconds"] < 0.06


class TestMultipleLoops:
    """Tests for handling multiple event loops."""

    def test_separate_loops_both_profiled(self):
        """Each new event loop should be profiled independently."""

        # First loop
        async def run_loop1():
            loop = asyncio.get_running_loop()
            install_event_loop_idle_profiler(loop)
            await asyncio.sleep(0.05)
            return get_event_loop_idle_stats()["idle_seconds"]

        asyncio.run(run_loop1())
        idle_after_loop1 = get_event_loop_idle_stats()["idle_seconds"]

        # Second loop (new loop, but profiler should install)
        async def run_loop2():
            loop = asyncio.get_running_loop()
            install_event_loop_idle_profiler(loop)
            await asyncio.sleep(0.05)
            return get_event_loop_idle_stats()["idle_seconds"]

        asyncio.run(run_loop2())
        idle_after_loop2 = get_event_loop_idle_stats()["idle_seconds"]

        # Both loops should have contributed to idle time
        # (stats accumulate across loops since they're global)
        assert idle_after_loop2 > idle_after_loop1


class TestEdgeCases:
    """Tests for edge cases."""

    async def test_double_install_is_safe(self):
        """Installing profiler twice on same loop should be safe."""
        loop = asyncio.get_running_loop()

        install_event_loop_idle_profiler(loop)
        install_event_loop_idle_profiler(loop)  # Should be no-op

        await asyncio.sleep(0.05)

        stats = get_event_loop_idle_stats()
        # Should not double-count
        assert stats["idle_seconds"] < 0.1

    async def test_stats_snapshot_is_copy(self):
        """get_event_loop_idle_stats should return current values."""
        loop = asyncio.get_running_loop()
        install_event_loop_idle_profiler(loop)

        await asyncio.sleep(0.05)
        stats1 = get_event_loop_idle_stats()

        await asyncio.sleep(0.05)
        stats2 = get_event_loop_idle_stats()

        # Second snapshot should have more idle time
        assert stats2["idle_seconds"] > stats1["idle_seconds"]
