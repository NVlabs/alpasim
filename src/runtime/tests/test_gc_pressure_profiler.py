# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Tests for the GC pressure profiler.

Verifies that GC callbacks fire correctly, stats accumulate as expected,
and freeze_gc() works correctly.
"""

import gc
from collections import Counter
from time import perf_counter
from unittest.mock import patch

import pytest
from alpasim_runtime.gc_pressure_profiler import (
    _gc_callback,
    freeze_gc,
    get_gc_pressure_stats,
    install_gc_pressure_profiler,
    reset_gc_pressure_stats,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset profiler state and restore flags before and after each test."""
    import alpasim_runtime.gc_pressure_profiler as mod

    saved_census = mod.STALL_CENSUS_ENABLED
    reset_gc_pressure_stats()
    yield
    reset_gc_pressure_stats()
    mod.STALL_CENSUS_ENABLED = saved_census


class TestGcCallback:
    """Tests for the gc.callbacks hook."""

    def test_callback_accumulates_duration(self):
        """A start/stop pair should record duration and count."""
        _gc_callback("start", {"generation": 0})
        _gc_callback("stop", {"generation": 0, "collected": 5})

        stats = get_gc_pressure_stats()
        assert stats["collection_count"] == 1
        assert stats["collected_total"] == 5
        assert stats["total_duration_s"] >= 0.0
        assert stats["gen0_count"] == 1
        assert stats["gen1_count"] == 0
        assert stats["gen2_count"] == 0

    def test_callback_tracks_max_duration(self):
        """Max duration should track the longest collection."""
        _gc_callback("start", {"generation": 1})
        _gc_callback("stop", {"generation": 1, "collected": 1})
        first_max = get_gc_pressure_stats()["max_duration_s"]

        _gc_callback("start", {"generation": 2})
        _gc_callback("stop", {"generation": 2, "collected": 10})
        stats = get_gc_pressure_stats()

        assert stats["collection_count"] == 2
        assert stats["max_duration_s"] >= first_max
        assert stats["gen1_count"] == 1
        assert stats["gen2_count"] == 1

    def test_callback_warns_on_stall(self):
        """Collections over STALL_THRESHOLD_S should log a warning."""
        with patch("alpasim_runtime.gc_pressure_profiler.logger") as mock_logger:
            with patch("alpasim_runtime.gc_pressure_profiler._snapshot_gen2_types"):
                _gc_callback("start", {"generation": 2})
                import alpasim_runtime.gc_pressure_profiler as mod

                mod._gc_phase_start = perf_counter() - 0.2  # 200ms ago
                _gc_callback("stop", {"generation": 2, "collected": 100})

                mock_logger.warning.assert_called_once()
                assert "GC stall" in mock_logger.warning.call_args[0][0]

    def test_stall_calls_diff_census(self):
        """A gen-2 stall with STALL_CENSUS_ENABLED should call _log_stall_diff."""
        import alpasim_runtime.gc_pressure_profiler as mod

        mod.STALL_CENSUS_ENABLED = True
        with patch("alpasim_runtime.gc_pressure_profiler._log_stall_diff") as mock_diff:
            with patch(
                "alpasim_runtime.gc_pressure_profiler._snapshot_gen2_types",
                return_value=Counter({"Task": 500, "dict": 200}),
            ):
                _gc_callback("start", {"generation": 2})
                mod._gc_phase_start = perf_counter() - 0.2
                _gc_callback("stop", {"generation": 2, "collected": 100})

                mock_diff.assert_called_once()


class TestStallSnapshotEdgeCases:
    """Tests for stale-snapshot and generation-mismatch edge cases."""

    def test_gen0_stall_does_not_use_stale_gen2_snapshot(self):
        """A gen-2 start followed by a gen-0 stall must not log a diff
        based on the gen-2 snapshot."""
        import alpasim_runtime.gc_pressure_profiler as mod

        mod.STALL_CENSUS_ENABLED = True

        with patch(
            "alpasim_runtime.gc_pressure_profiler._snapshot_gen2_types",
            return_value=Counter({"Task": 500}),
        ):
            # gen-2 start captures a snapshot
            _gc_callback("start", {"generation": 2})
            assert mod._pre_stall_snapshot is not None

            # gen-0 start should clear that stale snapshot
            _gc_callback("start", {"generation": 0})
            assert mod._pre_stall_snapshot is None

            # gen-0 stall should NOT attempt a diff
            mod._gc_phase_start = perf_counter() - 0.2
            with patch(
                "alpasim_runtime.gc_pressure_profiler._log_stall_diff"
            ) as mock_diff:
                _gc_callback("stop", {"generation": 0, "collected": 50})
                mock_diff.assert_not_called()

    def test_snapshot_cleared_after_normal_gen2_collection(self):
        """A gen-2 collection that does NOT stall should still clear the snapshot."""
        import alpasim_runtime.gc_pressure_profiler as mod

        mod.STALL_CENSUS_ENABLED = True

        with patch(
            "alpasim_runtime.gc_pressure_profiler._snapshot_gen2_types",
            return_value=Counter({"dict": 100}),
        ):
            _gc_callback("start", {"generation": 2})
            assert mod._pre_stall_snapshot is not None

            # Fast gen-2 stop (no stall)
            _gc_callback("stop", {"generation": 2, "collected": 10})
            assert mod._pre_stall_snapshot is None


class TestInstallAndStats:
    """Tests for install, stats retrieval, and reset."""

    def test_install_registers_callback(self):
        """install should add _gc_callback to gc.callbacks."""
        import alpasim_runtime.gc_pressure_profiler as mod

        mod._installed = False
        initial_count = len(gc.callbacks)

        install_gc_pressure_profiler()
        assert len(gc.callbacks) == initial_count + 1
        assert gc.callbacks[-1] is _gc_callback

        # Cleanup
        gc.callbacks.remove(_gc_callback)
        mod._installed = False

    def test_install_is_idempotent(self):
        """Calling install twice should not double-register."""
        import alpasim_runtime.gc_pressure_profiler as mod

        mod._installed = False
        initial_count = len(gc.callbacks)

        install_gc_pressure_profiler()
        install_gc_pressure_profiler()  # no-op
        assert len(gc.callbacks) == initial_count + 1

        # Cleanup
        gc.callbacks.remove(_gc_callback)
        mod._installed = False

    def test_reset_clears_all_stats(self):
        """reset should zero all accumulators."""
        _gc_callback("start", {"generation": 0})
        _gc_callback("stop", {"generation": 0, "collected": 42})

        stats = get_gc_pressure_stats()
        assert stats["collection_count"] > 0

        reset_gc_pressure_stats()
        stats = get_gc_pressure_stats()

        assert stats["total_duration_s"] == 0.0
        assert stats["max_duration_s"] == 0.0
        assert stats["collection_count"] == 0
        assert stats["collected_total"] == 0
        assert stats["gen0_count"] == 0
        assert stats["gen1_count"] == 0
        assert stats["gen2_count"] == 0

    def test_stats_returns_complete_dict(self):
        """get_gc_pressure_stats should return all expected keys."""
        expected_keys = {
            "total_duration_s",
            "max_duration_s",
            "collection_count",
            "collected_total",
            "gen0_count",
            "gen1_count",
            "gen2_count",
        }
        assert set(get_gc_pressure_stats().keys()) == expected_keys


class TestFreezeGc:
    """Tests for gc.freeze() via freeze_gc()."""

    def test_freeze_moves_objects_to_permanent_generation(self):
        """After freeze, tracked object count should drop dramatically."""
        _keep = [dict(x=i) for i in range(50)]

        gc.collect()
        before = len(gc.get_objects())

        freeze_gc()

        after = len(gc.get_objects())
        assert gc.get_freeze_count() > 0
        assert after < before

        del _keep
        gc.unfreeze()

    def test_freeze_returns_frozen_count(self):
        """freeze_gc should return the number of frozen objects."""
        result = freeze_gc()
        assert result > 0
        assert result == gc.get_freeze_count()

        gc.unfreeze()

    def test_freeze_makes_gen2_sweep_fast(self):
        """After freeze, a gen2 collection should be fast because
        it only traverses newly allocated objects."""
        freeze_gc()

        # Allocate some objects post-freeze
        _throwaway = [dict() for _ in range(100)]
        assert len(_throwaway) == 100

        t0 = perf_counter()
        gc.collect(2)
        duration = perf_counter() - t0

        # Should be well under 10ms (was 634ms pre-freeze with 383k objects)
        assert duration < 0.01

        gc.unfreeze()
