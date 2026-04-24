# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
GC pressure profiler: instrument and mitigate CPython garbage-collection stalls.

CPython's cyclic GC can hold the GIL for hundreds of milliseconds when
generation-2 sweeps clean up large numbers of weak-referenced objects
(e.g. asyncio.Task tracked via WeakSet, gRPC message churn).  This module:

1. Registers a ``gc.callbacks`` hook that times every collection and warns
   on stalls above a configurable threshold.
2. Optionally snapshots the top object types in generation 2 for diagnosis.

All tunables are module-level constants that control instrumentation
behaviour only.  ``gc.freeze()`` is not merely diagnostic -- it permanently
moves objects to a frozen generation for the process lifetime and is the
primary mitigation for GC stalls.
"""

from __future__ import annotations

import gc
import logging
from collections import Counter
from time import perf_counter
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

STALL_THRESHOLD_S: float = 0.1
"""Log a WARNING for any single GC collection longer than this."""

STALL_CENSUS_ENABLED: bool = False
"""Run _log_stall_diff() when a gen-2 stall is detected.  Disabled by default
because the pre/post snapshot itself traverses gen-2, adding latency.  Enable
only when diagnosing which object types are causing the stall."""

TYPE_CENSUS_TOP_N: int = 20
"""Number of top object types to log in the gen-2 census."""

# ---------------------------------------------------------------------------
# Global accumulators (written by callback, read by get_gc_pressure_stats)
# ---------------------------------------------------------------------------

GC_TOTAL_DURATION_S: float = 0.0
GC_MAX_DURATION_S: float = 0.0
GC_COLLECTION_COUNT: int = 0
GC_COLLECTED_TOTAL: int = 0
GC_GEN_COUNTS: list[int] = [0, 0, 0]

_gc_phase_start: float = 0.0
_gc_phase_generation: int = 0
_installed: bool = False

# Pre-collection snapshot for stall diff (captured in "start", consumed in "stop")
_pre_stall_snapshot: Optional[Counter[str]] = None

# ---------------------------------------------------------------------------
# GC callback (registered via gc.callbacks)
# ---------------------------------------------------------------------------


def _snapshot_gen2_types() -> Counter[str]:
    """Type-count snapshot of generation-2 objects."""
    try:
        objs = gc.get_objects(generation=2)
    except TypeError:
        objs = gc.get_objects()
    return Counter(type(obj).__qualname__ for obj in objs)


def _gc_callback(phase: str, info: dict) -> None:
    global GC_TOTAL_DURATION_S, GC_MAX_DURATION_S, GC_COLLECTION_COUNT
    global GC_COLLECTED_TOTAL
    global _gc_phase_start, _gc_phase_generation, _pre_stall_snapshot

    if phase == "start":
        _gc_phase_start = perf_counter()
        _gc_phase_generation = info.get("generation", 0)
        # Snapshot gen-2 types BEFORE collection so we can diff what was
        # collected (the dead objects) rather than only seeing survivors.
        # Only gen-2 (rare, ~once per 2 min) and only when census enabled.
        if STALL_CENSUS_ENABLED and _gc_phase_generation == 2:
            _pre_stall_snapshot = _snapshot_gen2_types()
        else:
            _pre_stall_snapshot = None
        return

    # phase == "stop"
    duration = perf_counter() - _gc_phase_start
    generation = _gc_phase_generation
    collected = info.get("collected", 0)

    GC_TOTAL_DURATION_S += duration
    GC_COLLECTION_COUNT += 1
    GC_COLLECTED_TOTAL += collected
    if duration > GC_MAX_DURATION_S:
        GC_MAX_DURATION_S = duration
    if 0 <= generation <= 2:
        GC_GEN_COUNTS[generation] += 1

    if duration >= STALL_THRESHOLD_S:
        logger.warning(
            "GC stall: gen%d collected %d objects in %.1f ms",
            generation,
            collected,
            duration * 1000,
        )
        if STALL_CENSUS_ENABLED and generation == 2 and _pre_stall_snapshot is not None:
            _log_stall_diff(_pre_stall_snapshot, top_n=TYPE_CENSUS_TOP_N)
    _pre_stall_snapshot = None


# ---------------------------------------------------------------------------
# Object-type census (expensive -- call sparingly)
# ---------------------------------------------------------------------------


def _log_stall_diff(pre_counts: Counter[str], top_n: int = TYPE_CENSUS_TOP_N) -> None:
    """Log what was *collected* (pre minus post) and what *survived*.

    Called from the "stop" callback after a stall, using the snapshot
    taken in the "start" callback.  This directly answers "what objects
    caused the expensive sweep" rather than showing only survivors.
    """
    post_counts = _snapshot_gen2_types()
    collected = pre_counts - post_counts  # elements removed by GC
    total_before = sum(pre_counts.values())
    total_after = sum(post_counts.values())
    total_collected = sum(collected.values())

    lines = [
        f"GC stall census: {total_before} objects before, "
        f"{total_after} after, {total_collected} collected"
    ]
    if total_collected > 0:
        lines.append(f"  Top collected types (of {total_collected}):")
        for type_name, count in collected.most_common(top_n):
            lines.append(f"    {type_name:>40s}  {count:>8d}")
    lines.append(f"  Top survivor types (of {total_after}):")
    for type_name, count in post_counts.most_common(top_n):
        lines.append(f"    {type_name:>40s}  {count:>8d}")
    logger.info("\n".join(lines))


def log_gc_type_census(top_n: int = TYPE_CENSUS_TOP_N) -> None:
    """Log the most common object types currently in generation 2."""
    counts = _snapshot_gen2_types()
    total = sum(counts.values())

    lines = [f"Gen-2 object census ({total} objects, top {top_n}):"]
    for type_name, count in counts.most_common(top_n):
        lines.append(f"  {type_name:>40s}  {count:>8d}  ({100 * count / total:.1f}%)")
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Install / stats / reset
# ---------------------------------------------------------------------------


def freeze_gc() -> int:
    """Move all currently tracked objects to a permanent generation.

    After this call gen-0/1/2 are nearly empty, so future cyclic-GC
    sweeps only traverse objects allocated *after* the freeze.  This is
    the primary mitigation for the 500 ms stall: the cost was in walking
    383 k long-lived objects (functions, modules, descriptors) each sweep,
    not in collecting the ~1 k dead ones.

    Called in worker startup after imports and Dispatcher creation, which
    covers the bulk of long-lived objects (~383 k).  If first-rollout
    warmup creates additional long-lived objects (e.g. gRPC session
    caches), a second freeze_gc() call can be added after warmup.
    Successive freezes are safe and additive.

    Returns the number of objects in the permanent (frozen) generation.
    """
    gc.collect()  # clean up garbage first so we don't freeze it
    tracked_before = len(gc.get_objects())
    gc.freeze()
    frozen_count = gc.get_freeze_count()
    tracked_after = len(gc.get_objects())
    logger.info(
        "gc.freeze(): %d objects frozen to permanent generation "
        "(%d tracked before, %d tracked after)",
        frozen_count,
        tracked_before,
        tracked_after,
    )
    return frozen_count


def install_gc_pressure_profiler() -> None:
    """Register the GC callback.  Safe to call multiple times."""
    global _installed

    if _installed:
        logger.debug("GC pressure profiler already installed, skipping")
        return

    gc.callbacks.append(_gc_callback)
    _installed = True

    thresholds = gc.get_threshold()
    logger.info(
        "GC pressure profiler installed (thresholds=%s, stall_threshold=%.3fs, census=%s)",
        thresholds,
        STALL_THRESHOLD_S,
        STALL_CENSUS_ENABLED,
    )


def get_gc_pressure_stats() -> dict:
    """Return a snapshot of all accumulated GC metrics."""
    return {
        "total_duration_s": GC_TOTAL_DURATION_S,
        "max_duration_s": GC_MAX_DURATION_S,
        "collection_count": GC_COLLECTION_COUNT,
        "collected_total": GC_COLLECTED_TOTAL,
        "gen0_count": GC_GEN_COUNTS[0],
        "gen1_count": GC_GEN_COUNTS[1],
        "gen2_count": GC_GEN_COUNTS[2],
    }


def reset_gc_pressure_counters() -> None:
    """Zero metric accumulators.

    Use after freeze_gc() to discard the startup sweep from telemetry.
    """
    global GC_TOTAL_DURATION_S, GC_MAX_DURATION_S, GC_COLLECTION_COUNT
    global GC_COLLECTED_TOTAL, GC_GEN_COUNTS

    GC_TOTAL_DURATION_S = 0.0
    GC_MAX_DURATION_S = 0.0
    GC_COLLECTION_COUNT = 0
    GC_COLLECTED_TOTAL = 0
    GC_GEN_COUNTS = [0, 0, 0]


def setup_gc_pressure_profiler() -> None:
    """Combined startup entry point: freeze, instrument, and reset counters.

    Calls :func:`freeze_gc` first so that the startup ``gc.collect()`` inside
    it is not captured by the profiler, then installs the callback, then resets
    the counters so telemetry reflects only post-startup collections.

    The three underlying functions are kept public for callers that need finer
    control (e.g. a second :func:`freeze_gc` after warmup, or running
    :func:`freeze_gc` without the profiler).
    """
    freeze_gc()
    install_gc_pressure_profiler()
    reset_gc_pressure_counters()


def reset_gc_pressure_stats() -> None:
    """Reset all accumulators and callback state to zero.  Useful for testing."""
    global _pre_stall_snapshot

    reset_gc_pressure_counters()
    _pre_stall_snapshot = None
