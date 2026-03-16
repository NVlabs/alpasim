# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import asyncio
import logging
from time import perf_counter
from typing import Optional

logger = logging.getLogger(__name__)

# Global accumulators for event loop time tracking
EVENT_LOOP_IDLE_SECONDS = 0.0
EVENT_LOOP_SELECT_CALLS = 0
EVENT_LOOP_POLL_SECONDS = 0.0
EVENT_LOOP_WORK_SECONDS = 0.0

# Track the end time of the last select() call to measure work time
_LAST_SELECT_END: Optional[float] = None


def install_event_loop_idle_profiler(
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Install profiler to measure event loop time breakdown.

    Monkey-patches the selector's select() method to measure time spent:
    - idle: blocking waits for I/O (timeout > 0 or None)
    - poll: non-blocking I/O checks (timeout = 0)
    - work: executing Python code between select() calls

    Safe to call multiple times on the same loop - only installs once per loop.

    Args:
        loop: The event loop to patch. If None, uses the current event loop.
    """
    global _LAST_SELECT_END

    loop_cls_name = loop.__class__.__name__
    selector = getattr(loop, "_selector", None)

    if selector is None:
        logger.warning(
            f"Event loop idle profiler: loop '{loop_cls_name}' has no _selector attribute. "
            "Profiling disabled."
        )
        return

    # Check if this selector is already patched by looking for our marker attribute.
    # This is more robust than tracking loop IDs, which can be reused after GC.
    if getattr(selector, "_idle_profiler_patched", False):
        logger.debug(
            "Event loop idle profiler already installed on this loop, skipping"
        )
        return

    original_select = selector.select

    def profiling_select(timeout: Optional[float] = None) -> list:
        global EVENT_LOOP_IDLE_SECONDS
        global EVENT_LOOP_SELECT_CALLS
        global EVENT_LOOP_POLL_SECONDS
        global EVENT_LOOP_WORK_SECONDS
        global _LAST_SELECT_END

        t0 = perf_counter()

        # Time since last select() ended = work time (executing Python code)
        if _LAST_SELECT_END is not None:
            EVENT_LOOP_WORK_SECONDS += t0 - _LAST_SELECT_END

        events = original_select(timeout)
        t1 = perf_counter()

        _LAST_SELECT_END = t1

        elapsed = t1 - t0
        EVENT_LOOP_SELECT_CALLS += 1

        # Heuristic: only count as "idle" when we *intended* to block
        # (timeout None or > 0). This avoids counting zero-timeout polls
        # which are used to check for ready events without blocking.
        if timeout is None or timeout > 0:
            EVENT_LOOP_IDLE_SECONDS += elapsed
        else:
            # Zero-timeout poll (non-blocking check)
            EVENT_LOOP_POLL_SECONDS += elapsed

        return events

    selector.select = profiling_select
    selector._idle_profiler_patched = True
    logger.info("Event loop idle profiler installed")


def get_event_loop_idle_stats() -> dict:
    """
    Get all event loop time statistics.

    Returns:
        Dict with:
        - idle_seconds: time spent blocking on I/O (waiting for events)
        - poll_seconds: time spent in non-blocking I/O checks
        - work_seconds: time spent executing Python code between select() calls
        - select_calls: number of select() calls made
    """
    return {
        "idle_seconds": EVENT_LOOP_IDLE_SECONDS,
        "poll_seconds": EVENT_LOOP_POLL_SECONDS,
        "work_seconds": EVENT_LOOP_WORK_SECONDS,
        "select_calls": EVENT_LOOP_SELECT_CALLS,
    }


def snapshot_event_loop_idle_stats() -> dict:
    """
    Take a snapshot of current idle time statistics.
    """
    return get_event_loop_idle_stats()


def reset_event_loop_idle_stats() -> None:
    """
    Reset all event loop statistics to zero.

    Useful for testing or when starting a new measurement period.
    """
    global EVENT_LOOP_IDLE_SECONDS
    global EVENT_LOOP_SELECT_CALLS
    global EVENT_LOOP_POLL_SECONDS
    global EVENT_LOOP_WORK_SECONDS
    global _LAST_SELECT_END

    EVENT_LOOP_IDLE_SECONDS = 0.0
    EVENT_LOOP_SELECT_CALLS = 0
    EVENT_LOOP_POLL_SECONDS = 0.0
    EVENT_LOOP_WORK_SECONDS = 0.0
    _LAST_SELECT_END = None
