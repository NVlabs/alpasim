# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import pytest
from alpasim_runtime.config import RuntimeCameraConfig
from alpasim_runtime.types import Clock, RuntimeCamera


# Testing for Clock
@pytest.fixture
def clock_10_1_25() -> Clock:
    return Clock(interval_us=10, duration_us=1, start_us=25)


def test_clock_invalid_config():
    with pytest.raises(ValueError):
        Clock(interval_us=0, duration_us=1, start_us=25)
    with pytest.raises(ValueError):
        Clock(interval_us=10, duration_us=-1, start_us=25)


def test_clock_ith_trigger(clock_10_1_25):
    trigger = clock_10_1_25.ith_trigger(0)
    assert trigger.time_range_us.start == 25
    assert trigger.time_range_us.stop == 26
    assert trigger.sequential_idx == 0

    trigger = clock_10_1_25.ith_trigger(1)
    assert trigger.time_range_us.start == 35
    assert trigger.time_range_us.stop == 36
    assert trigger.sequential_idx == 1


def test_clock_ith_trigger_negative_index_raises(clock_10_1_25):
    with pytest.raises(ValueError, match="non-negative"):
        clock_10_1_25.ith_trigger(-1)


def test_clock_first_trigger_can_use_recorded_window():
    clock = Clock(interval_us=100, duration_us=30, start_us=1_000, first_end_us=1_059)

    first_trigger = clock.ith_trigger(0)
    assert first_trigger.time_range_us == range(1_000, 1_059)

    second_trigger = clock.ith_trigger(1)
    assert second_trigger.time_range_us == range(1_129, 1_159)


def test_runtime_camera_clock_is_anchored_to_first_frame_range():
    camera = RuntimeCamera.from_camera_config(
        RuntimeCameraConfig(
            logical_id="camera_front",
            frame_interval_us=100_000,
            shutter_duration_us=17_000,
        ),
        first_frame_range_us=range(1_483_500, 1_500_000),
    )

    first_trigger = camera.clock.ith_trigger(0)
    assert first_trigger.time_range_us.start == 1_483_500
    assert first_trigger.time_range_us.stop == 1_500_000

    second_trigger = camera.clock.ith_trigger(1)
    assert second_trigger.time_range_us.start == 1_583_000
    assert second_trigger.time_range_us.stop == 1_600_000
