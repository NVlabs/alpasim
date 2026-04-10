# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import pytest
from alpasim_runtime.types import Clock


# Testing for Clock
@pytest.fixture
def clock_10_1_25() -> Clock:
    return Clock(interval_us=10, duration_us=1, start_us=25)


def check_triggers_against_expected(
    triggers, expected_trigger_starts, clock_duration_us=1
):
    assert len(triggers) == len(expected_trigger_starts)
    for i, trigger in enumerate(triggers):
        assert trigger.time_range_us.start == expected_trigger_starts[i]
        assert (
            trigger.time_range_us.stop == expected_trigger_starts[i] + clock_duration_us
        )


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


def test_clock_last_trigger(clock_10_1_25):
    times_trigger_is_25 = [25, 26, 27]
    times_trigger_is_35 = [35, 36, 37]
    for time in times_trigger_is_25:
        trigger = clock_10_1_25.last_trigger(time)
        assert trigger.time_range_us.start == 25
        assert trigger.time_range_us.stop == 26
        assert trigger.sequential_idx == 0
    for time in times_trigger_is_35:
        trigger = clock_10_1_25.last_trigger(time)
        assert trigger.time_range_us.start == 35
        assert trigger.time_range_us.stop == 36
        assert trigger.sequential_idx == 1


def test_clock_triggers_completed_in_range(clock_10_1_25):
    triggers = clock_10_1_25.triggers_completed_in_range(range(30, 60))
    expected_trigger_starts = [35, 45, 55]
    check_triggers_against_expected(triggers, expected_trigger_starts)


def test_clock_triggers_completed_in_range_boundaries(clock_10_1_25):
    # expect that the end of the range is inclusive, and the start is exclusive
    # return Clock(interval_us=10, duration_us=1, start_us=25)
    triggers = clock_10_1_25.triggers_completed_in_range(range(26, 46))
    # expect triggers include [35-36, 45-46]
    # 25-26 excluded because exclusive start
    # 45-46 included because inclusive end
    expected_trigger_starts = [35, 45]
    check_triggers_against_expected(triggers, expected_trigger_starts)


def test_clock_triggers_completed_in_range_zero_duration_input(clock_10_1_25):
    # Because there is ambiguity on whether or not a zero-duration input interval
    # started or ended on a trigger, no triggers should be returned
    # Perhaps there is some discussion as to whether or not this should
    # raise an error, but for now it will return an empty list
    for timestamp in range(25, 60):
        triggers = clock_10_1_25.triggers_completed_in_range(
            range(timestamp, timestamp)
        )
        assert len(triggers) == 0


def test_clock_triggers_completed_in_range_no_triggers(clock_10_1_25):
    triggers = clock_10_1_25.triggers_completed_in_range(range(0, 20))
    assert len(triggers) == 0


def test_clock_triggers_completed_in_range_with_straddles():
    clock = Clock(interval_us=10, duration_us=5, start_us=25)

    # results for 27-46 should include [25-30, 35-40] when skip straddling is disabled
    triggers = clock.triggers_completed_in_range(range(27, 46), skip_straddles=False)
    check_triggers_against_expected(triggers, [25, 35], clock.duration_us)

    # results for 27-46 should only include [35-40] when skip straddling is enabled
    triggers = clock.triggers_completed_in_range(range(27, 46), skip_straddles=True)
    check_triggers_against_expected(triggers, [35], clock.duration_us)

    # results for 25-46 should only include [25-30, 35-40] when skip straddling is enabled
    triggers = clock.triggers_completed_in_range(range(25, 46), skip_straddles=True)
    check_triggers_against_expected(triggers, [25, 35], clock.duration_us)


def test_clock_triggers_completed_in_range_zero_duration_clock():
    clock = Clock(interval_us=10, duration_us=0, start_us=25)

    triggers = clock.triggers_completed_in_range(range(25, 47))
    check_triggers_against_expected(triggers, [35, 45], clock.duration_us)
