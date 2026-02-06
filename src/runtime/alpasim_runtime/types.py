# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass

from alpasim_runtime.config import RuntimeCameraConfig


@dataclass
class Clock:
    """
    Represents a clock which ticks triggers with a given interval (`interval_us`)
    and a given duration (`duration_us`).
    """

    @dataclass
    class Trigger:
        # start and end of sensor acquisition
        time_range_us: range

        # unique and consecutive within camera_id,
        # equivalent to sorting all CameraTriggers by time_range_us.start
        sequential_idx: int

    interval_us: int
    duration_us: int
    start_us: int = 0

    def __post_init__(self) -> None:
        if self.interval_us <= 0:
            raise ValueError("interval_us must be positive")
        if self.duration_us < 0:
            raise ValueError("duration_us must be non-negative")

    @staticmethod
    def int_ceil_div(a: int, b: int) -> int:
        """
        Returns the ceiling of the division of a by b using only integer arithmetic,
        avoiding numerical accuracy issues.

        :param a: The numerator (integer).
        :param b: The denominator (integer, must not be zero).
        :return: The ceiling of the division of a by b.
        """
        if b == 0:
            raise ValueError("Division by zero is not allowed.")

        # Perform integer ceiling division
        return (a + b - 1) // b if a * b > 0 else a // b

    def ith_trigger(self, i: int) -> Trigger:
        """Returns the i-th trigger of the clock since self.start_us"""
        # TODO(mwatson): raise if i is negative???
        return Clock.Trigger(
            range(
                self.start_us + i * self.interval_us,
                self.start_us + self.duration_us + i * self.interval_us,
            ),
            sequential_idx=i,
        )

    def last_trigger(self, time_us: int) -> Trigger:
        """Returns the last trigger to have started (may still be ongoing) at `time_us`."""
        trigger_i = (time_us - self.start_us) // self.interval_us
        return self.ith_trigger(trigger_i)

    def triggers_completed_in_range(
        self, time_range_us: range, skip_straddles: bool = False
    ) -> list[Trigger]:
        """
        Return a list of camera triggers that are entirely completed within `time_range_us`
        Conventional notes:
        - triggers that complete at `time_range_us.stop` are included
        -    rationale: an event/trigger that completes at the given interval is
                        considered to be available on that interval
        - triggers that complete at `time_range_us.start` are not included
        -    rationale: an event/trigger that completed at the start of an interval
                        would have been "available" on the previous interval, since
                        interval[i].start == interval[i-1].end
        - triggers that start before `self.start_us` are not included in results

        Parameters:
        - time_range_us: the range of time to consider
        - skip_straddles: if True, exclude triggers that began before `time_range_us.start`
        """

        """
        Example:
        |           |           |           |     <<< clock trigger starts
        <  >        <  >        <  >        <  >  <<< clock trigger intervals
          [               ]                       <<< time_range_us 1
        ....        ....                          <<< return triggers completed in time_range_us 1 (!skipping straddles)
             [                                 ]  <<< time_range_us 2
                    ....        ....        ....  <<< return triggers completed in time_range_us 2 (!skipping straddles)
                     [                  ]         <<< time_range_us 3
                                ....              <<< return triggers completed in time_range_us 3 (skipping straddles)
        """

        # Note: time_range_us.start + 1 ensures exclusions of triggers that start at time_range_us.start
        #       and the 0-bounding ensures that we only consider triggers that start after self.start_us
        i_min = Clock.int_ceil_div(
            max(time_range_us.start + 1 - self.start_us - self.duration_us, 0),
            self.interval_us,
        )
        i_max = (
            time_range_us.stop - self.start_us - self.duration_us
        ) // self.interval_us

        if skip_straddles:
            first_trigger_start = self.start_us + i_min * self.interval_us
            if time_range_us.start > first_trigger_start:
                i_min += 1

        return [self.ith_trigger(i) for i in range(max(i_min, 0), max(i_max + 1, 0))]


@dataclass
class RuntimeCamera:
    """This class defines which cameras are rendered and how to render them.

    - `logical_id` is the unique identifier for the camera. This references a
        `CameraDefinition` in the camera catalog.
    - `render_resolution_hw` is the resolution of the camera in pixels.
    - `clock` is the clock that determines the timing of the camera.
    """

    logical_id: str
    render_resolution_hw: tuple[int, int]
    clock: Clock

    @classmethod
    def from_camera_config(
        cls, camera_cfg: RuntimeCameraConfig, rig_start_us: int
    ) -> RuntimeCamera:
        """Build a `RuntimeCamera` from a scenario `CameraConfig`."""

        clock = Clock(
            interval_us=camera_cfg.frame_interval_us,
            duration_us=camera_cfg.shutter_duration_us,
            start_us=rig_start_us + camera_cfg.first_frame_offset_us,
        )
        return cls(
            logical_id=camera_cfg.logical_id,
            render_resolution_hw=(camera_cfg.height, camera_cfg.width),
            clock=clock,
        )
