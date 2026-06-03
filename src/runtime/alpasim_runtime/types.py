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
    first_end_us: int | None = None

    def __post_init__(self) -> None:
        if self.interval_us <= 0:
            raise ValueError("interval_us must be positive")
        if self.duration_us < 0:
            raise ValueError("duration_us must be non-negative")

    def ith_trigger(self, i: int) -> Trigger:
        """Returns the i-th trigger of the clock since self.start_us"""
        if i < 0:
            raise ValueError(f"Trigger index must be non-negative, got {i}")
        if i == 0 and self.first_end_us is not None:
            return Clock.Trigger(
                range(self.start_us, self.first_end_us),
                sequential_idx=i,
            )
        first_end_us = (
            self.first_end_us
            if self.first_end_us is not None
            else self.start_us + self.duration_us
        )
        end_us = first_end_us + i * self.interval_us
        return Clock.Trigger(
            range(
                end_us - self.duration_us,
                end_us,
            ),
            sequential_idx=i,
        )


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
        cls, camera_cfg: RuntimeCameraConfig, first_frame_range_us: range
    ) -> RuntimeCamera:
        """Build a `RuntimeCamera` from a scenario `CameraConfig`."""

        first_frame_duration_us = first_frame_range_us.stop - first_frame_range_us.start
        duration_us = camera_cfg.shutter_duration_us or first_frame_duration_us
        clock = Clock(
            interval_us=camera_cfg.frame_interval_us,
            duration_us=duration_us,
            start_us=first_frame_range_us.start,
            first_end_us=first_frame_range_us.stop,
        )
        return cls(
            logical_id=camera_cfg.logical_id,
            render_resolution_hw=(camera_cfg.height, camera_cfg.width),
            clock=clock,
        )
