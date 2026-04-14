# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import copy
from collections import OrderedDict

from .models import FrameDataModel, FrameRefModel


class FrameStore:
    """Bounded in-memory store for rendered frames keyed by tick and sensor."""

    def __init__(self, max_retained_ticks: int) -> None:
        self._max_retained_ticks = max(1, max_retained_ticks)
        self._frames_by_tick: OrderedDict[int, dict[str, FrameDataModel]] = OrderedDict()

    def store_tick_frames(
        self,
        tick_id: int,
        frames: list[FrameDataModel],
    ) -> list[FrameRefModel]:
        bucket = {frame.sensor_id: frame for frame in frames}
        self._frames_by_tick[tick_id] = bucket
        self._frames_by_tick.move_to_end(tick_id)
        while len(self._frames_by_tick) > self._max_retained_ticks:
            self._frames_by_tick.popitem(last=False)

        return [
            FrameRefModel(
                sensor_id=frame.sensor_id,
                tick_id=tick_id,
                frame_start_us=frame.frame_start_us,
                frame_end_us=frame.frame_end_us,
                frame_encoding=frame.frame_encoding,
            )
            for frame in bucket.values()
        ]

    def get_frame(self, sensor_id: str, tick_id: int) -> FrameDataModel:
        """Return the newest retained frame for `sensor_id` at or before `tick_id`."""
        for stored_tick in reversed(self._frames_by_tick):
            if stored_tick > tick_id:
                continue
            bucket = self._frames_by_tick[stored_tick]
            if sensor_id in bucket:
                return bucket[sensor_id]
        raise KeyError(f"No retained frame for sensor_id={sensor_id} tick_id<={tick_id}")

    def snapshot(self) -> OrderedDict[int, dict[str, FrameDataModel]]:
        """Return a deep-copied snapshot of retained frames."""
        return copy.deepcopy(self._frames_by_tick)

    def restore(self, snapshot: OrderedDict[int, dict[str, FrameDataModel]]) -> None:
        """Restore retained frames from a previous snapshot."""
        self._frames_by_tick = copy.deepcopy(snapshot)
