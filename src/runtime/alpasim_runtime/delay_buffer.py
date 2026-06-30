# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from collections import deque
from typing import Any


class DelayBuffer:
    """
    This class provides the capability to perform time-based delays of arbitrary
    data types through the specification of a delay duration in microseconds and
    the addition/retrieval of objects to the delay model at specific timestamps.
    """

    def __init__(self, delay_us: int):
        """
        Initialize the delay model with a specified delay in microseconds.

        :param delay_us: Delay duration in microseconds.
        """
        self.delay_us = delay_us
        self.queue: deque = deque()

    def add(self, obj: Any, timestamp_us: int) -> None:
        """
        Add an object to the delay model at a specific timestamp.

        :param obj: The object to be delayed.
        :param timestamp_us: The timestamp at which the object is added.
        """
        if self.queue and (timestamp_us <= self.queue[-1][0]):
            raise ValueError("Timestamps must be in strictly ascending order")
        self.queue.append((timestamp_us, obj))

    def item_at(self, timestamp_us: int) -> tuple[int | None, Any]:
        """
        Retrieve the object and its stored timestamp that has completed its delay.

        :param timestamp_us: The current timestamp to check for delayed objects.
        :return: (stored_timestamp, object) of the oldest item that has met the delay
                 requirement, or (None, None) if the buffer is empty.
        """
        while (
            len(self.queue) > 1 and (timestamp_us - self.queue[1][0]) >= self.delay_us
        ):
            self.queue.popleft()
        if self.queue:
            return self.queue[0][0], self.queue[0][1]
        return None, None

    def at(self, timestamp_us: int) -> Any:
        """
        Retrieve the object that has completed its delay at the given timestamp.

        :param timestamp_us: The current timestamp to check for delayed objects.
        :return: The oldest object that has met the delay requirement, None if no object is ready.
        """
        _, obj = self.item_at(timestamp_us)
        return obj
