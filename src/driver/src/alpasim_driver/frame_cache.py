"""Session-specific helpers for driver service."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import Callable, List, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable)


@dataclass
class FrameEntry:
    """Represents a single camera frame."""

    timestamp_us: int
    image: np.ndarray


def synchronized(method: F) -> F:
    @wraps(method)
    def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        with self._lock:  # noqa: SLF001
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


@dataclass
class FrameCache:
    """Keeps a bounded, time-ordered buffer of frames for a session.

    When subsample_factor > 1, the cache stores more frames than context_length
    to allow selecting every Nth frame for inference. This enables running
    inference at higher frequencies while maintaining the expected temporal
    spacing between frames that the model was trained on.

    Example with context_length=3, subsample_factor=2:
        - Buffer stores up to 3*2 = 6 frames
        - At inference, selects frames [newest, newest-2, newest-4]
        - Next inference selects [newest, newest-2, newest-4] from shifted buffer
    """

    context_length: int
    camera_id: str = ""
    subsample_factor: int = 1  # 1 = no subsampling, 2 = every other frame, etc.
    entries: List[FrameEntry] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @synchronized
    def add_image(self, timestamp_us: int, image: np.ndarray) -> None:
        """Insert an image while keeping entries ordered by timestamp."""
        inserted = False
        # Iterate from newest to oldest since most inserts append.
        for offset, entry in enumerate(reversed(self.entries)):
            if entry.timestamp_us == timestamp_us:
                raise ValueError(f"Frame {timestamp_us} already exists in cache")
            if entry.timestamp_us < timestamp_us:
                insert_at = len(self.entries) - offset
                self.entries.insert(insert_at, FrameEntry(timestamp_us, image))
                inserted = True
                break
        if not inserted:
            self.entries.insert(0, FrameEntry(timestamp_us, image))

        self._prune()

    @synchronized
    def frame_count(self) -> int:
        """Total number of frames currently cached."""
        return len(self.entries)

    def min_frames_required(self) -> int:
        """Minimum number of frames needed for inference."""
        return (self.context_length - 1) * self.subsample_factor + 1

    @synchronized
    def has_enough_frames(self) -> bool:
        """Check if there are enough frames for inference."""
        return len(self.entries) >= self.min_frames_required()

    @synchronized
    def latest_frame_entries(self, count: int) -> List[FrameEntry]:
        """Return the newest `count` frame entries with subsampling (oldest first).

        When subsample_factor > 1, selects every Nth frame starting from the
        newest frame and walking backwards. This maintains the expected temporal
        spacing between frames while allowing inference at higher frequencies.

        Args:
            count: Number of frames to return.

        Returns:
            List of FrameEntry objects, ordered oldest to newest.

        Raises:
            ValueError: If insufficient frames are available for subsampled selection.
        """
        min_required = (count - 1) * self.subsample_factor + 1
        if len(self.entries) < min_required:
            raise ValueError(
                f"Insufficient frames: have {len(self.entries)}, need at least "
                f"{min_required} (count={count}, subsample_factor={self.subsample_factor})"
            )

        # Select frames: start from newest, walk backwards by subsample_factor
        selected_indices = []
        idx = len(self.entries) - 1  # Start at newest
        for _ in range(count):
            selected_indices.append(idx)
            idx -= self.subsample_factor

        # Reverse to get oldest-first order
        selected_indices = selected_indices[::-1]

        return [self.entries[i] for i in selected_indices]

    def _prune(self) -> None:
        """Bound the cache to accommodate subsampled context queries."""
        max_entries = self.context_length * self.subsample_factor
        excess = len(self.entries) - max_entries
        if excess <= 0:
            return
        del self.entries[:excess]
