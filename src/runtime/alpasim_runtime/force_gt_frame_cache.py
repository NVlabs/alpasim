# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Globally shared, filesystem-backed cache of force-GT camera frames.

During force-GT warmup the ego follows the recorded trajectory, so frames for a
given scene + render configuration are reused instead of re-rendered. The cache
lives on shared storage (e.g. Lustre) with layout::

    <cache_dir>/<scene_uuid>/<render_signature>/<cam_id>__<start_us>_<end_us>.<ext>

Entries are world readable/writable so other users on the same mount can extend
and reuse the cache. See ``force_gt_cache_signature`` for the render signature.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Permissions for cache entries: directories world rwx, files world rw. The
# cache is intentionally shared, so anyone on the same mount may extend it.
_DIR_MODE = 0o777
_FILE_MODE = 0o666


def _sanitize(component: str) -> str:
    """Make a single path component filesystem-safe (no separators)."""
    return component.replace("/", "_").replace("\\", "_").replace("..", "_")


@dataclass(frozen=True)
class ForceGtFrameKey:
    """Identity of one cached force-GT camera frame.

    ``scene_uuid`` and ``render_signature`` are the per-rollout directory levels
    (carried on ``RolloutState``); the rest identify the frame within them. The
    scene is keyed by its stable dataset UUID rather than the USDZ filename,
    since the two are not always the same (e.g. for the public dataset).
    """

    scene_uuid: str
    render_signature: str
    camera_logical_id: str
    frame_start_us: int
    frame_end_us: int
    extension: str

    def relative_path(self) -> Path:
        """Return the cache-relative path for this frame."""
        filename = (
            f"{_sanitize(self.camera_logical_id)}"
            f"__{self.frame_start_us}_{self.frame_end_us}.{self.extension}"
        )
        return (
            Path(_sanitize(self.scene_uuid))
            / _sanitize(self.render_signature)
            / filename
        )


class ForceGtFrameCache:
    """Filesystem-backed cache of force-GT frames rooted at ``cache_dir``.

    Reads and writes are safe to interleave across processes: writes go to a
    temporary file and are atomically renamed into place. Concurrent misses for
    the same key may render more than once; the last atomic write wins.
    """

    def __init__(self, cache_dir: str | os.PathLike[str]) -> None:
        self._root = Path(cache_dir)

    @property
    def root(self) -> Path:
        """Return the cache root directory."""
        return self._root

    def path_for(self, key: ForceGtFrameKey) -> Path:
        """Return the absolute on-disk path for ``key``."""
        return self._root / key.relative_path()

    def get(self, key: ForceGtFrameKey) -> bytes | None:
        """Return cached image bytes for ``key`` or ``None`` on a miss."""
        path = self.path_for(key)
        try:
            return path.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            return None
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read force-GT cache entry %s: %s", path, exc)
            return None

    def put(self, key: ForceGtFrameKey, image_bytes: bytes) -> None:
        """Store ``image_bytes`` for ``key`` atomically with shared permissions."""
        path = self.path_for(key)
        self._ensure_dir(path.parent)
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=".tmp-", suffix=f".{key.extension}"
            )
            try:
                with os.fdopen(fd, "wb") as tmp_file:
                    tmp_file.write(image_bytes)
                with contextlib.suppress(OSError):
                    os.chmod(tmp_name, _FILE_MODE)
                os.replace(tmp_name, path)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp_name)
                raise
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write force-GT cache entry %s: %s", path, exc)

    def _ensure_dir(self, directory: Path) -> None:
        """Create ``directory`` (within the root) and grant world rwx access."""
        directory.mkdir(parents=True, exist_ok=True)
        current = directory
        while True:
            with contextlib.suppress(OSError):
                os.chmod(current, _DIR_MODE)
            if current == self._root or current.parent == current:
                break
            current = current.parent
