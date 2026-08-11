# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the globally shared, filesystem-backed force-GT frame cache."""

from __future__ import annotations

import stat
from pathlib import Path

from alpasim_runtime.force_gt_frame_cache import ForceGtFrameCache, ForceGtFrameKey


def _key(
    camera: str = "cam_front", start: int = 0, end: int = 33_000
) -> ForceGtFrameKey:
    return ForceGtFrameKey(
        scene_uuid="scene-uuid-0001",
        render_signature="jpeg-abc123",
        camera_logical_id=camera,
        frame_start_us=start,
        frame_end_us=end,
        extension="jpg",
    )


def test_miss_then_roundtrip_and_layout(tmp_path: Path) -> None:
    cache = ForceGtFrameCache(tmp_path)
    assert cache.get(_key()) is None

    cache.put(_key(), b"frame-bytes")
    assert cache.get(_key()) == b"frame-bytes"
    assert cache.path_for(_key()) == (
        tmp_path / "scene-uuid-0001" / "jpeg-abc123" / "cam_front__0_33000.jpg"
    )
    # Shared across instances (another worker/process on the same mount).
    assert ForceGtFrameCache(tmp_path).get(_key()) == b"frame-bytes"


def test_keys_are_distinct_per_camera_and_window(tmp_path: Path) -> None:
    cache = ForceGtFrameCache(tmp_path)
    cache.put(_key(camera="cam_front"), b"front")
    cache.put(_key(camera="cam_rear"), b"rear")
    cache.put(_key(start=33_000, end=66_000), b"later")

    assert cache.get(_key(camera="cam_front")) == b"front"
    assert cache.get(_key(camera="cam_rear")) == b"rear"
    assert cache.get(_key(start=33_000, end=66_000)) == b"later"


def test_written_files_and_dirs_are_world_accessible(tmp_path: Path) -> None:
    cache = ForceGtFrameCache(tmp_path)
    cache.put(_key(), b"frame-bytes")

    file_path = cache.path_for(_key())
    assert stat.S_IMODE(file_path.stat().st_mode) & 0o666 == 0o666
    for directory in (file_path.parent, file_path.parent.parent):
        assert stat.S_IMODE(directory.stat().st_mode) & 0o777 == 0o777
