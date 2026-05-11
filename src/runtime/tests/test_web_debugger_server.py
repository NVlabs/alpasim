# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from pathlib import Path

from alpasim_runtime.web_debugger.server import _load_available_scene_ids


class _StubMapProvider:
    def __init__(self, scene_ids: list[str], fail: bool = False) -> None:
        self._scene_ids = scene_ids
        self._fail = fail

    def list_scene_ids(self) -> list[str]:
        if self._fail:
            raise RuntimeError("discovery failed")
        return list(self._scene_ids)


def test_load_available_scene_ids_prefers_discovered_union(tmp_path: Path) -> None:
    user_config = tmp_path / "user-config.yaml"
    user_config.write_text(
        """
scenes:
  - scene_id: scene-from-config
  - scene_id: shared-scene
""".strip(),
        encoding="utf-8",
    )

    scene_ids = _load_available_scene_ids(
        _StubMapProvider(["scene-from-glob", "shared-scene"]),
        str(user_config),
    )

    assert scene_ids == ["scene-from-config", "scene-from-glob", "shared-scene"]


def test_load_available_scene_ids_falls_back_to_user_config(tmp_path: Path) -> None:
    user_config = tmp_path / "user-config.yaml"
    user_config.write_text(
        """
scenes:
  - scene_id: scene-from-config
""".strip(),
        encoding="utf-8",
    )

    scene_ids = _load_available_scene_ids(
        _StubMapProvider([], fail=True),
        str(user_config),
    )

    assert scene_ids == ["scene-from-config"]
