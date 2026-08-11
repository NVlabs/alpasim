# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import os
import stat
from pathlib import Path
from typing import Any

import alpasim_wizard.utils as wizard_utils


def test_ensure_sqsh_path_ignores_creator_umask(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cache_dir = tmp_path / "sqsh"
    cache_dir.mkdir()
    enroot_config = tmp_path / "enroot"
    enroot_config.mkdir()

    def fake_enroot_import(command: list[str], **_: Any) -> None:
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_bytes(b"sqsh")

    monkeypatch.setattr(wizard_utils.subprocess, "run", fake_enroot_import)

    previous_umask = os.umask(0o077)
    try:
        sqsh_path = wizard_utils.ensure_sqsh_path(
            "nvcr.io/nvidian/alpamayo/alpasim-base:0.118.0",
            [str(cache_dir)],
            str(enroot_config),
        )
    finally:
        os.umask(previous_umask)

    lock_path = cache_dir / ".lock_alpasim_base_0.118.0.sqsh.lock"
    assert stat.S_IMODE(Path(sqsh_path).stat().st_mode) == 0o644
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o666
