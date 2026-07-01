# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

# pylint: disable=import-error,redefined-outer-name


import os
from pathlib import Path

import pytest

_PREFERRED_USDZ = "eae31fd6-fbf7-4303-995b-6451f5c303ef.usdz"


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _candidate_usdz_dirs(data_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("TRAFFICSIM_USDZ_DIR", "ALPASIM_USDZ_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())

    alpasim_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            data_dir / "usdz",
            alpasim_root / "data" / "nre-artifacts" / "all-usdzs",
            alpasim_root / "data" / "nre-artifacts" / "oss" / "all-usdzs",
        ]
    )
    return candidates


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return _default_data_dir()


@pytest.fixture(scope="session")
def usdz_data_dir(data_dir: Path) -> Path:
    candidates = _candidate_usdz_dirs(data_dir)
    for candidate in candidates:
        if any(candidate.glob("*.usdz")):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    pytest.skip(f"No USDZ files found under any configured data path: {searched}")


@pytest.fixture(scope="session")
def usdz_from_data_dir(usdz_data_dir: Path) -> Path:
    usdz_files = sorted(usdz_data_dir.glob("*.usdz"))
    preferred = usdz_data_dir / _PREFERRED_USDZ
    if preferred.is_file():
        return preferred
    return usdz_files[0]
