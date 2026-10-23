# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Writes drivr_defish.npz from drivr's own Defisher.

``defish.py`` is loaded by path so drivr's package dependencies are not
needed.  Run from this directory:
    uv run --no-project --with numpy --with structlog \
        --with opencv-python-headless python make_drivr_fixtures.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from images import smooth_image

DRIVR = Path.home() / "drivr"
CALIBRATION = DRIVR / "config" / "camera_calibration.json"
# Every STRIDE-th output pixel is stored to keep the fixture small.
STRIDE = 8


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "defish", DRIVR / "src" / "drivr" / "io" / "defish.py"
    )
    defish = importlib.util.module_from_spec(spec)
    sys.modules["defish"] = defish
    spec.loader.exec_module(defish)

    calib = defish.FisheyeCalibration.from_json(CALIBRATION)
    defisher = defish.Defisher(calib, calib.native_width, calib.native_height)
    image = smooth_image(calib.native_height, calib.native_width)
    np.savez_compressed(
        Path(__file__).with_name("drivr_defish.npz"),
        stride=np.array(STRIDE),
        output=defisher.apply(image)[::STRIDE, ::STRIDE],
    )


if __name__ == "__main__":
    main()
