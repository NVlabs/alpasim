# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import threading

import numpy as np
from alpasim_driver import main as driver_main
from alpasim_driver.main import Session
from alpasim_driver.schema import RectificationTargetConfig
from alpasim_grpc.v0 import sensorsim_pb2
from PIL import Image


def _target() -> RectificationTargetConfig:
    return RectificationTargetConfig(
        focal_length=(10.0, 10.0),
        principal_point=(3.0, 2.0),
        resolution_hw=(4, 6),
    )


def test_unconfigured_camera_bypasses_rectification() -> None:
    session = Session(
        uuid="session",
        seed=0,
        debug_scene_id="scene",
        frame_caches={},
    )
    image = Image.new("RGB", (6, 4))

    assert session.rectify_image("camera", image) is image


def test_configured_camera_builds_and_reuses_rectifier(monkeypatch) -> None:
    camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(logical_id="camera")
    camera.intrinsics.resolution_h = 4
    camera.intrinsics.resolution_w = 6
    camera.intrinsics.ftheta_param.angle_to_pixeldist_poly.extend([0.0, 1.0])

    session = Session(
        uuid="session",
        seed=0,
        debug_scene_id="scene",
        frame_caches={},
        rectification_cfg={"camera": _target()},
        rectification_camera_specs={"camera": camera},
        rectifier_locks={"camera": threading.Lock()},
    )
    builds: list[tuple[str, tuple[int, int]]] = []

    class _Rectifier:
        def rectify(self, image: np.ndarray) -> np.ndarray:
            return np.full((4, 6, 3), 7, dtype=np.uint8)

    def _build_rectifier(
        camera_proto: sensorsim_pb2.AvailableCamerasReturn.AvailableCamera,
        target_cfg: RectificationTargetConfig,
        source_resolution_hw: tuple[int, int],
    ) -> _Rectifier:
        builds.append((camera_proto.logical_id, source_resolution_hw))
        return _Rectifier()

    monkeypatch.setattr(
        driver_main,
        "build_ftheta_rectifier_for_resolution",
        _build_rectifier,
    )

    first = session.rectify_image("camera", Image.new("RGB", (6, 4)))
    second = session.rectify_image("camera", Image.new("RGB", (6, 4)))

    assert builds == [("camera", (4, 6))]
    assert np.array_equal(np.asarray(first), np.full((4, 6, 3), 7, dtype=np.uint8))
    assert np.array_equal(np.asarray(second), np.asarray(first))
