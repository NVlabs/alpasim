# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Which renders the driver accepts from the renderer."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from alpasim_grpc.v0.egodriver_pb2 import RolloutCameraImage
from PIL import Image

from ..main import EgoDriverService, Session
from ..models.base import BaseTrajectoryModel
from ..schema import DriverConfig

CAMERA_ID = "camera_front_wide_120fov"


class _Session:
    debug_scene_id = "test-scene"

    def __init__(self) -> None:
        self.frame_caches = {CAMERA_ID: object()}
        self.frames: list[np.ndarray] = []

    def rectify_image(self, logical_id: str, image: Image.Image) -> Image.Image:
        return image

    def add_image(self, logical_id: str, image: np.ndarray, timestamp_us: int) -> None:
        self.frames.append(image)


def _service(minimum: tuple[int, int] | None) -> tuple[EgoDriverService, _Session]:
    service = EgoDriverService.__new__(EgoDriverService)
    service._cfg = cast(
        DriverConfig,
        SimpleNamespace(
            plot_debug_images=False,
            output_dir="",
            model=SimpleNamespace(image_decode_device="cpu"),
        ),
    )
    service._model = cast(BaseTrajectoryModel, SimpleNamespace(MIN_FRAME_HW=minimum))
    session = _Session()
    service._sessions = {"session": cast(Session, session)}
    return service, session


def _request(width: int, height: int) -> RolloutCameraImage:
    buffer = BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="JPEG")
    return RolloutCameraImage(
        session_uuid="session",
        camera_image=RolloutCameraImage.CameraImage(
            logical_id=CAMERA_ID, frame_end_us=1, image_bytes=buffer.getvalue()
        ),
    )


def test_a_render_below_the_minimum_is_refused() -> None:
    """320x512 already fits Alpamayo's pixel budget, so it would pass silently."""
    service, _ = _service((320, 576))

    with pytest.raises(ValueError, match="320x512"):
        service.submit_image_observation(_request(512, 320), None)


@pytest.mark.parametrize("minimum", [(320, 576), None])
def test_renders_at_or_above_the_minimum_are_accepted(
    minimum: tuple[int, int] | None,
) -> None:
    service, session = _service(minimum)

    service.submit_image_observation(_request(576, 320), None)

    assert [frame.shape for frame in session.frames] == [(320, 576, 3)]
