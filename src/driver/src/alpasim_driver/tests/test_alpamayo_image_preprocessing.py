# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""How camera frames reach the Alpamayo vision processor.

Uses a stub model and a stub processor so the hand-off can be tested without a
checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from alpasim_driver.models.alpamayo_base import CAMERA_NAME_TO_INDEX, AlpamayoBaseModel
from alpasim_driver.models.base import DriveCommand, PredictionInput
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Vec3

CAMERA_IDS = ["camera_front_wide_120fov", "camera_cross_left_120fov"]
NUM_WAYPOINTS = 20
SPEED_M_S = 10.0
CONTEXT_LENGTH = 2
IMAGE_SHAPE = (4, 6, 3)
LATEST_US = 2_000_000


class _StubAlpamayo(AlpamayoBaseModel):
    """Alpamayo variant recording the frames and processor kwargs it produces."""

    def __init__(self) -> None:
        self.frames: torch.Tensor | None = None
        self.processor_kwargs: dict = {}
        self._init_common(
            model=SimpleNamespace(
                action_space=SimpleNamespace(
                    get_action_space_dims=lambda: (NUM_WAYPOINTS, 3), dt=0.1
                ),
                sample_trajectories_from_data_with_vlm_rollout=self._sample,
            ),
            processor=SimpleNamespace(apply_chat_template=self._apply_chat_template),
            helper_module=SimpleNamespace(to_device=lambda inputs, device: inputs),
            device=torch.device("cpu"),
            camera_ids=CAMERA_IDS,
            context_length=CONTEXT_LENGTH,
        )

    def _apply_chat_template(self, messages: list, **kwargs) -> dict:
        self.processor_kwargs = kwargs
        return {}

    def _create_chat_message(
        self, image_frames: torch.Tensor, nav_text: str | None
    ) -> list:
        self.frames = image_frames
        return []

    def _sample(self, **kwargs) -> tuple:
        positions = torch.zeros(1, 1, 1, NUM_WAYPOINTS, 3)
        positions[..., 0] = torch.arange(1, NUM_WAYPOINTS + 1)
        rotations = torch.eye(3).expand(1, 1, 1, NUM_WAYPOINTS, 3, 3)
        return positions, rotations, {}


def _poses(latest_us: int) -> list[PoseAtTime]:
    return [
        PoseAtTime(
            timestamp_us=timestamp_us,
            pose=Pose(
                vec=Vec3(x=SPEED_M_S * timestamp_us / 1e6, y=0.0, z=0.0),
                quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
        )
        for timestamp_us in range(0, latest_us + 1, 100_000)
    ]


def _prediction_input(images: dict) -> PredictionInput:
    return PredictionInput(
        camera_images=images,
        command=DriveCommand.STRAIGHT,
        speed=SPEED_M_S,
        acceleration=0.0,
        ego_pose_history=_poses(LATEST_US),
        inference_seed=0,
        previous_plan=None,
        route=None,
    )


def _images(on_device: bool = False) -> dict:
    """One distinct HWC uint8 frame per camera and context slot.

    Device frames are the same layout, they just live on the inference device,
    so the model handles both with one expression.
    """
    rng = np.random.default_rng(0)
    frames = {}
    for cam_id in CAMERA_IDS:
        frames[cam_id] = []
        for frame in range(CONTEXT_LENGTH):
            image = rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8)
            frames[cam_id].append(
                (
                    LATEST_US - (CONTEXT_LENGTH - 1 - frame) * 100_000,
                    torch.from_numpy(image) if on_device else image,
                )
            )
    return frames


def _expected(images: dict) -> torch.Tensor:
    """The frames as CHW, stacked camera-major in the order the model sorts."""
    return torch.stack(
        [
            torch.stack(
                [torch.as_tensor(image).permute(2, 0, 1) for _, image in images[cam_id]]
            )
            for cam_id in sorted(CAMERA_IDS, key=lambda cam: CAMERA_NAME_TO_INDEX[cam])
        ]
    )


@pytest.mark.parametrize("on_device", [False, True])
def test_frames_reach_the_processor_as_decoded(on_device: bool) -> None:
    """A host frame and a device frame arrive the same way: uint8 ``[0, 255]``.

    The processor rescales them itself, so the driver only stacks them.
    """
    images = _images(on_device=on_device)
    model = _StubAlpamayo()

    model.predict(_prediction_input(images))

    assert model.frames is not None
    assert model.frames.dtype == torch.uint8
    assert model.frames.shape == (len(CAMERA_IDS), CONTEXT_LENGTH, 3, *IMAGE_SHAPE[:2])
    torch.testing.assert_close(model.frames, _expected(images))
