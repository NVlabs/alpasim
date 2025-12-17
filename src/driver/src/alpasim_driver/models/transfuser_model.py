# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Transfuser model wrapper implementing the common interface."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction
from .transfuser_impl import load_tf

logger = logging.getLogger(__name__)


class TransfuserModel(BaseTrajectoryModel):
    """Transfuser wrapper implementing the common interface.

    Transfuser is a single-frame model that uses multiple cameras
    concatenated horizontally for inference.
    """

    # Transfuser with 4 cameras (NAVSIM configuration)
    NUM_CAMERAS = 4
    # Expected per-camera dimensions (from NAVSIM config)
    EXPECTED_HEIGHT = 270
    EXPECTED_WIDTH_PER_CAM = 480

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
    ):
        """Initialize Transfuser model.

        Args:
            checkpoint_path: Path to model checkpoint (.pth file).
                The config.json must be in the same directory.
            device: Torch device for inference.
            camera_ids: List of camera IDs in order for horizontal
                concatenation. Must be exactly 4 cameras.
        """
        if len(camera_ids) != self.NUM_CAMERAS:
            raise ValueError(
                f"Transfuser requires exactly {self.NUM_CAMERAS} cameras, "
                f"got {len(camera_ids)}"
            )

        self._model = load_tf(checkpoint_path, device)
        self._config = self._model.config
        self._device = device
        self._camera_ids = camera_ids

        # Per-camera dimensions (hardcoded for this model variant)
        self._per_cam_height = self.EXPECTED_HEIGHT
        self._per_cam_width = self.EXPECTED_WIDTH_PER_CAM

        logger.info(
            "Loaded Transfuser model from %s with %d cameras (%dx%d each)",
            checkpoint_path,
            len(camera_ids),
            self._per_cam_width,
            self._per_cam_height,
        )

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return 1  # Single frame model

    @property
    def output_frequency_hz(self) -> int:
        return 2  # NAVSIM config: waypoints_spacing=10 at 20fps → 2Hz

    def _concatenate_cameras(self, camera_images: dict[str, np.ndarray]) -> np.ndarray:
        """Resize each camera and concatenate horizontally in camera_ids order.

        Args:
            camera_images: Dict mapping camera_id to HWC uint8 image.

        Returns:
            Concatenated image as HWC uint8 array.
        """
        resized_images = []
        for cam_id in self._camera_ids:
            resized = self._resize_and_center_crop(
                camera_images[cam_id], self._per_cam_height, self._per_cam_width
            )
            resized_images.append(resized)
        return np.concatenate(resized_images, axis=1)  # Horizontal concat

    def _encode_command(self, command: DriveCommand) -> int:
        """Convert DriveCommand to Transfuser format.

        DriveCommand:       LEFT=0, STRAIGHT=1, RIGHT=2, UNKNOWN=3
        Transfuser Command: LEFT=0, FORWARD=1, RIGHT=2, UNDEFINED=3
        """
        COMMAND_MAP = {
            DriveCommand.LEFT: 0,  # Transfuser LEFT
            DriveCommand.STRAIGHT: 1,  # Transfuser FORWARD
            DriveCommand.RIGHT: 2,  # Transfuser RIGHT
            DriveCommand.UNKNOWN: 3,  # Transfuser UNDEFINED
        }
        return COMMAND_MAP[command]

    def predict(
        self,
        camera_images: dict[str, list[tuple[int, np.ndarray]]],
        command: DriveCommand,
        speed: float,
        acceleration: float,
        ego_pose_at_time_history_local: list[Any] | None = None,
    ) -> ModelPrediction:
        """Generate trajectory prediction.

        Args:
            camera_images: Dict mapping camera_id to list of
                (timestamp_us, image) tuples. For Transfuser,
                list length must be 1 (single frame).
            command: Canonical navigation command.
            speed: Current vehicle speed in m/s.
            acceleration: Current longitudinal acceleration in m/s².
            ego_pose_at_time_history_local: Optional list of PoseAtTime for building ego history.
                PoseAtTime contains pairs of (timestamp_us, Pose) where Pose is 3D position and
                orientation in local frame.

        Returns:
            ModelPrediction with trajectory in rig frame coordinates.
            CARLA uses Y+ right, rig frame uses Y+ left, so Y axis
            is inverted.
        """
        del ego_pose_at_time_history_local
        self._validate_cameras(camera_images)

        # Validate frame count (Transfuser uses single frame)
        for cam_id, frames in camera_images.items():
            if len(frames) != 1:
                raise ValueError(
                    f"Transfuser expects 1 frame per camera, "
                    f"got {len(frames)} for {cam_id}"
                )

        # Extract single frame from each camera
        current_images = {
            cam_id: frames[0][1] for cam_id, frames in camera_images.items()
        }

        # Resize each camera and concatenate horizontally
        concatenated = self._concatenate_cameras(current_images)

        # Convert to tensor: HWC uint8 -> CHW uint8
        # NOTE: Model internally converts to float and applies ImageNet normalization
        rgb = (
            torch.from_numpy(concatenated)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )

        # Encode command using model-specific encoding
        encoded_command = self._encode_command(command)

        # Prepare data dict as expected by Transfuser Model.forward()
        # Command must be one-hot encoded as a float tensor of shape (batch, 4)
        command_one_hot = torch.nn.functional.one_hot(
            torch.tensor([encoded_command], device=self._device, dtype=torch.long),
            num_classes=4,
        ).float()

        data = {
            "rgb": rgb,  # (1, 3, H, W) uint8, model handles normalization
            "command": command_one_hot,  # (1, 4) one-hot encoded float
            "speed": torch.tensor(
                [speed], device=self._device, dtype=self._config.torch_float_type
            ),
            "acceleration": torch.tensor(
                [acceleration],
                device=self._device,
                dtype=self._config.torch_float_type,
            ),
        }

        with torch.no_grad():
            prediction = self._model(data)

        # Extract waypoints and convert coordinates
        # Model was trained in CARLA coordinate system, convert to NavSim/NuPlan/rig frame
        # CARLA: X+ forward, Y+ right; Rig: X+ forward, Y+ left
        waypoints = prediction.pred_future_waypoints[0].cpu().numpy()  # (N, 2)
        waypoints[:, 1] *= -1  # Invert Y axis

        # Extract headings if available, otherwise compute from trajectory
        if prediction.pred_headings is not None:
            headings = prediction.pred_headings[0].cpu().numpy()  # (N,)
            headings *= -1  # Invert heading angles for coordinate transform
        else:
            headings = self._compute_headings_from_trajectory(waypoints)

        return ModelPrediction(trajectory_xy=waypoints, headings=headings)
