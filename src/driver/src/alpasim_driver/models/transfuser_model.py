# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Transfuser model wrapper implementing the common interface."""

from __future__ import annotations

import logging

import numpy as np
import torch

from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction, PredictionInput
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

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Generate trajectory prediction.

        Delegates to :meth:`predict_batch` so there is one inference path.
        """
        return self.predict_batch([prediction_input])[0]

    def predict_batch(
        self, prediction_inputs: list[PredictionInput]
    ) -> list[ModelPrediction]:
        """Generate trajectory predictions for a batch of inputs.

        All camera images are preprocessed and concatenated per sample, then
        stacked into a single batch tensor for one forward pass.

        Transfuser uses camera images, command, speed, and acceleration.
        Ego pose history is unused.

        Args:
            prediction_inputs: List of prediction inputs, one per concurrent
                session.

        Returns:
            List of ModelPrediction with trajectories in rig frame
            coordinates. CARLA uses Y+ right, rig frame uses Y+ left,
            so Y axis is inverted.
        """
        if not prediction_inputs:
            return []

        batch_size = len(prediction_inputs)

        # --- Validate and collect per-sample tensors ---
        rgb_tensors: list[torch.Tensor] = []
        encoded_commands: list[int] = []
        speeds: list[float] = []
        accelerations: list[float] = []

        for inp in prediction_inputs:
            self._validate_cameras(inp.camera_images)

            # Validate frame count (Transfuser uses single frame)
            for cam_id, frames in inp.camera_images.items():
                if len(frames) != 1:
                    raise ValueError(
                        f"Transfuser expects 1 frame per camera, "
                        f"got {len(frames)} for {cam_id}"
                    )

            # Extract single frame from each camera
            current_images = {
                cam_id: frames[0][1] for cam_id, frames in inp.camera_images.items()
            }

            # Resize each camera and concatenate horizontally
            concatenated = self._concatenate_cameras(current_images)

            # Convert to tensor: HWC uint8 -> CHW uint8
            rgb = torch.from_numpy(concatenated).permute(2, 0, 1)
            rgb_tensors.append(rgb)

            encoded_commands.append(self._encode_command(inp.command))
            speeds.append(inp.speed)
            accelerations.append(inp.acceleration)

        # --- Stack into batch tensors ---
        # RGB: (B, 3, H, W)
        batched_rgb = torch.stack(rgb_tensors, dim=0).to(self._device)

        # Command: (B, 4) one-hot encoded float
        batched_command = torch.nn.functional.one_hot(
            torch.tensor(encoded_commands, device=self._device, dtype=torch.long),
            num_classes=4,
        ).float()

        # Speed and acceleration: (B,)
        float_dtype = self._config.torch_float_type
        batched_speed = torch.tensor(speeds, device=self._device, dtype=float_dtype)
        batched_acceleration = torch.tensor(
            accelerations, device=self._device, dtype=float_dtype
        )

        data = {
            "rgb": batched_rgb,  # (B, 3, H, W) uint8, model handles normalization
            "command": batched_command,  # (B, 4) one-hot encoded float
            "speed": batched_speed,  # (B,)
            "acceleration": batched_acceleration,  # (B,)
        }

        # --- Single forward pass ---
        with torch.no_grad():
            prediction = self._model(data)

        # --- Convert to rig frame and compute headings on batch ---
        # CARLA: X+ forward, Y+ right; Rig: X+ forward, Y+ left
        waypoints_batch = prediction.pred_future_waypoints.cpu().numpy()  # (B, N, 2)
        waypoints_batch[:, :, 1] *= -1  # Invert Y axis

        if prediction.pred_headings is not None:
            headings_batch = prediction.pred_headings.cpu().numpy()  # (B, N)
            headings_batch = headings_batch * -1  # Invert for coordinate transform
        else:
            headings_batch = self._compute_headings_from_trajectory_batch(
                waypoints_batch
            )

        # --- Unpack per-sample results ---
        results: list[ModelPrediction] = []
        for i in range(batch_size):
            results.append(
                ModelPrediction(
                    trajectory_xy=waypoints_batch[i].copy(),
                    headings=headings_batch[i].copy(),
                )
            )
        return results
