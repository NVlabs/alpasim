# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Abstract base class for trajectory prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from PIL import Image


class DriveCommand(IntEnum):
    """Canonical driving command representation.

    This is the "semantic" command that the driver determines from
    route/navigation data. Each model converts this to its own format
    via _encode_command().
    """

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    UNKNOWN = 3


@dataclass
class ModelPrediction:
    """Unified model output."""

    trajectory_xy: np.ndarray  # (T, 2) x,y offsets in rig frame
    headings: np.ndarray  # (T,) headings in radians (rig frame)
    reasoning_text: str | None = (
        None  # optional text output (e.g. chain-of-causation reasoning)
    )


class BaseTrajectoryModel(ABC):
    """Abstract base class for trajectory prediction models.

    Models receive raw camera images and handle all preprocessing internally:
    - Validate received images (camera names, dimensions)
    - Resize/crop to model-specific dimensions
    - Concatenate multi-camera images if needed
    - Apply model-specific normalization (NeuroNCAP, ImageNet, etc.)

    Each model implements _encode_command() to convert the canonical DriveCommand
    to its own format (VAM vs Transfuser have different encodings).
    """

    @staticmethod
    def _compute_headings_from_trajectory(trajectory_xy: np.ndarray) -> np.ndarray:
        """Compute headings from trajectory positions in rig frame.

        Computes heading for each waypoint based on the direction of travel
        from the previous position. For the first waypoint, the previous
        position is the origin (0, 0) since trajectory is ego-relative.

        Args:
            trajectory_xy: (N, 2) array of x,y positions in rig frame.

        Returns:
            (N,) array of heading angles in radians.
        """
        previous_positions = np.vstack(([0.0, 0.0], trajectory_xy[:-1]))
        deltas = trajectory_xy - previous_positions
        return np.arctan2(deltas[:, 1], deltas[:, 0])

    @staticmethod
    def _resize_and_center_crop(
        image: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """Resize image to target height and center-crop to target width.

        Args:
            image: HWC uint8 numpy array
            target_height: Target height in pixels
            target_width: Target width in pixels

        Returns:
            Resized and cropped image as HWC uint8 numpy array.

        Raises:
            ValueError: If image is too narrow after resize to reach target width.
        """
        h, w = image.shape[:2]
        if h == target_height and w == target_width:
            return image

        # Resize maintaining aspect ratio based on height
        pil_img = Image.fromarray(image)
        scale = target_height / h
        new_w = int(w * scale)
        pil_img = pil_img.resize((new_w, target_height), Image.Resampling.BILINEAR)

        # Center crop width if needed
        if new_w > target_width:
            left = (new_w - target_width) // 2
            pil_img = pil_img.crop((left, 0, left + target_width, target_height))
        elif new_w < target_width:
            raise ValueError(
                f"Image width {new_w} too small after resize, need {target_width}"
            )

        return np.array(pil_img)

    def _validate_cameras(
        self,
        camera_images: dict[str, list[tuple[int, np.ndarray]]],
    ) -> None:
        """Validate received camera images match expected configuration.

        Args:
            camera_images: Dictionary from predict() - only keys are checked.

        Raises:
            ValueError: If camera names don't match expected camera_ids.
        """
        received = set(camera_images.keys())
        expected = set(self.camera_ids)
        if received != expected:
            raise ValueError(
                f"{self.__class__.__name__} expects cameras {expected}, got {received}"
            )

    @abstractmethod
    def _encode_command(self, command: DriveCommand) -> Any:
        """Convert canonical DriveCommand to model-specific format.

        Each model implements this to produce its expected encoding:
        - VAM: returns int (RIGHT=0, LEFT=1, STRAIGHT=2)
        - Transfuser: returns int (LEFT=0, FORWARD=1, RIGHT=2, UNDEFINED=3)
        """
        pass

    @abstractmethod
    def predict(
        self,
        camera_images: dict[
            str, list[tuple[int, np.ndarray]]
        ],  # camera_name -> [(timestamp_us, image), ...]
        command: DriveCommand,  # Canonical navigation command
        speed: float,  # Current speed m/s
        acceleration: float,  # Current longitudinal acceleration m/s²
        ego_pose_at_time_history_local: list | None = None,
    ) -> ModelPrediction:
        """Generate trajectory prediction.

        Args:
            camera_images: Dictionary mapping camera logical ID to list of
                (timestamp_us, image) tuples. List length equals context_length
                (e.g., 8 for VAM, 1 for Transfuser). Images are HWC uint8 RGB
                at whatever resolution the service received. Model validates
                camera names, checks list length, and handles resize/preprocessing
                internally. Timestamps allow caching of preprocessed frames.
            command: Canonical DriveCommand enum value.
                Model encodes this internally via _encode_command().
            speed: Current vehicle speed in m/s (magnitude of velocity).
            acceleration: Current longitudinal acceleration in m/s²
            ego_pose_at_time_history_local: Optional list of PoseAtTime for building ego history.
                PoseAtTime contains pairs of (timestamp_us, Pose) where Pose is 3D position and
                orientation in local frame.

        Returns:
            ModelPrediction with trajectory and headings in rig frame
            coordinates (x forward, y left). Headings must always be
            provided - use _compute_headings_from_trajectory() if the
            model doesn't natively output headings.

        Raises:
            ValueError: If camera_images keys don't match expected cameras
                or list lengths are wrong.
        """
        pass

    @property
    @abstractmethod
    def camera_ids(self) -> list[str]:
        """List of expected camera logical IDs in order."""
        pass

    @property
    def num_cameras(self) -> int:
        """Number of cameras, derived from camera_ids."""
        return len(self.camera_ids)

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Number of temporal frames required (1 for single-frame models)."""
        pass

    @property
    @abstractmethod
    def output_frequency_hz(self) -> int:
        """Output trajectory frequency in Hz (e.g., 2 for 0.5s between waypoints)."""
        pass
