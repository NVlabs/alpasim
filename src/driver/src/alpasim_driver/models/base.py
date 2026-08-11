# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Abstract base class for trajectory prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, NamedTuple

import numpy as np
import torch
from alpasim_utils.geometry import Pose, Trajectory
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


class CameraFrame(NamedTuple):
    """A single camera frame with timestamp."""

    timestamp_us: int
    # HWC uint8, on the inference device when the driver decoded it there.
    image: np.ndarray | torch.Tensor


CameraImages = dict[str, list[CameraFrame]]
"""Mapping from camera ID to temporal frames (length == context length)."""


@dataclass
class PredictionInput:
    """All inputs needed for a single trajectory prediction.

    The servicer always populates every field. Models read only the
    fields they need.
    """

    camera_images: CameraImages
    command: DriveCommand
    speed: float  # m/s
    acceleration: float  # m/s²
    ego_pose_history: list[Any]  # list[PoseAtTime]
    inference_seed: int  # Session seed plus the zero-based inference count
    # Plan selected in the previous cycle of this session, in the local frame.
    # None before the first plan; only used by models that select between
    # multiple sampled trajectories.
    previous_plan: Trajectory | None
    # Latest route submitted for this session, with waypoints in the rig frame
    # at the route's own timestamp.  None until the first route arrives, and
    # for scenarios that run without a route.
    route: Any | None  # Route


@dataclass
class ModelPrediction:
    """Unified model output.

    Waypoints are full 6-DoF poses in the rig frame.  Models that sample a
    single trajectory report it as the only candidate.
    """

    candidate_positions: np.ndarray  # (K, T, 3) waypoint positions, rig frame
    candidate_rotations: np.ndarray  # (K, T, 3, 3) waypoint rotations, rig frame
    # Index into the candidate axis of the trajectory the vehicle drives.
    selected_index: int = 0
    reasoning_text: str | None = (
        None  # optional text output (e.g. chain-of-causation reasoning)
    )
    # The returned plan in the local frame, fed back as
    # ``PredictionInput.previous_plan``.  Only set by models that select between
    # multiple sampled trajectories.
    selected_plan: Trajectory | None = None
    # Exact spatial and temporal origin used to build the model inputs.  When
    # present, response conversion must use this origin instead of a later pose
    # snapshot or the DriveRequest time.
    model_t0_us: int | None = None
    pose_local_to_rig_t0: Pose | None = None
    waypoint_timestamps_us: np.ndarray | None = None

    @property
    def selected_positions(self) -> np.ndarray:
        """(T, 3) positions of the driven waypoints in the rig frame."""
        return self.candidate_positions[self.selected_index]

    @property
    def selected_rotations(self) -> np.ndarray:
        """(T, 3, 3) rotations of the driven waypoints in the rig frame."""
        return self.candidate_rotations[self.selected_index]

    @classmethod
    def from_planar(
        cls, trajectory_xy: np.ndarray, headings: np.ndarray
    ) -> ModelPrediction:
        """Build a single-candidate prediction from a ground-plane trajectory.

        Args:
            trajectory_xy: (T, 2) x,y offsets in the rig frame.
            headings: (T,) headings in radians in the rig frame.

        Returns:
            A prediction whose waypoints sit in the rig's z=0 plane and rotate
            about the rig's z axis only.
        """
        positions = np.zeros((len(trajectory_xy), 3), dtype=float)
        positions[:, :2] = trajectory_xy

        cos_heading = np.cos(headings)
        sin_heading = np.sin(headings)
        rotations = np.zeros((len(headings), 3, 3), dtype=float)
        rotations[:, 0, 0] = cos_heading
        rotations[:, 0, 1] = -sin_heading
        rotations[:, 1, 0] = sin_heading
        rotations[:, 1, 1] = cos_heading
        rotations[:, 2, 2] = 1.0

        return cls(
            candidate_positions=positions[np.newaxis],
            candidate_rotations=rotations[np.newaxis],
        )


class ModelInputValidationError(ValueError):
    """Raised when model inputs fail a precondition checked before inference."""


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

    # Smallest render ``(height, width)`` the model may be driven with.
    MIN_FRAME_HW: tuple[int, int] | None = None

    @staticmethod
    def _compute_headings_from_trajectory_batch(
        trajectory_xy: np.ndarray,
    ) -> np.ndarray:
        """Compute headings from batched trajectory positions in rig frame.

        For each waypoint, heading is the direction of travel from the previous
        position. For the first waypoint, the previous position is the origin
        (0, 0) since trajectory is ego-relative.

        Args:
            trajectory_xy: (B, N, 2) array of x,y positions in rig frame.

        Returns:
            (B, N) array of heading angles in radians.
        """
        prev = np.zeros_like(trajectory_xy)
        prev[:, 1:, :] = trajectory_xy[:, :-1, :]
        deltas = trajectory_xy - prev
        return np.arctan2(deltas[:, :, 1], deltas[:, :, 0])

    @staticmethod
    def _compute_headings_from_trajectory(trajectory_xy: np.ndarray) -> np.ndarray:
        """Compute headings from trajectory positions in rig frame.

        Single-trajectory wrapper around :meth:`_compute_headings_from_trajectory_batch`.

        Args:
            trajectory_xy: (N, 2) array of x,y positions in rig frame.

        Returns:
            (N,) array of heading angles in radians.
        """
        return BaseTrajectoryModel._compute_headings_from_trajectory_batch(
            trajectory_xy[np.newaxis, ...]
        )[0]

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
        camera_images: CameraImages,
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

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        model_cfg: Any,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "BaseTrajectoryModel":
        """Create a model instance from driver configuration.

        Each model implementation extracts the parameters it needs from the
        generic argument set. This is the standard factory interface used by
        the plugin registry — the driver calls this instead of __init__
        directly, so new models can be added without changing driver code.

        Args:
            model_cfg: ModelConfig dataclass with checkpoint_path, device,
                tokenizer_path, etc.
            device: Torch device for inference.
            camera_ids: List of camera logical IDs in order.
            context_length: Number of temporal frames (None uses model default).
            output_frequency_hz: Trajectory output frequency in Hz.

        Returns:
            Configured model instance.
        """
        ...

    @abstractmethod
    def _encode_command(self, command: DriveCommand) -> Any:
        """Convert canonical DriveCommand to model-specific format.

        Each model implements this to produce its expected encoding:
        - VAM: returns int (RIGHT=0, LEFT=1, STRAIGHT=2)
        - Transfuser: returns int (LEFT=0, FORWARD=1, RIGHT=2, UNDEFINED=3)
        """
        pass

    @abstractmethod
    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Generate trajectory prediction for a single input.

        Args:
            prediction_input: All observation data needed for prediction. See
                :class:`PredictionInput` for field descriptions.

        Returns:
            ModelPrediction with waypoint poses in rig frame coordinates
            (x forward, y left, z up). Models that plan in the ground plane
            build it with :meth:`ModelPrediction.from_planar`, deriving the
            headings with :meth:`_compute_headings_from_trajectory` when they
            do not natively output them.

        Raises:
            ValueError: If camera_images keys don't match expected cameras
                or list lengths are wrong.
        """
        pass

    def predict_batch(
        self, prediction_inputs: list[PredictionInput]
    ) -> list[ModelPrediction]:
        """Generate predictions for a batch of inputs.

        Default implementation calls :meth:`predict` sequentially.
        Models that support GPU batching should override this to stack
        tensors and run a single forward pass.
        """
        return [
            self.predict(prediction_input) for prediction_input in prediction_inputs
        ]

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
