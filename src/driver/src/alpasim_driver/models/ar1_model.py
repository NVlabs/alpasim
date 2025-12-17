# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo-R1 (AR1) wrapper implementing the common interface."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction

logger = logging.getLogger(__name__)

# Camera name to index mapping - must match the order expected by the model
# This is the same mapping used in load_physical_aiavdataset.py
CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}


def _adjust_orientation(quats: np.ndarray) -> np.ndarray:
    """Adjust quaternion signs for interpolation consistency.

    Ensures the dot product between consecutive quaternions is non-negative,
    which is required for proper SLERP interpolation.

    Args:
        quats: Array of quaternions with shape (N, 4).

    Returns:
        Adjusted quaternions with consistent hemisphere.
    """
    N = quats.shape[0]
    signs = np.ones(N)
    for i in range(1, N):
        if np.dot(quats[i - 1], quats[i]) < 0:
            signs[i] = -signs[i - 1]
        else:
            signs[i] = signs[i - 1]
    return quats * signs[:, None]


def _format_trajs(pred_xyz: torch.Tensor) -> np.ndarray:
    """Extract and format trajectory from AR1 output.

    Args:
        pred_xyz: Predicted trajectory tensor with shape
                  [batch=1, num_traj_sets=1, num_traj_samples, T, 3]

    Returns:
        Trajectory array of shape (T, 2) with x, y coordinates.
    """
    # Extract first batch, first trajectory set, first sample
    # Shape: [batch, num_traj_sets, num_traj_samples, T, 3] -> [T, 3]
    traj = pred_xyz[0, 0, 0, :, :].detach().cpu().numpy()

    # Return only x, y coordinates
    return traj[:, :2]


class AR1Model(BaseTrajectoryModel):
    """Alpamayo-R1 wrapper implementing the common interface."""

    # AR1 uses bfloat16 for inference
    DTYPE = torch.bfloat16
    # Number of historical frames for ego trajectory
    NUM_HISTORY_STEPS = 16
    # Time step between history frames (seconds)
    HISTORY_TIME_STEP = 0.1
    # Default context length (number of image frames)
    DEFAULT_CONTEXT_LENGTH = 4
    # Default output frequency (Hz)
    OUTPUT_FREQUENCY_HZ = 10

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
    ):
        """Initialize AR1 model.

        Args:
            checkpoint_path: Path or HuggingFace model ID for AR1 checkpoint.
            device: Torch device for inference.
            camera_ids: List of camera IDs (supports multiple cameras).
            context_length: Number of temporal frames per camera (default 4).
            num_traj_samples: Number of trajectory samples to generate.
            top_p: Top-p sampling parameter for VLM generation.
            temperature: Temperature for VLM sampling.
        """
        logger.info("Loading AR1 checkpoint from %s", checkpoint_path)

        self._model = AlpamayoR1.from_pretrained(checkpoint_path, dtype=self.DTYPE).to(
            device
        )
        self._processor = helper.get_processor(self._model.tokenizer)

        self._device = device
        self._camera_ids = camera_ids
        self._context_length = context_length
        self._num_traj_samples = num_traj_samples
        self._top_p = top_p
        self._temperature = temperature

        # Extract number of output waypoints from model config
        output_shape = self._model.action_space.get_action_space_dims()
        self._pred_num_waypoints, _ = output_shape

        # Verify camera_ids are valid
        missing_cameras = [
            cam_id for cam_id in camera_ids if cam_id not in CAMERA_NAME_TO_INDEX
        ]
        if missing_cameras:
            raise ValueError(f"Cameras {missing_cameras} not found in AR1 model.")

        logger.info(
            "Initialized AR1 with %d cameras, context_length=%d",
            len(camera_ids),
            context_length,
        )

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self.OUTPUT_FREQUENCY_HZ

    def _encode_command(self, command: DriveCommand) -> Any:
        """AR1 reasons about navigation from context, no explicit command encoding."""
        return None

    def _build_ego_history(
        self,
        poses: list[Any],  # List of PoseAtTime
        current_timestamp_us: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ego history tensors from pose data.

        Constructs ego_history_xyz and ego_history_rot tensors in the rig frame relative to the
        current pose (t0). Uses linear interpolation for positions and SLERP for rotations.

        Args:
            poses: List of PoseAtTime messages with timestamp_us and Pose in local frame.
            current_timestamp_us: Current timestamp (t0) in microseconds.

        Returns:
            Tuple of (ego_history_xyz, ego_history_rot) in rig frame:
                - ego_history_xyz: shape (1, 1, num_history_steps, 3) in rig frame
                - ego_history_rot: shape (1, 1, num_history_steps, 3, 3) in rig frame
        """
        # 1. Extract raw pose data and sort by timestamp
        pose_data = sorted(
            [(p.timestamp_us, p.pose) for p in poses], key=lambda x: x[0]
        )
        timestamps_us = np.array([t for t, _ in pose_data], dtype=np.float64)
        ego_history_local_xyz = np.array(
            [[p.vec.x, p.vec.y, p.vec.z] for _, p in pose_data]
        )
        ego_history_local_quats = np.array(
            [[p.quat.x, p.quat.y, p.quat.z, p.quat.w] for _, p in pose_data]
        )

        # 2. Normalize and adjust quaternions for consistent interpolation
        ego_history_local_quats = ego_history_local_quats / np.linalg.norm(
            ego_history_local_quats, axis=1, keepdims=True
        )
        ego_history_local_quats = _adjust_orientation(ego_history_local_quats)

        # 3. Create interpolators
        xyz_interp = interp1d(
            timestamps_us,
            ego_history_local_xyz,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        rot_interp = Slerp(timestamps_us, Rotation.from_quat(ego_history_local_quats))

        # 4. Calculate target history timestamps (going backward from t0)
        history_timestamps_us = np.array(
            [
                current_timestamp_us - int(i * self.HISTORY_TIME_STEP * 1_000_000)
                for i in range(self.NUM_HISTORY_STEPS - 1, -1, -1)
            ],
            dtype=np.float64,
        )

        # Verify we have enough ego pose history to interpolate
        if (
            history_timestamps_us[0] < timestamps_us[0]
            or history_timestamps_us[-1] > timestamps_us[-1]
        ):
            raise ValueError(
                "Ego pose history is insufficient to interpolate. "
                f"Required:  [{history_timestamps_us[0]}, {history_timestamps_us[-1]}], "
                f"Available: [{timestamps_us[0]}, {timestamps_us[-1]}]"
            )

        # 5. Interpolate positions and rotations at target timestamps
        ego_history_local_xyz = xyz_interp(history_timestamps_us)
        ego_history_local_rot = rot_interp(history_timestamps_us)
        ego_history_local_quat = ego_history_local_rot.as_quat()

        # 6. Transform to local frame relative to t0 (last history step)
        ego_t0_local_xyz = ego_history_local_xyz[-1].copy()
        ego_t0_local_quat = ego_history_local_quat[-1].copy()
        ego_t0_local_rot = Rotation.from_quat(ego_t0_local_quat)
        ego_t0_local_rot_inv = ego_t0_local_rot.inv()

        # Transform positions and rotations to rig frame
        ego_history_rig_xyz = ego_t0_local_rot_inv.apply(
            ego_history_local_xyz - ego_t0_local_xyz
        )
        ego_history_rig_rot = (ego_t0_local_rot_inv * ego_history_local_rot).as_matrix()

        # 7. Convert to torch tensors with batch dimensions: (B=1, n_traj_group=1, T, ...)
        ego_history_rig_xyz_tensor = (
            torch.from_numpy(ego_history_rig_xyz).float().unsqueeze(0).unsqueeze(0)
        )
        ego_history_rig_rot_tensor = (
            torch.from_numpy(ego_history_rig_rot).float().unsqueeze(0).unsqueeze(0)
        )

        return ego_history_rig_xyz_tensor, ego_history_rig_rot_tensor

    def _preprocess_images(
        self, camera_images: dict[str, list[tuple[int, np.ndarray]]]
    ) -> torch.Tensor:
        """Preprocess multi-camera images for AR1.

        Args:
            camera_images: Dict mapping camera_id to list of (timestamp_us, image).

        Returns:
            Image tensor in CHW format, shape (N_cameras, num_frames, 3, H, W).
            Images are kept as uint8 [0, 255] - the processor handles normalization.
            Cameras are sorted by their index to match the expected model order.
        """
        frames_list = []

        # Sort camera IDs by their index to ensure consistent ordering
        sorted_camera_ids = sorted(
            self._camera_ids, key=lambda cam_id: CAMERA_NAME_TO_INDEX[cam_id]
        )

        # Process each camera in sorted order
        for cam_id in sorted_camera_ids:
            frames = camera_images[cam_id]

            # Extract images (HWC format)
            images = [img for _, img in frames]

            # Convert to CHW format, keeping uint8 dtype
            # NOTE: Do NOT normalize to [0, 1] - the Qwen3-VL processor expects
            # uint8 images and handles normalization internally. Pre-normalizing
            # causes double normalization which corrupts the visual features.
            camera_frames = []
            for img in images:
                # Convert HWC uint8 RGB to CHW uint8 tensor [0, 255]
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # stays uint8
                camera_frames.append(img_tensor)

            # Stack frames for this camera: (num_frames, C, H, W)
            camera_tensor = torch.stack(camera_frames, dim=0)
            frames_list.append(camera_tensor)

        # Stack all cameras: (N_cameras, num_frames, C, H, W)
        all_frames = torch.stack(frames_list, dim=0)

        return all_frames

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
                (timestamp_us, image) tuples. List length must equal
                context_length.
            command: Canonical navigation command (unused by AR1).
            speed: Current vehicle speed in m/s (unused by AR1).
            acceleration: Current longitudinal acceleration (unused by AR1).
            ego_pose_at_time_history_local: Optional list of PoseAtTime for building ego history.
                PoseAtTime contains pairs of (timestamp_us, Pose) where Pose is 3D position and
                orientation in local frame.

        Returns:
            ModelPrediction with trajectory in rig frame.
        """
        self._validate_cameras(camera_images)

        # Check context length
        for cam_id in self._camera_ids:
            if len(camera_images[cam_id]) != self._context_length:
                logger.warning(
                    "AR1 expects %d frames per camera, got %d for %s",
                    self._context_length,
                    len(camera_images[cam_id]),
                    cam_id,
                )
                return ModelPrediction(
                    trajectory_xy=np.zeros((self._pred_num_waypoints, 2)),
                    headings=np.zeros(self._pred_num_waypoints),
                )

        # Check ego history length
        if (
            ego_pose_at_time_history_local is None
            or len(ego_pose_at_time_history_local) < self.NUM_HISTORY_STEPS
        ):
            num_poses = (
                0
                if ego_pose_at_time_history_local is None
                else len(ego_pose_at_time_history_local)
            )
            logger.warning(
                "Not enough pose history: %d < %d. Using zero history, output will be invalid.",
                num_poses,
                self.NUM_HISTORY_STEPS,
            )
            return ModelPrediction(
                trajectory_xy=np.zeros((self._pred_num_waypoints, 2)),
                headings=np.zeros(self._pred_num_waypoints),
            )

        # Get current timestamp from the latest frame
        latest_timestamp = max(
            max(ts for ts, _ in frames) for frames in camera_images.values()
        )
        ego_history_xyz, ego_history_rot = self._build_ego_history(
            ego_pose_at_time_history_local, latest_timestamp
        )

        # Preprocess images
        image_frames = self._preprocess_images(camera_images)

        # Create chat message using AR1's helper
        # Flatten camera and temporal dimensions for the message
        messages = helper.create_message(image_frames.flatten(0, 1))

        # Apply chat template
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Prepare model inputs
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }

        # Move to device
        model_inputs = helper.to_device(model_inputs, self._device)

        # Run inference with autocast
        with torch.no_grad():
            with torch.autocast(str(self._device.type), dtype=self.DTYPE):
                pred_xyz, pred_rot, extra = (
                    self._model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=self._top_p,
                        temperature=self._temperature,
                        num_traj_samples=self._num_traj_samples,
                        return_extra=True,
                    )
                )

        # Extract trajectory (x, y coordinates)
        trajectory_xy = _format_trajs(pred_xyz)

        # Compute headings from trajectory
        headings = self._compute_headings_from_trajectory(trajectory_xy)

        # Log reasoning trace if available
        if "cot" in extra and len(extra["cot"]) > 0:
            reasoning_text = str(extra["cot"][0, 0])
            logger.info("AR1 Chain-of-Causation: %s", reasoning_text)
        else:
            reasoning_text = None

        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning_text,
        )
