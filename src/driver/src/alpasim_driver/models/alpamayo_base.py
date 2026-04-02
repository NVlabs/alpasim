# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared base class for the Alpamayo model family (Alpamayo 1, 1.5, etc.)."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from alpasim_grpc.v0.common_pb2 import PoseAtTime
from alpasim_utils.geometry import Trajectory

from ..schema import ModelConfig
from .base import (
    BaseTrajectoryModel,
    CameraImages,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)

logger = logging.getLogger(__name__)

# Camera name to index mapping — must match the order expected by the model.
# Shared across all Alpamayo variants (same sensor rig).
# Source: alpamayo1_5.load_physical_aiavdataset (camera_name_to_index local var).
# Defined here because upstream has no importable constant for this mapping.
CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}


def _format_trajs(pred_xyz: torch.Tensor) -> np.ndarray:
    """Extract and format trajectory from Alpamayo output.

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


def so3_to_yaw_torch(rot: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from SO(3) rotation matrices.

    Uses the standard ZYX Euler decomposition: yaw = atan2(R[1,0], R[0,0]).

    Args:
        rot: Rotation matrices of shape (..., 3, 3).

    Returns:
        Yaw angles in radians as a torch tensor, shape (...).
    """
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def build_ego_history(
    poses: list[PoseAtTime],
    current_timestamp_us: int,
    num_history_steps: int = 16,
    history_time_step: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ego history tensors from pose data.

    Constructs ego_history_xyz and ego_history_rot tensors in the rig frame
    relative to the current pose (t0).

    Args:
        poses: List of PoseAtTime messages with timestamp_us and Pose in local
            frame.  Each pose must have .timestamp_us (int),
            .pose.vec.{x,y,z}, .pose.quat.{x,y,z,w}.  Quaternions must be
            unit-length (validated by the Trajectory constructor).
        current_timestamp_us: Current timestamp (t0) in microseconds.
        num_history_steps: Number of history steps to produce.
        history_time_step: Time step between history frames in seconds.

    Returns:
        Tuple of (ego_history_xyz, ego_history_rot) in rig frame:
            - ego_history_xyz: shape (1, 1, num_history_steps, 3)
            - ego_history_rot: shape (1, 1, num_history_steps, 3, 3)
    """
    # 1. Extract raw pose data and sort by timestamp
    pose_data = sorted([(p.timestamp_us, p.pose) for p in poses], key=lambda x: x[0])
    timestamps_us = np.array([t for t, _ in pose_data], dtype=np.uint64)
    ego_history_xyz_in_local = np.array(
        [[p.vec.x, p.vec.y, p.vec.z] for _, p in pose_data], dtype=np.float32
    )
    ego_history_quat_rig_to_local = np.array(
        [[p.quat.x, p.quat.y, p.quat.z, p.quat.w] for _, p in pose_data],
        dtype=np.float32,
    )

    # 2. Build Trajectory and interpolate at target history timestamps
    trajectory_of_rig_in_local = Trajectory(
        timestamps_us, ego_history_xyz_in_local, ego_history_quat_rig_to_local
    )

    history_timestamps_us = np.array(
        [
            current_timestamp_us - int(i * history_time_step * 1_000_000)
            for i in range(num_history_steps - 1, -1, -1)
        ],
        dtype=np.uint64,
    )

    interpolated_rig_in_local = trajectory_of_rig_in_local.interpolate(
        history_timestamps_us
    )

    # 3. Transform to rig frame relative to t0 (last history step)
    pose_local_to_rig_t0 = interpolated_rig_in_local.last_pose
    interpolated_rig_in_rig_t0 = interpolated_rig_in_local.transform(
        pose_local_to_rig_t0.inverse()
    )

    # 4. Extract positions and rotation matrices as numpy arrays
    ego_history_xyz_in_rig_t0 = interpolated_rig_in_rig_t0.positions  # (N, 3)
    ego_history_rot_rig_to_rig_t0 = (
        interpolated_rig_in_rig_t0.rotation_matrices()  # (N, 3, 3)
    )

    # 5. Convert to torch tensors with batch dimensions: (B=1, n_traj_group=1, T, ...)
    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_in_rig_t0).float().unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_rig_to_rig_t0)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return ego_history_xyz_tensor, ego_history_rot_tensor


class AlpamayoBaseModel(BaseTrajectoryModel):
    """Shared logic for the Alpamayo model family.

    Subclasses must implement :meth:`_create_chat_message` and provide their
    own ``__init__`` that loads the model-specific checkpoint, then calls
    :meth:`_init_common`.
    """

    # Alpamayo uses bfloat16 for inference
    DTYPE: torch.dtype = torch.bfloat16
    # Number of historical frames for ego trajectory
    NUM_HISTORY_STEPS: int = 16
    # Time step between history frames (seconds)
    HISTORY_TIME_STEP: float = 0.1
    # Default context length (number of image frames)
    DEFAULT_CONTEXT_LENGTH: int = 4
    # Default output frequency (Hz)
    OUTPUT_FREQUENCY_HZ: int = 10
    # Expected input frame rate per camera
    IMAGE_INPUT_FREQUENCY_HZ: int = 10

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_common(
        self,
        *,
        model: Any,
        processor: Any,
        helper_module: Any,
        device: torch.device,
        camera_ids: list[str],
        context_length: int,
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
    ) -> None:
        """Shared initialisation called by every subclass ``__init__``."""
        self._model = model
        self._processor = processor
        self._helper = helper_module
        self._device = device
        self._camera_ids = camera_ids
        self._context_length = context_length
        self._num_traj_samples = num_traj_samples
        self._top_p = top_p
        self._temperature = temperature

        output_shape = self._model.action_space.get_action_space_dims()
        self._pred_num_waypoints, _ = output_shape

        missing_cameras = [
            cam_id for cam_id in camera_ids if cam_id not in CAMERA_NAME_TO_INDEX
        ]
        if missing_cameras:
            raise ValueError(
                f"Cameras {missing_cameras} not found in Alpamayo camera mapping."
            )

        logger.info(
            "Initialised %s with %d cameras, context_length=%d",
            self.__class__.__name__,
            len(camera_ids),
            context_length,
        )

    # ------------------------------------------------------------------
    # BaseTrajectoryModel interface
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "AlpamayoBaseModel":
        """Create an instance from the driver configuration.

        Subclasses may override to extract additional model-specific config
        values (e.g. ``use_classifier_free_guidance_nav``).
        """
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length or cls.DEFAULT_CONTEXT_LENGTH,
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
        """Alpamayo models reason about navigation from context."""
        return None

    # ------------------------------------------------------------------
    # Template-method hooks (override in subclasses as needed)
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_chat_message(self, image_frames: torch.Tensor) -> list:
        """Build the chat-message list from preprocessed image frames.

        Alpamayo 1 passes flattened frames; Alpamayo 1.5 additionally passes camera indices.
        """
        ...

    def _run_inference(
        self, model_inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Run trajectory inference.  Override to use an alternative method
        (e.g. classifier-free guidance navigation in A1.5)."""
        return self._model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=self._top_p,
            temperature=self._temperature,
            num_traj_samples=self._num_traj_samples,
            return_extra=True,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _select_frames_at_target_rate(
        self,
        frames: list[tuple[int, np.ndarray]],
    ) -> list[tuple[int, np.ndarray]]:
        """Select frames that best approximate the target input rate.

        When more frames are available than ``context_length``, selects the
        subset whose timestamps most closely match evenly-spaced intervals at
        :attr:`IMAGE_INPUT_FREQUENCY_HZ`, anchored at the most recent frame.
        """
        if len(frames) <= self._context_length:
            return frames

        sorted_frames = sorted(frames, key=lambda x: x[0])
        t0 = sorted_frames[-1][0]
        interval_us = int(1_000_000 / self.IMAGE_INPUT_FREQUENCY_HZ)

        # Target timestamps from oldest to newest
        targets = [
            t0 - (self._context_length - 1 - i) * interval_us
            for i in range(self._context_length)
        ]

        selected: list[tuple[int, np.ndarray]] = []
        used_indices: set[int] = set()

        for target_ts in targets:
            best_idx = -1
            best_dist = float("inf")
            for idx, (ts, _) in enumerate(sorted_frames):
                if idx in used_indices:
                    continue
                dist = abs(ts - target_ts)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            used_indices.add(best_idx)
            selected.append(sorted_frames[best_idx])

        return selected

    def _preprocess_images(self, camera_images: CameraImages) -> torch.Tensor:
        """Preprocess multi-camera images into a stacked tensor.

        Returns:
            Image tensor of shape ``(N_cameras, num_frames, 3, H, W)`` as
            uint8 ``[0, 255]``.  Cameras are sorted by their index.
        """
        frames_list = []

        # Sort camera IDs by their index to ensure consistent ordering
        sorted_camera_ids = sorted(
            self._camera_ids, key=lambda cam_id: CAMERA_NAME_TO_INDEX[cam_id]
        )

        # Process each camera in sorted order
        for cam_id in sorted_camera_ids:
            frames = camera_images[cam_id]
            images = [img for _, img in frames]

            # Convert to CHW format, keeping uint8 dtype.
            # NOTE: Do NOT normalize to [0, 1] — the VL processor expects
            # uint8 images and handles normalization internally.
            # Pre-normalizing causes double normalization which corrupts
            # the visual features.
            camera_frames = [
                torch.from_numpy(img).permute(2, 0, 1)  # HWC uint8 -> CHW uint8
                for img in images
            ]

            # Stack frames for this camera: (num_frames, C, H, W)
            camera_tensor = torch.stack(camera_frames, dim=0)
            frames_list.append(camera_tensor)

        # Stack all cameras: (N_cameras, num_frames, C, H, W)
        return torch.stack(frames_list, dim=0)

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Generate trajectory prediction.

        Uses camera images and ego pose history.  Command, speed, and
        acceleration are unused.
        """
        self._validate_cameras(prediction_input.camera_images)

        # Down-sample frames to approximate the target input rate.
        camera_images = {
            cam_id: self._select_frames_at_target_rate(frames)
            for cam_id, frames in prediction_input.camera_images.items()
        }

        # Validate context length per camera.
        for cam_id in self._camera_ids:
            if len(camera_images[cam_id]) != self._context_length:
                logger.warning(
                    "%s expects %d frames per camera, got %d for %s",
                    self.__class__.__name__,
                    self._context_length,
                    len(camera_images[cam_id]),
                    cam_id,
                )
                return ModelPrediction(
                    trajectory_xy=np.zeros((0, 2)),
                    headings=np.zeros(0),
                )

        # Check ego history covers the required time span for interpolation.
        # We need at least 2 poses spanning (NUM_HISTORY_STEPS-1)*HISTORY_TIME_STEP
        # seconds back from the current timestamp.  build_ego_history() will
        # interpolate the 16 target samples from these poses.
        required_span_us = (self.NUM_HISTORY_STEPS - 1) * self.HISTORY_TIME_STEP * 1e6
        if (
            prediction_input.ego_pose_history is None
            or len(prediction_input.ego_pose_history) < 2
        ):
            num_poses = (
                0
                if prediction_input.ego_pose_history is None
                else len(prediction_input.ego_pose_history)
            )
            logger.warning(
                "Not enough pose history: %d < 2. Returning empty trajectory.",
                num_poses,
            )
            return ModelPrediction(
                trajectory_xy=np.zeros((0, 2)),
                headings=np.zeros(0),
            )

        # Get current timestamp from the latest frame
        latest_timestamp = max(
            max(ts for ts, _ in frames) for frames in camera_images.values()
        )
        earliest_required_us = latest_timestamp - required_span_us
        earliest_available_us = min(
            p.timestamp_us for p in prediction_input.ego_pose_history
        )
        if earliest_available_us > earliest_required_us:
            logger.warning(
                "Pose history too short: available span %.1fms < required %.1fms. "
                "Returning empty trajectory.",
                (latest_timestamp - earliest_available_us) / 1e3,
                required_span_us / 1e3,
            )
            return ModelPrediction(
                trajectory_xy=np.zeros((0, 2)),
                headings=np.zeros(0),
            )

        ego_history_xyz, ego_history_rot = build_ego_history(
            prediction_input.ego_pose_history,
            latest_timestamp,
            self.NUM_HISTORY_STEPS,
            self.HISTORY_TIME_STEP,
        )

        # Preprocess images and create chat message (model-specific hook).
        image_frames = self._preprocess_images(camera_images)
        messages = self._create_chat_message(image_frames)

        # Apply chat template via the processor.
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
        model_inputs = self._helper.to_device(model_inputs, self._device)

        # Run inference with autocast
        with torch.no_grad():
            with torch.autocast(str(self._device.type), dtype=self.DTYPE):
                pred_xyz, pred_rot, extra = self._run_inference(model_inputs)

        # Extract trajectory (x, y coordinates)
        trajectory_xy = _format_trajs(pred_xyz)

        # Extract headings (yaw per waypoint).
        rot_first = pred_rot[0, 0, 0, :, :, :]  # (T, 3, 3)
        headings = so3_to_yaw_torch(rot_first).detach().cpu().numpy()

        # Log reasoning trace if available.
        reasoning_text = None
        if "cot" in extra and len(extra["cot"]) > 0:
            reasoning_text = str(extra["cot"][0, 0])
            logger.info(
                "%s Chain-of-Causation: %s",
                self.__class__.__name__,
                reasoning_text,
            )

        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning_text,
        )
