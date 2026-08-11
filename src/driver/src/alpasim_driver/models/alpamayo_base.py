# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared base class for the Alpamayo model family (Alpamayo 1, 1.5, etc.)."""

from __future__ import annotations

import logging
import math
import os
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from alpasim_grpc.v0.common_pb2 import PoseAtTime
from alpasim_grpc.v0.egodriver_pb2 import Route
from alpasim_utils.geometry import Pose, Trajectory
from scipy.interpolate import interp1d

from ..schema import ModelConfig
from .base import (
    BaseTrajectoryModel,
    CameraImages,
    DriveCommand,
    ModelInputValidationError,
    ModelPrediction,
    PredictionInput,
)
from .nav_text import route_to_nav_text
from .trajectory_selection import (
    TrajectorySelectionStrategy,
    plan_in_local_frame,
    select_trajectory,
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


def configure_deterministic_runtime() -> None:
    """Configure deterministic PyTorch execution for Alpamayo inference."""
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _trajectory_from_pose_messages(poses: list[PoseAtTime]) -> Trajectory:
    """Convert pose messages into a trajectory, carrying their frame along.

    Args:
        poses: List of PoseAtTime messages.  Each must have .timestamp_us (int),
            .pose.vec.{x,y,z} and .pose.quat.{x,y,z,w}, with a unit-length
            quaternion (validated by the Trajectory constructor).

    Returns:
        The poses as a trajectory, ordered by timestamp, in whichever frame the
        messages were expressed in.
    """
    pose_data = sorted([(p.timestamp_us, p.pose) for p in poses], key=lambda x: x[0])
    return Trajectory(
        np.array([t for t, _ in pose_data], dtype=np.uint64),
        np.array([[p.vec.x, p.vec.y, p.vec.z] for _, p in pose_data], dtype=np.float32),
        np.array(
            [[p.quat.x, p.quat.y, p.quat.z, p.quat.w] for _, p in pose_data],
            dtype=np.float32,
        ),
    )


def _adjust_quaternion_orientation(quaternions: np.ndarray) -> np.ndarray:
    """Match the internal driver's antipodal-quaternion adjustment."""
    signs = np.ones(len(quaternions), dtype=quaternions.dtype)
    signs[1:] = np.where(
        0 <= (quaternions[:-1] * quaternions[1:]).sum(axis=1), 1.0, -1.0
    )
    return quaternions * np.cumprod(signs).reshape((-1, 1))


def _quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Match Kornia 0.8.2's float32 quaternion-to-matrix arithmetic."""
    quaternion = torch.nn.functional.normalize(quaternion, p=2.0, dim=-1, eps=1.0e-12)
    w, x, y, z = quaternion.unbind(dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)
    return torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).reshape((*quaternion.shape[:-1], 3, 3))


def route_in_rig_at_t0(
    route: Route,
    ego_pose_history: list[PoseAtTime],
    pose_local_to_rig_t0: Pose,
) -> np.ndarray:
    """Re-express route waypoints in the rig frame at t0.

    A route carries the rig frame of its own timestamp, which trails t0 (the
    newest camera frame the model conditions on) whenever routes and camera
    frames arrive on different schedules.  Hopping through the local frame takes
    the ego motion in between back out.

    Args:
        route: Route with waypoints in the rig frame at ``route.timestamp_us``.
            NaN waypoints, which pad a route shorter than the requested
            lookahead, stay NaN.
        ego_pose_history: Rig poses in the local frame (``pose_local_to_rig`` at
            each ``timestamp_us``), spanning the route timestamp.
        pose_local_to_rig_t0: Pose of the rig at t0 in the local frame.

    Returns:
        Waypoints of shape ``(N, 3)`` in the rig frame at t0.
    """
    waypoints_in_rig_route = np.array(
        [[w.x, w.y, w.z] for w in route.waypoints], dtype=np.float32
    ).reshape(-1, 3)
    rig_route_in_local = _trajectory_from_pose_messages(
        ego_pose_history
    ).interpolate_pose(np.uint64(route.timestamp_us))
    rig_route_in_rig_t0 = (
        pose_local_to_rig_t0.inverse().as_se3() @ rig_route_in_local.as_se3()
    )
    return (
        waypoints_in_rig_route @ rig_route_in_rig_t0[:3, :3].T
        + rig_route_in_rig_t0[:3, 3]
    )


def build_ego_history(
    poses: list[PoseAtTime],
    current_timestamp_us: int,
    num_history_steps: int = 16,
    history_time_step: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, Pose]:
    """Build ego history tensors from pose data.

    Constructs ego_history_xyz and ego_history_rot tensors in the rig frame
    relative to the current pose (t0).

    Args:
        poses: Rig poses in the local frame (``pose_local_to_rig`` at each
            ``timestamp_us``), spanning the requested history.
        current_timestamp_us: Current timestamp (t0) in microseconds.
        num_history_steps: Number of history steps to produce.
        history_time_step: Time step between history frames in seconds.

    Returns:
        Tuple of (ego_history_xyz, ego_history_rot, pose_local_to_rig_t0):
            - ego_history_xyz: shape (1, 1, num_history_steps, 3), rig frame
            - ego_history_rot: shape (1, 1, num_history_steps, 3, 3), rig frame
            - pose_local_to_rig_t0: pose of the rig at t0 in the local frame, which
              relates model outputs to the frame the driver reports in
    """
    # Mirror the internal driver at the model boundary: float32 relative times,
    # SciPy linear interpolation over concatenated XYZ/WXYZ, quaternion
    # normalization/orientation adjustment, Kornia rotation conversion, then
    # the torch global-to-local transform.

    # 1. Interpolate the ego poses at the target history timestamps
    pose_data = sorted(
        [(pose.timestamp_us, pose.pose) for pose in poses], key=lambda item: item[0]
    )
    timestamps_us = [timestamp for timestamp, _ in pose_data]
    xyz = np.array(
        [[pose.vec.x, pose.vec.y, pose.vec.z] for _, pose in pose_data],
        dtype=np.float32,
    )
    quaternion_wxyz = np.array(
        [[pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z] for _, pose in pose_data],
        dtype=np.float32,
    )
    tparam = np.array(
        [(timestamp - current_timestamp_us) * 1e-6 for timestamp in timestamps_us],
        dtype=np.float32,
    )
    if not np.all(0 <= tparam[1:] - tparam[:-1]):
        raise ValueError(f"ego pose timestamps not sorted {tparam=}")

    quaternion_norm = np.linalg.norm(quaternion_wxyz, axis=1)
    quaternion_wxyz = _adjust_quaternion_orientation(
        quaternion_wxyz / quaternion_norm[:, None]
    )
    interpolate_pose = interp1d(
        tparam,
        np.concatenate((xyz, quaternion_wxyz), axis=1),
        kind="linear",
        axis=0,
        copy=False,
        bounds_error=True,
        assume_sorted=True,
    )
    history_timestamps_us = [
        int(index * history_time_step * 1_000_000) + current_timestamp_us
        for index in range(-num_history_steps + 1, 1)
    ]
    teval = np.array(
        [
            (timestamp - current_timestamp_us) * 1e-6
            for timestamp in history_timestamps_us
        ],
        dtype=np.float32,
    )
    interpolated = interpolate_pose(teval)
    history_xyz = interpolated[..., :3]
    history_quaternion_wxyz = interpolated[..., 3:]
    history_quaternion_wxyz /= np.linalg.norm(
        history_quaternion_wxyz, axis=-1, keepdims=True
    )

    # 2. Convert to torch tensors and rotation matrices
    history_xyz_tensor = torch.tensor(history_xyz)
    history_quaternion_tensor = torch.tensor(history_quaternion_wxyz)
    history_rotation_tensor = _quaternion_wxyz_to_matrix(history_quaternion_tensor)

    # 3. Transform to rig frame relative to t0 (last history step) and add
    #    batch dimensions: (B=1, n_traj_group=1, T, ...)
    xyz_t0 = history_xyz_tensor[[-1]]
    rotation_t0 = history_rotation_tensor[[-1]]
    ego_history_xyz_tensor = (
        (rotation_t0.mT @ (history_xyz_tensor - xyz_t0).unsqueeze(-1))
        .squeeze(-1)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    ego_history_rot_tensor = (
        (rotation_t0.mT @ history_rotation_tensor).unsqueeze(0).unsqueeze(0)
    )

    # 4. Keep the t0 pose in the local frame, reordering WXYZ to the XYZW that
    #    `Pose` expects, so callers can map model outputs back to that frame.
    quaternion_t0_xyzw = history_quaternion_wxyz[-1, [1, 2, 3, 0]]
    pose_local_to_rig_t0 = Pose(history_xyz[-1], quaternion_t0_xyzw)

    return ego_history_xyz_tensor, ego_history_rot_tensor, pose_local_to_rig_t0


def _validate_ego_history_span(
    ego_pose_history: list[PoseAtTime] | None,
    *,
    planning_t0_us: int,
    num_history_steps: int,
    history_time_step: float,
    model_name: str,
) -> None:
    """Validate that ego poses cover the model's required history window."""
    required_span_us = int((num_history_steps - 1) * history_time_step * 1_000_000)
    required_span_ms = required_span_us / 1e3
    num_poses = 0 if ego_pose_history is None else len(ego_pose_history)

    if ego_pose_history is None or len(ego_pose_history) < 2:
        raise ModelInputValidationError(
            f"{model_name} needs at least 2 ego poses spanning "
            f"{required_span_ms:.1f}ms before planning_t0_us="
            f"{planning_t0_us}, got {num_poses} pose(s)."
        )

    earliest_required_us = planning_t0_us - required_span_us
    earliest_available_us = min(p.timestamp_us for p in ego_pose_history)
    latest_available_us = max(p.timestamp_us for p in ego_pose_history)
    available_span_ms = (latest_available_us - earliest_available_us) / 1e3

    if latest_available_us < planning_t0_us:
        raise ModelInputValidationError(
            f"{model_name} ego pose history is stale: available_span="
            f"{available_span_ms:.1f}ms, required_span={required_span_ms:.1f}ms, "
            f"latest_available_us={latest_available_us}, "
            f"planning_t0_us={planning_t0_us}."
        )

    if earliest_available_us > earliest_required_us:
        raise ModelInputValidationError(
            f"{model_name} ego pose history is too short: available_span="
            f"{available_span_ms:.1f}ms, required_span={required_span_ms:.1f}ms, "
            f"earliest_available_us={earliest_available_us}, "
            f"earliest_required_us={earliest_required_us}, "
            f"planning_t0_us={planning_t0_us}."
        )


def _validate_camera_frame_at_or_before_t0(
    camera_images: CameraImages,
    *,
    planning_t0_us: int,
    model_name: str,
) -> None:
    """Require visual context that is not newer than the planning origin."""
    camera_timestamps_us = [
        timestamp_us
        for frames in camera_images.values()
        for timestamp_us, _image in frames
    ]
    if any(timestamp_us <= planning_t0_us for timestamp_us in camera_timestamps_us):
        return

    earliest_camera_frame_us = (
        min(camera_timestamps_us) if camera_timestamps_us else None
    )
    raise ModelInputValidationError(
        f"{model_name} needs at least one camera frame at or before "
        f"planning_t0_us={planning_t0_us}, got "
        f"earliest_camera_frame_us={earliest_camera_frame_us}."
    )


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
    # Token budget for the chain of thought the VLM generates before it emits
    # the trajectory tokens.  Left unset, the models fall back to
    # ``tokens_per_future_traj`` (64), which cuts the reasoning short: the
    # trajectory expert then runs on an unterminated chain of thought and the
    # checkpoint logs "No <traj_future_start> token found".
    MAX_GENERATION_LENGTH: int = 256
    # What the vision processor resizes 16:9 renders to.
    MIN_FRAME_HW: tuple[int, int] = (320, 576)

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
        force_determinism: bool = False,
        selection_strategy: TrajectorySelectionStrategy = (
            TrajectorySelectionStrategy.ALWAYS_FIRST
        ),
        max_num_distance_points: int = 64,
        skip_first_n_distance_points: int = 0,
    ) -> None:
        """Shared initialisation called by every subclass ``__init__``."""
        if (
            selection_strategy is not TrajectorySelectionStrategy.ALWAYS_FIRST
            and num_traj_samples < 2
        ):
            raise ValueError(
                f"Trajectory selection strategy {selection_strategy} needs at least "
                f"2 samples to choose from, got num_traj_samples={num_traj_samples}."
            )
        # An empty or reversed distance window still yields an index, so a
        # mistyped value would silently drive candidate 0 on every cycle.
        if max_num_distance_points < 1:
            raise ValueError(
                "max_num_distance_points must be at least 1, got "
                f"{max_num_distance_points}."
            )
        if skip_first_n_distance_points < 0:
            raise ValueError(
                "skip_first_n_distance_points must not be negative, got "
                f"{skip_first_n_distance_points}."
            )

        self._model = model
        self._processor = processor
        self._helper = helper_module
        self._device = device
        self._camera_ids = camera_ids
        self._context_length = context_length
        self._num_traj_samples = num_traj_samples
        self._top_p = top_p
        self._temperature = temperature
        self._force_determinism = force_determinism
        self._selection_strategy = selection_strategy
        self._max_num_distance_points = max_num_distance_points
        self._skip_first_n_distance_points = skip_first_n_distance_points

        action_space = self._model.action_space
        self._pred_num_waypoints, _ = action_space.get_action_space_dims()

        # Waypoints are stamped at OUTPUT_FREQUENCY_HZ, both for the trajectory
        # handed to the controller and for the timestamps candidate selection
        # compares against.  A checkpoint trained on a different step would be
        # read back as motion that is uniformly too fast or too slow.
        # The tolerance is loose enough to accept a step that went through
        # float32 on its way out of a checkpoint, and far tighter than a step
        # difference that would distort the motion.
        reported_time_step = 1.0 / self.OUTPUT_FREQUENCY_HZ
        if not math.isclose(
            action_space.dt, reported_time_step, rel_tol=1e-5, abs_tol=1e-6
        ):
            raise ValueError(
                f"{self.__class__.__name__} checkpoint predicts a waypoint every "
                f"{action_space.dt}s, but the driver reports waypoints every "
                f"{reported_time_step}s (OUTPUT_FREQUENCY_HZ="
                f"{self.OUTPUT_FREQUENCY_HZ})."
            )

        missing_cameras = [
            cam_id for cam_id in camera_ids if cam_id not in CAMERA_NAME_TO_INDEX
        ]
        if missing_cameras:
            raise ValueError(
                f"Cameras {missing_cameras} not found in Alpamayo camera mapping."
            )

        logger.info(
            "Initialised %s with %d cameras, context_length=%d, "
            "num_traj_samples=%d, selection_strategy=%s",
            self.__class__.__name__,
            len(camera_ids),
            context_length,
            num_traj_samples,
            selection_strategy,
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

        Variants contribute their own constructor arguments through
        :meth:`_variant_init_kwargs` instead of overriding this method.
        """
        # The checkpoint fixes the waypoint spacing, so unlike other models
        # Alpamayo cannot follow a configured frequency; refuse to ignore one.
        if output_frequency_hz != cls.OUTPUT_FREQUENCY_HZ:
            raise ValueError(
                f"{cls.__name__} predicts waypoints at {cls.OUTPUT_FREQUENCY_HZ}Hz "
                f"and cannot resample, but output_frequency_hz is "
                f"{output_frequency_hz}."
            )

        selection = model_cfg.trajectory_selection
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length or cls.DEFAULT_CONTEXT_LENGTH,
            force_determinism=model_cfg.force_determinism,
            num_traj_samples=model_cfg.num_trajectory_samples,
            selection_strategy=TrajectorySelectionStrategy(selection.strategy),
            max_num_distance_points=selection.max_num_distance_points,
            skip_first_n_distance_points=selection.skip_first_n_distance_points,
            **cls._variant_init_kwargs(model_cfg),
        )

    @classmethod
    def _variant_init_kwargs(cls, model_cfg: ModelConfig) -> dict[str, Any]:
        """Constructor arguments only some Alpamayo variants accept."""
        return {}

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
    def _create_chat_message(
        self, image_frames: torch.Tensor, nav_text: str | None
    ) -> list:
        """Build the chat-message list from preprocessed image frames.

        Alpamayo 1 passes flattened frames; Alpamayo 1.5 additionally passes
        camera indices and conditions on ``nav_text``.
        """
        ...

    def _run_inference(
        self, model_inputs: dict[str, Any], nav_text: str | None
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Run trajectory inference.  Override to use an alternative method
        (e.g. classifier-free guidance navigation in A1.5)."""
        return self._model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=self._top_p,
            temperature=self._temperature,
            num_traj_samples=self._num_traj_samples,
            max_generation_length=self.MAX_GENERATION_LENGTH,
            return_extra=True,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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

            # Convert to CHW format, keeping uint8 dtype.  ``as_tensor`` takes
            # a host array or a frame already on the inference device.
            # NOTE: Do NOT normalize to [0, 1] — the VL processor expects
            # uint8 images and handles normalization internally.
            # Pre-normalizing causes double normalization which corrupts
            # the visual features.
            camera_frames = [
                torch.as_tensor(img).permute(2, 0, 1)  # HWC uint8 -> CHW uint8
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
        if self._force_determinism:
            torch.manual_seed(prediction_input.inference_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(prediction_input.inference_seed)

        self._validate_cameras(prediction_input.camera_images)

        # The servicer hands over the newest ``context_length`` frames of each
        # camera, so the model conditions on the most recent stretch of video at
        # whatever rate the cameras run.
        camera_images = prediction_input.camera_images

        # The servicer holds drive requests until every cache is full, so a
        # short camera means the cache and the model disagree on how many frames
        # that is.
        for cam_id in self._camera_ids:
            if len(camera_images[cam_id]) != self._context_length:
                raise ModelInputValidationError(
                    f"{self.__class__.__name__} expects {self._context_length} "
                    f"frames per camera, got {len(camera_images[cam_id])} for "
                    f"{cam_id}."
                )

        # Get current timestamp from the latest frame
        latest_timestamp = max(
            max(ts for ts, _ in frames) for frames in camera_images.values()
        )
        _validate_ego_history_span(
            prediction_input.ego_pose_history,
            planning_t0_us=latest_timestamp,
            num_history_steps=self.NUM_HISTORY_STEPS,
            history_time_step=self.HISTORY_TIME_STEP,
            model_name=self.__class__.__name__,
        )

        ego_history_xyz, ego_history_rot, pose_local_to_rig_t0 = build_ego_history(
            prediction_input.ego_pose_history,
            latest_timestamp,
            self.NUM_HISTORY_STEPS,
            self.HISTORY_TIME_STEP,
        )

        # Alpamayo takes navigation intent as text, so the route reaches the
        # model only through this string.  Without it the model has to guess
        # which way to go at intersections.
        nav_text = (
            route_to_nav_text(
                route_in_rig_at_t0(
                    prediction_input.route,
                    prediction_input.ego_pose_history,
                    pose_local_to_rig_t0,
                )
            )
            if prediction_input.route is not None
            else None
        )
        logger.debug("nav_text=%r", nav_text)

        # Preprocess images and create chat message (model-specific hook).
        image_frames = self._preprocess_images(camera_images)
        messages = self._create_chat_message(image_frames, nav_text)

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
                pred_xyz, pred_rot, extra = self._run_inference(model_inputs, nav_text)

        # Sampled candidates in the rig frame at t0, shapes
        # [batch=1, num_traj_sets=1, K, T, ...] -> (K, T, ...).
        candidate_positions = pred_xyz[0, 0].float().cpu().numpy()
        candidate_rotations = pred_rot[0, 0].float().cpu().numpy()
        waypoint_timestamps_us = self._waypoint_timestamps_us(
            latest_timestamp, num_waypoints=candidate_positions.shape[1]
        )

        select_ix = select_trajectory(
            candidate_positions=candidate_positions,
            candidate_timestamps_us=waypoint_timestamps_us,
            previous_plan_in_local=prediction_input.previous_plan,
            pose_local_to_rig_t0=pose_local_to_rig_t0,
            strategy=self._selection_strategy,
            max_num_distance_points=self._max_num_distance_points,
            skip_first_n_distance_points=self._skip_first_n_distance_points,
        )
        selected_positions = candidate_positions[select_ix]  # (T, 3)
        selected_rotations = candidate_rotations[select_ix]  # (T, 3, 3)

        # Log the reasoning trace of the selected candidate if available.  Text
        # outputs have shape [batch, num_traj_sets, K] like the trajectories.
        reasoning_text = None
        if "cot" in extra:
            reasoning_text = str(extra["cot"][0, 0, select_ix])
            logger.info(
                "%s Chain-of-Causation: %s",
                self.__class__.__name__,
                reasoning_text,
            )

        return ModelPrediction(
            candidate_positions=candidate_positions,
            candidate_rotations=candidate_rotations,
            selected_index=select_ix,
            reasoning_text=reasoning_text,
            model_t0_us=latest_timestamp,
            pose_local_to_rig_t0=pose_local_to_rig_t0,
            waypoint_timestamps_us=waypoint_timestamps_us,
            selected_plan=plan_in_local_frame(
                selected_positions,
                selected_rotations,
                waypoint_timestamps_us,
                pose_local_to_rig_t0,
            ),
        )

    def _waypoint_timestamps_us(self, t0_us: int, num_waypoints: int) -> np.ndarray:
        """Timestamps of predicted waypoints, which start one step after t0."""
        step_us = 1_000_000 // self.output_frequency_hz
        return t0_us + np.arange(1, num_waypoints + 1, dtype=np.uint64) * step_us
