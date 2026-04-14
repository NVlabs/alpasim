# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Unified driver implementation for Alpasim supporting multiple model backends."""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
import os
import pickle
import queue
import socket
import threading
from dataclasses import dataclass, field
from importlib.metadata import version
from io import BytesIO
from typing import Any, Callable, Optional, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_grpc.v0.common_pb2 import (
    DynamicState,
    Empty,
    Pose,
    PoseAtTime,
    Quat,
    SessionRequestStatus,
    Trajectory,
    Vec3,
    VersionId,
)
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    Route,
    RouteRequest,
)
from alpasim_grpc.v0.egodriver_pb2_grpc import (
    EgodriverServiceServicer,
    add_EgodriverServiceServicer_to_server,
)
from alpasim_plugins.plugins import models as model_registry
from omegaconf import OmegaConf
from PIL import Image

import grpc
import grpc.aio

from .frame_cache import FrameCache
from .models import DriveCommand
from .models.base import (
    BaseTrajectoryModel,
    CameraImages,
    ModelPrediction,
    PredictionInput,
)
from .models.manual_model import ManualModel
from .navigation import determine_command_from_route
from .rectification import (
    FthetaToPinholeRectifier,
    build_ftheta_rectifier_for_resolution,
)
from .schema import DriverConfig, ModelConfig, RectificationTargetConfig
from .trajectory_optimizer import (
    TrajectoryOptimizer,
    VehicleConstraints,
    add_heading_to_trajectory,
)

logger = logging.getLogger(__name__)


def _get_external_ip() -> str:
    """Get the external IP address of this machine.

    Uses a UDP socket to determine which local interface would be used
    to reach an external address (without actually sending any data).

    Returns:
        The external IP address as a string, or "unknown" if detection fails.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to an external address (doesn't send data, just determines route)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "unknown"


def _quat_to_yaw(quaternion: Quat) -> float:
    """Extract the yaw component (rotation about +Z) from a quaternion."""

    return np.arctan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _yaw_to_quat(yaw: float) -> Quat:
    """Create a Z-only rotation quaternion from the provided yaw angle."""

    half_yaw = 0.5 * yaw
    return Quat(w=float(np.cos(half_yaw)), x=0.0, y=0.0, z=float(np.sin(half_yaw)))


def _rig_est_offsets_to_local_positions(
    current_pose_in_local: PoseAtTime, offsets_in_rig: np.ndarray
) -> np.ndarray:
    """Project rig-est displacements onto the local-frame pose anchored by `current_pose`."""

    curr_x = current_pose_in_local.pose.vec.x
    curr_y = current_pose_in_local.pose.vec.y

    curr_quat = current_pose_in_local.pose.quat
    curr_yaw = _quat_to_yaw(curr_quat)

    cos_yaw = np.cos(curr_yaw)
    sin_yaw = np.sin(curr_yaw)

    offsets_array = np.asarray(offsets_in_rig, dtype=float).reshape(-1, 2)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)
    rotated_offsets = offsets_array @ rotation.T

    translation = np.array([curr_x, curr_y], dtype=float)
    return rotated_offsets + translation


# Unique queue marker instructing the worker thread to flush and exit.
_SENTINEL_JOB = object()
_DRIVER_CONTROL_PREFIX = b"ALPASIM_DRIVER_CTRL_V1:"


def _decode_driver_control(
    renderer_data: bytes | None,
) -> tuple[str | None, dict[str, Any] | None, bytes]:
    """Decode per-frame driver control envelope from renderer_data."""
    if renderer_data is None:
        return None, None, b""
    if not renderer_data.startswith(_DRIVER_CONTROL_PREFIX):
        return None, None, renderer_data

    payload = renderer_data[len(_DRIVER_CONTROL_PREFIX) :]
    try:
        obj = json.loads(payload.decode("utf-8"))
    except Exception:
        logger.warning("Invalid driver control payload; ignoring override")
        return None, None, b""

    next_model = obj.get("next_model")
    if next_model is not None and not isinstance(next_model, str):
        logger.warning("driver control next_model must be string; ignoring override")
        next_model = None

    planner_context = obj.get("planner_context")
    if planner_context is not None and not isinstance(planner_context, dict):
        logger.warning("driver control planner_context must be object; ignoring")
        planner_context = None

    raw_renderer_data = obj.get("renderer_data_b64")
    if raw_renderer_data is None:
        return next_model, planner_context, b""
    if not isinstance(raw_renderer_data, str):
        logger.warning("driver control renderer_data_b64 must be string")
        return next_model, planner_context, b""

    try:
        decoded = base64.b64decode(raw_renderer_data.encode("ascii"))
    except Exception:
        logger.warning("Failed to decode renderer_data_b64 from driver control payload")
        return next_model, planner_context, b""
    return next_model, planner_context, decoded


@dataclass
class DriveJob:
    """Unit of work processed by the background inference worker."""

    session_id: str
    session: "Session"
    command: DriveCommand
    model_type: str
    planner_context: dict[str, Any] | None
    pose: Optional[PoseAtTime]
    timestamp_us: int
    result: asyncio.Future[ModelPrediction]


@dataclass
class Session:
    """Represents a driver session."""

    uuid: str
    seed: int
    debug_scene_id: str

    frame_caches: dict[str, FrameCache]
    available_cameras_logical_ids: set[str]
    desired_cameras_logical_ids: set[str]
    camera_specs: dict[str, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera]
    rectification_cfg: Optional[dict[str, RectificationTargetConfig]] = None
    rectifiers: dict[str, Optional[FthetaToPinholeRectifier]] = field(
        default_factory=dict
    )
    poses: list[PoseAtTime] = field(default_factory=list)
    dynamic_states: list[tuple[int, DynamicState]] = field(default_factory=list)
    current_command: DriveCommand = DriveCommand.STRAIGHT  # Default to straight

    @staticmethod
    def create(
        request: DriveSessionRequest,
        cfg: DriverConfig,
        context_length: int,
        subsample_factor: int = 1,
    ) -> Session:
        """Create a new driver session.

        Args:
            request: The gRPC session request with vehicle/camera definitions.
            cfg: Driver configuration.
            context_length: Number of temporal frames needed.
            subsample_factor: Subsampling factor for frames.

        Returns:
            A new Session instance.

        Note:
            Camera count validation is now handled by the model's __init__
            which raises ValueError if the camera count doesn't match.
        """
        debug_scene_id = (
            request.debug_info.scene_id
            if request.debug_info is not None
            else request.session_uuid
        )

        available_cameras_logical_ids: set[str] = set()
        vehicle = request.rollout_spec.vehicle
        if vehicle is None:
            raise ValueError("Vehicle definition is required in DriveSessionRequest")

        camera_specs: dict[
            str, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera
        ] = {}
        for camera_def in vehicle.available_cameras:
            if not camera_def.logical_id:
                raise ValueError(
                    "Logical ID is required for each camera in VehicleDefinition"
                )
            available_cameras_logical_ids.add(camera_def.logical_id)
            camera_specs[camera_def.logical_id] = camera_def
            logger.debug(
                f"Available camera: {camera_def.logical_id}, "
                f"resolution: ({camera_def.intrinsics.resolution_h}, {camera_def.intrinsics.resolution_w}), "
                f"intrinsics: {camera_def.intrinsics}"
            )

        desired_cameras_logical_ids = set(cfg.inference.use_cameras)
        if not desired_cameras_logical_ids:
            raise ValueError("No cameras specified in inference configuration")

        missing_defs = desired_cameras_logical_ids - set(camera_specs.keys())
        if missing_defs:
            raise ValueError(
                f"Requested cameras {sorted(missing_defs)} are missing from the rollout spec"
            )
        if cfg.rectification is not None:
            missing_rect = desired_cameras_logical_ids - set(cfg.rectification.keys())
            if missing_rect:
                raise ValueError(
                    "Missing rectification targets for cameras "
                    f"{sorted(missing_rect)} in driver configuration"
                )

        rectifiers: dict[str, Optional[FthetaToPinholeRectifier]] = {
            logical_id: None for logical_id in desired_cameras_logical_ids
        }

        # Create a FrameCache for each desired camera
        frame_caches: dict[str, FrameCache] = {}
        for camera_id in cfg.inference.use_cameras:
            frame_caches[camera_id] = FrameCache(
                context_length=context_length,
                camera_id=camera_id,
                subsample_factor=subsample_factor,
            )

        session = Session(
            uuid=request.session_uuid,
            seed=request.random_seed,
            debug_scene_id=debug_scene_id,
            frame_caches=frame_caches,
            available_cameras_logical_ids=available_cameras_logical_ids,
            desired_cameras_logical_ids=desired_cameras_logical_ids,
            camera_specs=camera_specs,
            rectification_cfg=cfg.rectification,
            rectifiers=rectifiers,
        )

        return session

    def add_image(
        self, logical_id: str, image_tensor: np.ndarray, timestamp_us: int
    ) -> None:
        """Add an image observation for a specific camera."""
        if logical_id not in self.frame_caches:
            raise ValueError(
                f"Camera {logical_id} not in desired cameras: {list(self.frame_caches.keys())}"
            )
        self.frame_caches[logical_id].add_image(timestamp_us, image_tensor)

    def all_cameras_ready(self) -> bool:
        """Check if all cameras have enough frames for inference."""
        return all(cache.has_enough_frames() for cache in self.frame_caches.values())

    def min_frame_count(self) -> int:
        """Return the minimum frame count across all cameras."""
        if not self.frame_caches:
            return 0
        return min(cache.frame_count() for cache in self.frame_caches.values())

    def _maybe_build_rectifier(
        self, logical_id: str, source_resolution_hw: tuple[int, int]
    ) -> Optional[FthetaToPinholeRectifier]:
        """Instantiate and cache a rectifier once the true source resolution is known."""

        # Check if there's a rectifier for target camera in the config
        if self.rectification_cfg is None or logical_id not in self.rectification_cfg:
            return None

        # Check if we already have a rectifier for this camera
        if self.rectifiers.get(logical_id) is not None:
            return self.rectifiers[logical_id]

        # Build the rectifier
        rectifier = build_ftheta_rectifier_for_resolution(
            camera_proto=self.camera_specs[logical_id],
            target_cfg=self.rectification_cfg[logical_id],
            source_resolution_hw=source_resolution_hw,
        )
        self.rectifiers[logical_id] = rectifier
        logger.debug(
            "Built f-theta rectifier for %s using source resolution %s",
            logical_id,
            source_resolution_hw,
        )
        return rectifier

    def rectify_image(self, logical_id: str, image: Image.Image) -> Image.Image:
        """Apply rectification for logical_id if configured."""
        source_resolution_hw = (image.height, image.width)

        # Need to do this lazily as we won't know the source resolution until
        # after the first image is received.
        # (The available cameras define the native camera resolutio, not the
        # rendering resolution.)
        rectifier = self._maybe_build_rectifier(logical_id, source_resolution_hw)

        if rectifier is None:
            return image
        return Image.fromarray(rectifier.rectify(np.array(image)))

    def add_egoposes(self, egoposes: Trajectory) -> None:
        """Add rig-est pose observations in the local frame."""
        self.poses.extend(egoposes.poses)
        self.poses = sorted(self.poses, key=lambda pose: pose.timestamp_us)
        logger.debug(f"poses: {self.poses}")

    def add_dynamic_state(
        self, timestamp_us: int, dynamic_state: Optional[DynamicState]
    ) -> None:
        """Add a dynamic state observation at the given timestamp.

        Args:
            timestamp_us: Timestamp in microseconds for this observation.
            dynamic_state: The dynamic state (velocities, accelerations) in rig frame.
                May be None if not provided by the client.
        """
        if dynamic_state is None:
            raise ValueError("Dynamic state is required")
        self.dynamic_states.append((timestamp_us, dynamic_state))
        self.dynamic_states = sorted(self.dynamic_states, key=lambda x: x[0])
        logger.debug(
            f"dynamic_state at {timestamp_us}: "
            f"lin_vel=({dynamic_state.linear_velocity.x:.2f}, "
            f"{dynamic_state.linear_velocity.y:.2f}, "
            f"{dynamic_state.linear_velocity.z:.2f})"
        )

    def update_command_from_route(
        self,
        route: Route,
        use_waypoint_commands: bool,
        command_distance_threshold: Optional[float] = None,
        min_lookahead_distance: Optional[float] = None,
    ) -> None:
        """Derive command from waypoints using route geometry.

        Note: this is called for RouteRequest and assumed to be in the
        true rig frame.

        Args:
            route: Route containing waypoints in the rig frame.
            use_waypoint_commands: Whether to derive commands from waypoints.
            command_distance_threshold: Lateral distance threshold (meters) for
                determining turn commands. Waypoints beyond this threshold trigger
                LEFT/RIGHT commands.
            min_lookahead_distance: Minimum forward distance (meters) to consider
                a waypoint as the target for command derivation.
        """
        if not use_waypoint_commands or len(route.waypoints) < 1:
            return

        if len(self.poses) == 0:
            return

        if command_distance_threshold is None or min_lookahead_distance is None:
            raise ValueError(
                "command_distance_threshold and min_lookahead_distance must be provided "
                "when use_waypoint_commands is True"
            )

        # Use the navigation module to determine command
        self.current_command = determine_command_from_route(
            route=route,
            command_distance_threshold=command_distance_threshold,
            min_lookahead_distance=min_lookahead_distance,
        )

        logger.debug(
            "Command updated: %s",
            self.current_command.name,
        )


def async_log_call(func: Callable) -> Callable:
    """Helper to add logging for gRPC calls (sync or async)."""

    @functools.wraps(func)
    async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            logger.debug("Calling %s", func.__name__)
            return await func(*args, **kwargs)
        except Exception:  # pragma: no cover - logging assistance
            logger.exception("Exception in %s", func.__name__)
            raise

    return async_wrapped


def _create_model(
    cfg: ModelConfig,
    device: torch.device,
    camera_ids: list[str],
    context_length: int | None,
    output_frequency_hz: int,
) -> BaseTrajectoryModel:
    """Factory method to create the appropriate model.

    Uses the alpasim.models plugin registry to discover and instantiate
    models.  Each registered model class provides a ``from_config()``
    classmethod that extracts the parameters it needs from the generic
    argument set, so no model-specific branching is required here.

    Args:
        cfg: Model configuration (``cfg.model_type`` is the entry-point name).
        device: Torch device to load model on.
        camera_ids: List of camera logical IDs in order.
        context_length: Number of temporal frames (None uses model default).
        output_frequency_hz: Trajectory output frequency in Hz.

    Returns:
        Model instance implementing BaseTrajectoryModel.

    Raises:
        PluginNotFoundError: If model_type is not found in the plugin registry.
    """
    model_cls = model_registry.get(cfg.model_type)
    return model_cls.from_config(
        cfg, device, camera_ids, context_length, output_frequency_hz
    )


class EgoDriverService(EgodriverServiceServicer):
    """Unified policy service supporting multiple model backends."""

    def __init__(
        self,
        cfg: DriverConfig,
        loop: asyncio.AbstractEventLoop,
        grpc_server: grpc.aio.Server,
    ) -> None:
        """Initialize the Ego Driver service.

        Sets up the model backend, and starts a background
        worker thread for batched inference processing.

        Args:
            cfg: Hydra configuration containing model paths and inference settings
            loop: Asyncio event loop for coordinating async operations and scheduling
                futures from the worker thread back to the async gRPC handlers
            grpc_server: gRPC server instance for service registration
        """

        # Private members
        self._cfg = cfg
        self._loop = loop
        self._grpc_server = grpc_server

        # Determine device
        self._device = torch.device(
            cfg.model.device if torch.cuda.is_available() else "cpu"
        )

        # Create default model using factory
        self._default_model_type = cfg.model.model_type
        self._model = _create_model(
            cfg.model,
            self._device,
            camera_ids=cfg.inference.use_cameras,
            context_length=cfg.inference.context_length,
            output_frequency_hz=cfg.inference.output_frequency_hz,
        )
        self._models: dict[str, BaseTrajectoryModel] = {
            self._default_model_type: self._model
        }

        # Get context length from model or config override
        self._context_length = (
            cfg.inference.context_length
            if cfg.inference.context_length is not None
            else self._model.context_length
        )

        logger.info(
            "Initialized %s model with %d cameras, context_length=%d",
            self._default_model_type,
            self._model.num_cameras,
            self._context_length,
        )

        self._max_batch_size = cfg.inference.max_batch_size
        self._job_queue: queue.Queue[DriveJob | object] = queue.Queue()
        self._worker_stop = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_main,
            name="ego-driver-worker",
            daemon=True,
        )
        self._sessions: dict[str, Session] = {}

        # Initialize trajectory optimizer if enabled
        self._trajectory_optimizer: Optional[TrajectoryOptimizer] = None
        self._vehicle_constraints: Optional[VehicleConstraints] = None
        if cfg.trajectory_optimizer.enabled:
            opt_cfg = cfg.trajectory_optimizer
            self._trajectory_optimizer = TrajectoryOptimizer(
                smoothness_weight=opt_cfg.smoothness_weight,
                deviation_weight=opt_cfg.deviation_weight,
                comfort_weight=opt_cfg.comfort_weight,
                max_iterations=opt_cfg.max_iterations,
                enable_frenet_retiming=opt_cfg.retime_in_frenet,
                retime_alpha=opt_cfg.retime_alpha,
            )
            self._vehicle_constraints = VehicleConstraints(
                max_deviation=opt_cfg.max_deviation,
                max_heading_change=opt_cfg.max_heading_change,
                max_speed=opt_cfg.max_speed,
                max_accel=opt_cfg.max_accel,
                max_abs_yaw_rate=opt_cfg.max_abs_yaw_rate,
                max_abs_yaw_acc=opt_cfg.max_abs_yaw_acc,
                max_lon_acc_pos=opt_cfg.max_lon_acc_pos,
                max_lon_acc_neg=opt_cfg.max_lon_acc_neg,
                max_abs_lon_jerk=opt_cfg.max_abs_lon_jerk,
            )

            logger.info(
                "Trajectory optimizer enabled with retiming=%s, alpha=%.2f",
                opt_cfg.retime_in_frenet,
                opt_cfg.retime_alpha,
            )
            logger.info(f"Trajectory optimizer config: {opt_cfg}")

        self._worker_thread.start()

    async def stop_worker(self) -> None:
        """Signal the worker thread to stop and wait for it to exit."""
        if not self._worker_stop.is_set():
            self._worker_stop.set()
            self._job_queue.put_nowait(_SENTINEL_JOB)
        if self._worker_thread.is_alive():
            await asyncio.to_thread(self._worker_thread.join)

    def _worker_main(self) -> None:
        """Blocking worker loop that batches drive jobs for inference."""
        torch.set_grad_enabled(False)
        batch_count = 0
        total_items = 0
        while True:
            if self._worker_stop.is_set():
                break

            # Get at least one job
            try:
                job = self._job_queue.get()
            except queue.Empty:
                continue

            # Check if we should stop
            if job is _SENTINEL_JOB:
                break

            batch: list[DriveJob] = [job]

            # Get as many jobs as we can
            stop_after_batch = False
            while len(batch) < self._max_batch_size:
                try:
                    next_job = self._job_queue.get_nowait()
                except queue.Empty:
                    break
                if next_job is _SENTINEL_JOB:
                    stop_after_batch = True
                    break
                batch.append(next_job)

            try:
                logger.debug("Running inference batch of size %s", len(batch))
                responses = self._run_batch(batch)
                batch_count += 1
                total_items += len(batch)
                if batch_count % 100 == 0:
                    logger.info(
                        "Inference batches: %d processed, %d total items, avg size %.1f",
                        batch_count,
                        total_items,
                        total_items / batch_count,
                    )
            except Exception as exc:
                logger.exception("Inference batch failed")
                for pending_job in batch:
                    self._loop.call_soon_threadsafe(
                        pending_job.result.set_exception, exc
                    )
            else:
                logger.debug("Inference batch succeeded")
                for pending_job, response in zip(batch, responses, strict=True):
                    self._loop.call_soon_threadsafe(
                        pending_job.result.set_result, response
                    )

            if stop_after_batch:
                break

        # Signal the worker thread to stop
        self._worker_stop.set()
        while True:
            try:
                leftover = self._job_queue.get_nowait()
            except queue.Empty:
                break
            if leftover is _SENTINEL_JOB:
                continue
            self._loop.call_soon_threadsafe(leftover.result.cancel)

    def _get_model(self, model_type: str) -> BaseTrajectoryModel:
        """Return model instance for model_type, creating it lazily."""
        existing = self._models.get(model_type)
        if existing is not None:
            return existing

        model_cfg = ModelConfig(
            model_type=model_type,
            checkpoint_path=self._cfg.model.checkpoint_path,
            device=self._cfg.model.device,
            tokenizer_path=self._cfg.model.tokenizer_path,
        )
        candidate = _create_model(
            model_cfg,
            self._device,
            camera_ids=self._cfg.inference.use_cameras,
            context_length=self._cfg.inference.context_length,
            output_frequency_hz=self._cfg.inference.output_frequency_hz,
        )

        # Switching is only safe when model I/O cadence matches the default runtime setup.
        if candidate.camera_ids != self._model.camera_ids:
            raise ValueError(
                f"Model '{model_type}' camera_ids={candidate.camera_ids} do not match "
                f"default model camera_ids={self._model.camera_ids}"
            )
        if candidate.context_length != self._model.context_length:
            raise ValueError(
                f"Model '{model_type}' context_length={candidate.context_length} does "
                f"not match default model context_length={self._model.context_length}"
            )
        if candidate.output_frequency_hz != self._model.output_frequency_hz:
            raise ValueError(
                f"Model '{model_type}' output_frequency_hz="
                f"{candidate.output_frequency_hz} does not match default model "
                f"output_frequency_hz={self._model.output_frequency_hz}"
            )

        self._models[model_type] = candidate
        logger.info("Loaded switchable model '%s'", model_type)
        return candidate

    def _resolve_model_type_for_request(self, request: DriveRequest) -> str:
        """Resolve per-request model type from optional control payload."""
        override_model, _, _ = _decode_driver_control(request.renderer_data)
        if override_model is None or override_model.strip() == "":
            return self._default_model_type
        return override_model

    def _resolve_planner_context_for_request(
        self, request: DriveRequest
    ) -> dict[str, Any] | None:
        """Resolve per-request planner context from optional control payload."""
        _, planner_context, _ = _decode_driver_control(request.renderer_data)
        return planner_context

    def _get_speed_and_acceleration(self, session: Session) -> tuple[float, float]:
        """Extract speed and acceleration from session's dynamic state.

        Falls back to finite differences from ego positions if dynamic state
        reports zero speed and acceleration.

        Args:
            session: Session containing dynamic state history.

        Returns:
            Tuple of (speed_m_s, acceleration_m_s2).

        Raises:
            ValueError: If no dynamic states are available.
        """
        if not session.dynamic_states:
            raise ValueError(
                "No dynamic states available in session. "
                "Ensure egomotion observations are submitted before calling drive."
            )

        _, state = session.dynamic_states[-1]
        speed = np.sqrt(state.linear_velocity.x**2 + state.linear_velocity.y**2)
        acceleration = state.linear_acceleration.x

        return float(speed), float(acceleration)

    def _prepare_camera_images(
        self, session: Session, model: BaseTrajectoryModel
    ) -> CameraImages:
        """Collect raw images from frame caches for all cameras.

        Returns dict mapping camera_id to list of CameraFrame tuples.
        List length equals context_length.
        """
        camera_images: CameraImages = {}

        for cam_id in model.camera_ids:
            frame_cache = session.frame_caches[cam_id]
            entries = frame_cache.latest_frame_entries(self._context_length)
            camera_images[cam_id] = [(e.timestamp_us, e.image) for e in entries]

        return camera_images

    def _maybe_save_rectification_debug_image(
        self,
        pre_image: Image.Image,
        post_image: Image.Image,
        scene_id: str,
        logical_id: str,
        timestamp_us: int,
    ) -> None:
        """Save pre- and post-rectification images side by side for
        debugging."""

        if not self._cfg.plot_debug_images:
            return

        if not self._cfg.output_dir:
            logger.warning("Output directory is not set; skipping rectification dump")
            return

        session_folder = os.path.join(
            self._cfg.output_dir, scene_id, "rectification_debug"
        )
        os.makedirs(session_folder, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(np.array(pre_image))
        axes[0].set_title(f"Pre-rectification ({pre_image.width}x{pre_image.height})")
        axes[0].axis("off")

        axes[1].imshow(np.array(post_image))
        axes[1].set_title(
            f"Post-rectification ({post_image.width}x{post_image.height})"
        )
        axes[1].axis("off")

        fig.suptitle(f"{logical_id} @ {timestamp_us} µs")
        fig.tight_layout()

        filename = f"{timestamp_us}_{logical_id}_rectification.png"
        output_path = os.path.join(session_folder, filename)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _run_batch(self, batch: list[DriveJob]) -> list[ModelPrediction]:
        """Run inference for a batch of jobs using the model abstraction.

        Builds a PredictionInput per job and delegates to predict_batch(),
        which models can override for GPU-level batching.
        """
        predictions: list[ModelPrediction | None] = [None] * len(batch)
        grouped: dict[str, list[tuple[int, DriveJob]]] = {}
        for index, job in enumerate(batch):
            grouped.setdefault(job.model_type, []).append((index, job))

        for model_type, indexed_jobs in grouped.items():
            model = self._get_model(model_type)
            inputs: list[PredictionInput] = []
            for _, job in indexed_jobs:
                speed, acceleration = self._get_speed_and_acceleration(job.session)
                inputs.append(
                    PredictionInput(
                        camera_images=self._prepare_camera_images(job.session, model),
                        command=job.command,
                        speed=speed,
                        acceleration=acceleration,
                        ego_pose_history=job.session.poses,
                        planner_context=job.planner_context,
                    )
                )

            model_predictions = model.predict_batch(inputs)
            for (index, _), prediction in zip(
                indexed_jobs, model_predictions, strict=True
            ):
                predictions[index] = prediction

        return [cast(ModelPrediction, prediction) for prediction in predictions]

    @async_log_call
    async def start_session(
        self, request: DriveSessionRequest, context: grpc.aio.ServicerContext
    ) -> SessionRequestStatus:
        if request.session_uuid in self._sessions:
            context.abort(
                grpc.StatusCode.ALREADY_EXISTS,
                f"Session {request.session_uuid} already exists.",
            )
            return SessionRequestStatus()

        logger.info(
            "Starting %s session %s",
            self._cfg.model.model_type,
            request.session_uuid,
        )
        session = Session.create(
            request,
            self._cfg,
            self._context_length,
            subsample_factor=self._cfg.inference.subsample_factor,
        )
        self._sessions[request.session_uuid] = session

        return SessionRequestStatus()

    @async_log_call
    async def close_session(
        self, request: DriveSessionCloseRequest, context: grpc.aio.ServicerContext
    ) -> Empty:
        if request.session_uuid not in self._sessions:
            raise KeyError(f"Session {request.session_uuid} does not exist.")

        logger.info(f"Closing session {request.session_uuid}")
        del self._sessions[request.session_uuid]
        return Empty()

    @async_log_call
    async def get_version(
        self, request: Empty, context: grpc.aio.ServicerContext
    ) -> VersionId:
        driver_version = version("alpasim_driver")
        model_type = self._cfg.model.model_type
        return VersionId(
            version_id=f"{model_type}-driver-{driver_version}",
            git_hash="unknown",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    @async_log_call
    async def submit_image_observation(
        self, request: RolloutCameraImage, context: grpc.aio.ServicerContext
    ) -> Empty:
        grpc_image = request.camera_image
        image = Image.open(BytesIO(grpc_image.image_bytes))
        session = self._sessions[request.session_uuid]
        if grpc_image.logical_id not in session.desired_cameras_logical_ids:
            raise ValueError(f"Camera {grpc_image.logical_id} not in desired cameras")

        rectified_image = session.rectify_image(grpc_image.logical_id, image)
        self._maybe_save_rectification_debug_image(
            image,
            rectified_image,
            session.debug_scene_id,
            grpc_image.logical_id,
            grpc_image.frame_end_us,
        )
        session.add_image(
            grpc_image.logical_id,
            np.array(rectified_image),
            grpc_image.frame_end_us,
        )

        return Empty()

    @async_log_call
    async def submit_egomotion_observation(
        self, request: RolloutEgoTrajectory, context: grpc.aio.ServicerContext
    ) -> Empty:
        session = self._sessions[request.session_uuid]

        session.add_egoposes(request.trajectory)

        # Track dynamic states if provided (velocities, accelerations in rig frame)
        # Entries correspond 1:1 with trajectory.poses
        for pose, dynamic_state in zip(
            request.trajectory.poses, request.dynamic_states, strict=True
        ):
            session.add_dynamic_state(pose.timestamp_us, dynamic_state)

        return Empty()

    @async_log_call
    async def submit_route(
        self, request: RouteRequest, context: grpc.aio.ServicerContext
    ) -> Empty:
        logger.debug("submit_route: waypoint count=%s", len(request.route.waypoints))
        if self._cfg.route is not None:
            self._sessions[request.session_uuid].update_command_from_route(
                request.route,
                self._cfg.route.use_waypoint_commands,
                self._cfg.route.command_distance_threshold,
                self._cfg.route.min_lookahead_distance,
            )
        else:
            self._sessions[request.session_uuid].update_command_from_route(
                request.route,
                use_waypoint_commands=False,
            )
        return Empty()

    @async_log_call
    async def submit_recording_ground_truth(
        self, request: GroundTruthRequest, context: grpc.aio.ServicerContext
    ) -> Empty:
        logger.debug("Ground truth received but not used by driver")
        return Empty()

    def _check_frames_ready(self, session: Session) -> bool:
        """Check if all cameras have enough frames for inference."""
        return session.all_cameras_ready()

    @async_log_call
    async def drive(
        self, request: DriveRequest, context: grpc.aio.ServicerContext
    ) -> DriveResponse:
        if request.session_uuid not in self._sessions:
            raise KeyError(f"Session {request.session_uuid} not found")

        session = self._sessions[request.session_uuid]

        if not self._check_frames_ready(session):
            empty_traj = Trajectory()
            # Get required frame count from first cache (all have same config)
            min_required = next(
                iter(session.frame_caches.values())
            ).min_frames_required()
            logger.debug(
                "Drive request received with insufficient frames: "
                "got %s min frames across cameras, need at least %s frames "
                "(context_length=%s, subsample_factor=%s). Returning empty trajectory",
                session.min_frame_count(),
                min_required,
                self._context_length,
                self._cfg.inference.subsample_factor,
            )
            return DriveResponse(
                trajectory=empty_traj,
            )

        pose_snapshot = session.poses[-1] if session.poses else None
        logger.debug(f"pose_snapshot: {pose_snapshot}")
        if pose_snapshot is None:
            empty_traj = Trajectory()
            logger.debug(
                "Drive request received with no pose snapshot available "
                "(poses list length: %s). Returning empty trajectory",
                len(session.poses),
            )
            return DriveResponse(
                trajectory=empty_traj,
            )

        selected_model_type = self._resolve_model_type_for_request(request)
        planner_context = self._resolve_planner_context_for_request(request)
        # Validate model eagerly to fail this request immediately if unavailable.
        self._get_model(selected_model_type)

        future: asyncio.Future[ModelPrediction] = self._loop.create_future()
        job = DriveJob(
            session_id=request.session_uuid,
            session=session,
            command=session.current_command,
            model_type=selected_model_type,
            planner_context=planner_context,
            pose=pose_snapshot,
            timestamp_us=request.time_now_us,
            result=future,
        )
        self._job_queue.put_nowait(job)

        prediction = await future

        # Convert model prediction to Alpasim trajectory format
        selected_model = self._get_model(job.model_type)
        alpasim_traj: Trajectory = self._convert_prediction_to_alpasim_trajectory(
            prediction,
            job.pose,
            job.timestamp_us,
            output_frequency_hz=selected_model.output_frequency_hz,
        )
        reasoning_text: str | None = prediction.reasoning_text

        debug_data = {
            "selected_model_type": job.model_type,
            "has_planner_context": job.planner_context is not None,
            "command": int(session.current_command),
            "command_name": session.current_command.name,
            "num_frames": {
                cam_id: cache.frame_count()
                for cam_id, cache in session.frame_caches.items()
            },
            "num_cameras": len(session.frame_caches),
            "num_poses": len(session.poses),
            "trajectory_points": len(alpasim_traj.poses),
            "reasoning_text": reasoning_text,
        }
        if prediction.debug_metadata is not None:
            debug_data["model_debug"] = prediction.debug_metadata
            debug_data.update(
                {
                    key: value
                    for key, value in prediction.debug_metadata.items()
                    if key not in debug_data
                }
            )
        debug_info = DriveResponse.DebugInfo(
            unstructured_debug_info=pickle.dumps(debug_data)
        )
        response = DriveResponse(trajectory=alpasim_traj, debug_info=debug_info)

        logger.debug("Returning drive response at time %s", request.time_now_us)
        return response

    def _convert_prediction_to_alpasim_trajectory(
        self,
        prediction: ModelPrediction,
        current_pose: PoseAtTime,
        time_now_us: int,
        output_frequency_hz: int,
    ) -> Trajectory:
        """Convert model prediction to Alpasim trajectory format.

        If the model provides headings, use them directly.
        Otherwise, compute headings from position deltas (existing behavior).

        Args:
            prediction: Model prediction with trajectory_xy and optional headings.
            current_pose: Current vehicle pose in local frame.
            time_now_us: Current time in microseconds.
            output_frequency_hz: Output waypoint frequency of the selected model.

        Returns:
            Alpasim Trajectory protobuf message.
        """
        trajectory = Trajectory()

        model_trajectory = prediction.trajectory_xy
        if model_trajectory is None or len(model_trajectory) == 0:
            return trajectory

        trajectory.poses.append(current_pose)

        curr_z = current_pose.pose.vec.z
        frequency_hz = output_frequency_hz
        time_delta_us = int(1_000_000 / frequency_hz)
        time_step = 1.0 / frequency_hz

        # Apply trajectory optimization in rig frame if enabled
        optimized_trajectory = model_trajectory
        if self._trajectory_optimizer is not None and len(model_trajectory) >= 2:
            # Add heading to create [N, 3] trajectory for optimizer
            rig_trajectory = add_heading_to_trajectory(model_trajectory)

            # Run optimization
            opt_cfg = self._cfg.trajectory_optimizer
            result = self._trajectory_optimizer.optimize(
                trajectory=rig_trajectory,
                time_step=time_step,
                vehicle_constraints=self._vehicle_constraints,
                retime_in_frenet=opt_cfg.retime_in_frenet,
                retime_alpha=opt_cfg.retime_alpha,
            )

            if result.success:
                # Extract x,y from optimized trajectory
                optimized_trajectory = result.trajectory[:, :2]
                logger.debug(
                    "Trajectory optimization succeeded: iterations=%s, cost=%.4f",
                    result.iterations,
                    result.final_cost,
                )
            else:
                logger.warning("Trajectory optimization failed: %s", result.message)

        # Convert rig offsets to local frame positions
        local_positions = _rig_est_offsets_to_local_positions(
            current_pose, optimized_trajectory
        )
        num_positions = local_positions.shape[0]

        if num_positions == 0:
            return trajectory

        # Pre-compute timestamps
        steps = np.arange(1, num_positions + 1, dtype=np.int64)
        timestamps_us = (time_now_us + steps * time_delta_us).tolist()

        # Transform model headings from rig frame to local frame
        current_yaw = _quat_to_yaw(current_pose.pose.quat)
        local_yaws = prediction.headings + current_yaw

        for local_xy, yaw, timestamp_us in zip(
            local_positions, local_yaws, timestamps_us, strict=True
        ):
            local_x, local_y = map(float, local_xy)

            trajectory.poses.append(
                PoseAtTime(
                    pose=Pose(
                        vec=Vec3(x=local_x, y=local_y, z=curr_z),
                        quat=_yaw_to_quat(float(yaw)),
                    ),
                    timestamp_us=timestamp_us,
                )
            )

        return trajectory

    @async_log_call
    async def shut_down(
        self, request: Empty, context: grpc.aio.ServicerContext
    ) -> Empty:
        logger.info("shut_down requested, scheduling deferred shutdown")
        # Schedule shutdown to happen after RPC completes to avoid CancelledError
        asyncio.create_task(self._deferred_shutdown())
        return Empty()

    async def _deferred_shutdown(self) -> None:
        """Shutdown the server and worker after the shut_down RPC completes.

        This deferred approach prevents the shut_down RPC from cancelling itself
        when stopping the server, which would result in asyncio.exceptions.CancelledError.
        """
        # Small delay to ensure the shut_down RPC response is sent first
        await asyncio.sleep(0.1)
        logger.info("Executing deferred shutdown")
        await self._grpc_server.stop(grace=None)
        await self.stop_worker()


async def serve(cfg: DriverConfig) -> None:
    """Start the gRPC server with the driver service."""
    server = grpc.aio.server()
    loop = asyncio.get_running_loop()

    service = EgoDriverService(
        cfg=cfg,
        loop=loop,
        grpc_server=server,
    )
    add_EgodriverServiceServicer_to_server(service, server)

    address = f"{cfg.host}:{cfg.port}"
    server.add_insecure_port(address)

    await server.start()
    external_ip = _get_external_ip()
    logger.info(
        "Starting %s driver on %s (external IP: %s:%d)",
        cfg.model.model_type,
        address,
        external_ip,
        cfg.port,
    )

    try:
        await server.wait_for_termination()
    finally:
        await service.stop_worker()


def _run_grpc_in_thread(cfg: DriverConfig, ready_event: threading.Event) -> None:
    """Run the gRPC server in a background thread.

    Used when the main thread is needed for GUI (e.g., ManualModel on macOS).

    Args:
        cfg: Driver configuration.
        ready_event: Event to signal when the service is initialized.
    """

    async def serve_with_signal() -> None:
        server = grpc.aio.server()
        loop = asyncio.get_running_loop()

        service = EgoDriverService(
            cfg=cfg,
            loop=loop,
            grpc_server=server,
        )
        add_EgodriverServiceServicer_to_server(service, server)

        address = f"{cfg.host}:{cfg.port}"
        server.add_insecure_port(address)

        await server.start()
        external_ip = _get_external_ip()
        logger.info(
            "Starting %s driver on %s (external IP: %s:%d)",
            cfg.model.model_type,
            address,
            external_ip,
            cfg.port,
        )

        # Signal that the service (and model) is ready
        ready_event.set()

        try:
            await server.wait_for_termination()
        finally:
            await service.stop_worker()

    asyncio.run(serve_with_signal())


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="driver",
)
def main(hydra_cfg: DriverConfig) -> None:
    """Main entry point for the driver service."""
    schema = OmegaConf.structured(DriverConfig)
    cfg = cast(DriverConfig, OmegaConf.merge(schema, hydra_cfg))

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        config_filename = f"{cfg.model.model_type}-driver.yaml"
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, config_filename), resolve=True)

    # For ManualModel, run the GUI on the main thread and gRPC in a background
    # thread. This is required on macOS (Cocoa), and we use the same approach
    # on Linux for consistency and simpler maintenance.
    if cfg.model.model_type == "manual":
        logger.info("Starting gRPC server in background thread (GUI mode)")

        ready_event = threading.Event()
        grpc_thread = threading.Thread(
            target=_run_grpc_in_thread,
            args=(cfg, ready_event),
            name="grpc-server",
            daemon=True,
        )
        grpc_thread.start()

        # Wait for the service (and ManualModel) to be created
        ready_event.wait(timeout=30.0)

        # Run pygame loop on main thread using the singleton GUI instance
        if ManualModel._gui_instance is not None:
            ManualModel._gui_instance.run_main_loop()
        else:
            logger.warning("ManualModel GUI not initialized, waiting for gRPC thread")
            grpc_thread.join()

        return

    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
