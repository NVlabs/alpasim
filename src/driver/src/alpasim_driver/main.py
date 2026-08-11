# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Unified driver implementation for Alpasim supporting multiple model backends."""

from __future__ import annotations

import functools
import logging
import os
import pickle
import queue
import socket
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib.metadata import version
from io import BytesIO
from typing import Any, Callable, cast

import hydra
import numpy as np
import torch
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_grpc.v0.common_pb2 import (
    DynamicState,
    Empty,
    PoseAtTime,
    SessionRequestStatus,
    Trajectory,
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
from alpasim_utils.geometry import Pose as GeometryPose
from alpasim_utils.geometry import Trajectory as GeometryTrajectory
from alpasim_utils.geometry import pose_from_grpc, pose_to_grpc_at_time
from omegaconf import OmegaConf
from PIL import Image
from torchvision.io import decode_jpeg

import grpc

from .frame_cache import FrameCache
from .models import DriveCommand
from .models.base import (
    BaseTrajectoryModel,
    CameraImages,
    ModelInputValidationError,
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


def _rig_est_waypoints_to_local_trajectory(
    positions_in_rig: np.ndarray,
    rotations_in_rig: np.ndarray,
    pose_local_to_rig_t0: GeometryPose,
    model_t0_us: int,
    waypoint_timestamps_us: np.ndarray,
) -> Trajectory:
    """Express rig-est waypoint poses in the local frame anchored at model t0.

    Args:
        positions_in_rig: (T, 3) waypoint positions in the rig-est frame.
        rotations_in_rig: (T, 3, 3) waypoint rotation matrices in the rig-est
            frame.
        pose_local_to_rig_t0: Rig pose in the local frame at model t0.
        model_t0_us: Exact planning time used to build the model inputs.
        waypoint_timestamps_us: Exact timestamps for the predicted waypoints.

    Returns:
        The trajectory in the local frame, led by the rig pose at model t0.
        Empty when there are no waypoints.
    """
    if len(positions_in_rig) == 0:
        return Trajectory()
    if len(waypoint_timestamps_us) != len(positions_in_rig):
        raise ValueError(
            "Waypoint timestamps and positions must have the same length, got "
            f"{len(waypoint_timestamps_us)} and {len(positions_in_rig)}."
        )

    waypoints_in_rig = np.tile(
        np.eye(4, dtype=np.float32), (len(positions_in_rig), 1, 1)
    )
    waypoints_in_rig[:, :3, :3] = rotations_in_rig
    waypoints_in_rig[:, :3, 3] = positions_in_rig

    trajectory = Trajectory(
        poses=[pose_to_grpc_at_time(pose_local_to_rig_t0, model_t0_us)]
    )
    for waypoint_in_rig, timestamp_us in zip(
        waypoints_in_rig, waypoint_timestamps_us, strict=True
    ):
        waypoint_in_local = pose_local_to_rig_t0 @ GeometryPose.from_se3(
            waypoint_in_rig
        )
        trajectory.poses.append(
            pose_to_grpc_at_time(waypoint_in_local, int(timestamp_us))
        )
    return trajectory


# Unique queue marker instructing the worker thread to flush and exit.
_SENTINEL_JOB = object()


@dataclass
class DriveJob:
    """Unit of work processed by the background inference worker."""

    session_id: str
    session: "Session"
    command: DriveCommand
    pose: PoseAtTime | None
    timestamp_us: int
    result: Future[ModelPrediction]


@dataclass
class Session:
    """Represents a driver session."""

    uuid: str
    seed: int
    debug_scene_id: str

    frame_caches: dict[str, FrameCache]
    rectification_cfg: dict[str, RectificationTargetConfig] | None = None
    rectification_camera_specs: dict[
        str, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera
    ] = field(default_factory=dict)
    rectifiers: dict[str, FthetaToPinholeRectifier] = field(default_factory=dict)
    rectifier_locks: dict[str, threading.Lock] = field(default_factory=dict, repr=False)
    poses: list[PoseAtTime] = field(default_factory=list)
    dynamic_states: list[tuple[int, DynamicState]] = field(default_factory=list)
    current_command: DriveCommand = DriveCommand.STRAIGHT  # Default to straight
    inference_count: int = 0
    # Plan the model selected in the most recent inference of this session, in
    # the local frame.  Used by models that select between multiple sampled
    # trajectories.  None until such a model returns one.
    last_selected_plan: GeometryTrajectory | None = None
    # Most recent route, with waypoints in the rig frame at its own timestamp.
    # None until the first route arrives, and for scenarios without a route.
    route: Route | None = None
    frames_trail_request_warned: bool = False

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
            camera_specs[camera_def.logical_id] = camera_def
            logger.debug(
                f"Available camera: {camera_def.logical_id}, "
                f"resolution: ({camera_def.intrinsics.resolution_h}, {camera_def.intrinsics.resolution_w}), "
                f"intrinsics: {camera_def.intrinsics}"
            )

        desired_cameras_logical_ids = set(cfg.inference.use_cameras)
        if not desired_cameras_logical_ids:
            raise ValueError("No cameras specified in inference configuration")

        missing_defs = desired_cameras_logical_ids - camera_specs.keys()
        if missing_defs:
            raise ValueError(
                f"Requested cameras {sorted(missing_defs)} are missing from the rollout spec"
            )

        rectification_camera_ids = set(cfg.rectification or {})
        unexpected_rectification = (
            rectification_camera_ids - desired_cameras_logical_ids
        )
        if unexpected_rectification:
            raise ValueError(
                "Rectification configured for cameras not used by inference: "
                f"{sorted(unexpected_rectification)}"
            )

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
            rectification_cfg=cfg.rectification,
            rectification_camera_specs={
                logical_id: camera_specs[logical_id]
                for logical_id in rectification_camera_ids
            },
            rectifier_locks={
                logical_id: threading.Lock() for logical_id in rectification_camera_ids
            },
        )

        return session

    def add_image(
        self,
        logical_id: str,
        image_tensor: np.ndarray | torch.Tensor,
        timestamp_us: int,
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

    def rectify_image(self, logical_id: str, image: Image.Image) -> Image.Image:
        """Rectify an f-theta image when this camera explicitly opts in."""
        if self.rectification_cfg is None or logical_id not in self.rectification_cfg:
            return image

        with self.rectifier_locks[logical_id]:
            rectifier = self.rectifiers.get(logical_id)
            if rectifier is None:
                rectifier = build_ftheta_rectifier_for_resolution(
                    camera_proto=self.rectification_camera_specs[logical_id],
                    target_cfg=self.rectification_cfg[logical_id],
                    source_resolution_hw=(image.height, image.width),
                )
                self.rectifiers[logical_id] = rectifier
                logger.info(
                    "Enabled f-theta-to-pinhole rectification for %s at %sx%s",
                    logical_id,
                    image.width,
                    image.height,
                )

        return Image.fromarray(rectifier.rectify(np.asarray(image)))

    def add_egoposes(self, egoposes: Trajectory) -> None:
        """Add rig-est pose observations in the local frame."""
        self.poses.extend(egoposes.poses)
        self.poses = sorted(self.poses, key=lambda pose: pose.timestamp_us)
        logger.debug(f"poses: {self.poses}")

    def add_dynamic_state(
        self, timestamp_us: int, dynamic_state: DynamicState | None
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
        command_distance_threshold: float | None = None,
        min_lookahead_distance: float | None = None,
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


def log_call(func: Callable) -> Callable:
    """Helper to add logging for gRPC calls."""

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            logger.debug("Calling %s", func.__name__)
            return func(*args, **kwargs)
        except Exception:  # pragma: no cover - logging assistance
            logger.exception("Exception in %s", func.__name__)
            raise

    return wrapped


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
    ) -> None:
        """Initialize the Ego Driver service.

        Sets up the model backend, and starts a background
        worker thread for batched inference processing.

        Args:
            cfg: Hydra configuration containing model paths and inference settings
        """

        # Private members
        self._cfg = cfg

        # Determine device
        self._device = torch.device(
            cfg.model.device if torch.cuda.is_available() else "cpu"
        )

        # Create model using factory
        self._model = _create_model(
            cfg.model,
            self._device,
            camera_ids=cfg.inference.use_cameras,
            context_length=cfg.inference.context_length,
            output_frequency_hz=cfg.inference.output_frequency_hz,
        )

        decode_device = cfg.model.image_decode_device
        if decode_device not in ("cpu", "cuda"):
            raise ValueError(
                f"Unknown image_decode_device {decode_device!r}, expected cpu or cuda"
            )
        if decode_device == "cuda" and cfg.rectification:
            raise ValueError(
                "Rectification runs on host images, so it cannot be combined with "
                "image_decode_device=cuda"
            )

        # Get context length from model or config override
        self._context_length = (
            cfg.inference.context_length
            if cfg.inference.context_length is not None
            else self._model.context_length
        )

        logger.info(
            "Initialized %s model with %d cameras, context_length=%d",
            cfg.model.model_type,
            self._model.num_cameras,
            self._context_length,
        )

        self._max_batch_size = cfg.inference.max_batch_size
        self._job_queue: queue.Queue[DriveJob | object] = queue.Queue()
        self._worker_stop = threading.Event()
        self._worker_lifecycle_lock = threading.Lock()
        self._worker_thread = threading.Thread(
            target=self._worker_main,
            name="ego-driver-worker",
            daemon=True,
        )
        self._sessions: dict[str, Session] = {}
        self._sessions_lock = threading.Lock()

        # Initialize trajectory optimizer if enabled
        self._trajectory_optimizer: TrajectoryOptimizer | None = None
        self._vehicle_constraints: VehicleConstraints | None = None
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

    def stop_worker(self) -> None:
        """Signal the worker thread to stop and wait for it to exit."""
        with self._worker_lifecycle_lock:
            if not self._worker_stop.is_set():
                self._worker_stop.set()
                self._job_queue.put_nowait(_SENTINEL_JOB)
        if self._worker_thread.is_alive():
            self._worker_thread.join()

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
                    pending_job.result.set_exception(exc)
            else:
                logger.debug("Inference batch succeeded")
                for pending_job, response in zip(batch, responses, strict=True):
                    pending_job.result.set_result(response)

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
            leftover.result.cancel()

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

    def _prepare_camera_images(self, session: Session) -> CameraImages:
        """Collect raw images from frame caches for all cameras.

        Returns dict mapping camera_id to list of CameraFrame tuples.
        List length equals context_length.
        """
        camera_images: CameraImages = {}

        for cam_id in self._model.camera_ids:
            frame_cache = session.frame_caches[cam_id]
            entries = frame_cache.latest_frame_entries(self._context_length)
            camera_images[cam_id] = [(e.timestamp_us, e.image) for e in entries]

        return camera_images

    def _maybe_save_debug_image(
        self,
        frame: np.ndarray | torch.Tensor,
        scene_id: str,
        logical_id: str,
        timestamp_us: int,
    ) -> None:
        """Save the HWC frame supplied to the model for debugging."""

        if not self._cfg.plot_debug_images:
            return

        if not self._cfg.output_dir:
            logger.warning("Output directory is not set; skipping debug image dump")
            return

        session_folder = os.path.join(self._cfg.output_dir, scene_id, "debug_images")
        os.makedirs(session_folder, exist_ok=True)

        filename = f"{timestamp_us}_{logical_id}.png"
        output_path = os.path.join(session_folder, filename)
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        Image.fromarray(frame).save(output_path)

    def _run_batch(self, batch: list[DriveJob]) -> list[ModelPrediction]:
        """Run inference for a batch of jobs using the model abstraction.

        Builds a PredictionInput per job and delegates to predict_batch(),
        which models can override for GPU-level batching.
        """
        inputs = []
        for job in batch:
            speed, acceleration = self._get_speed_and_acceleration(job.session)
            inference_seed = job.session.seed + job.session.inference_count
            job.session.inference_count += 1
            camera_images = self._prepare_camera_images(job.session)
            self._warn_once_if_frames_trail_request(job, camera_images)
            inputs.append(
                PredictionInput(
                    camera_images=camera_images,
                    command=job.command,
                    speed=speed,
                    acceleration=acceleration,
                    ego_pose_history=job.session.poses,
                    inference_seed=inference_seed,
                    previous_plan=job.session.last_selected_plan,
                    route=job.session.route,
                )
            )

        predictions = self._model.predict_batch(inputs)

        for job, prediction in zip(batch, predictions, strict=True):
            # Carry the plan into the next inference of this session so the model
            # can keep its trajectory choice consistent across planning cycles.
            job.session.last_selected_plan = prediction.selected_plan

        return predictions

    def _warn_once_if_frames_trail_request(
        self, job: DriveJob, camera_images: CameraImages
    ) -> None:
        """Report a gap between the newest camera frame and the request time.

        Models predict from the newest camera frame. Alpamayo predictions carry
        that frame's exact t0 and interpolated pose through response conversion;
        older plugins without explicit reference metadata retain the historical
        request-time fallback. Report any gap so fallback plugins cannot silently
        anchor a trajectory at a different time than the frame they observed.
        """
        if job.session.frames_trail_request_warned:
            return

        # Models that drive without cameras have nothing to compare.
        newest_frame_us = max(
            (
                timestamp_us
                for frames in camera_images.values()
                for timestamp_us, _ in frames
            ),
            default=job.timestamp_us,
        )
        if newest_frame_us == job.timestamp_us:
            return

        job.session.frames_trail_request_warned = True
        logger.warning(
            "Newest camera frame is %+dus from the drive request "
            "(frame=%d, request=%d). Predictions with explicit model-t0 metadata "
            "retain the frame anchor; fallback plugins use the request anchor.",
            newest_frame_us - job.timestamp_us,
            newest_frame_us,
            job.timestamp_us,
        )

    @log_call
    def start_session(
        self, request: DriveSessionRequest, context: grpc.ServicerContext
    ) -> SessionRequestStatus:
        with self._sessions_lock:
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

    @log_call
    def close_session(
        self, request: DriveSessionCloseRequest, context: grpc.ServicerContext
    ) -> Empty:
        with self._sessions_lock:
            logger.info(f"Closing session {request.session_uuid}")
            del self._sessions[request.session_uuid]
        return Empty()

    @log_call
    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        driver_version = version("alpasim_driver")
        model_type = self._cfg.model.model_type
        return VersionId(
            version_id=f"{model_type}-driver-{driver_version}",
            git_hash="unknown",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    @log_call
    def submit_image_observation(
        self, request: RolloutCameraImage, context: grpc.ServicerContext
    ) -> Empty:
        grpc_image = request.camera_image
        session = self._sessions[request.session_uuid]
        if grpc_image.logical_id not in session.frame_caches:
            raise ValueError(f"Camera {grpc_image.logical_id} not in desired cameras")

        if self._cfg.model.image_decode_device == "cuda":
            # nvJPEG straight onto the inference device, so the model's
            # preprocessing resizes there.  Permuting to HWC costs nothing, it
            # only relabels the strides, and it keeps one frame layout.
            buffer = torch.frombuffer(
                bytearray(grpc_image.image_bytes), dtype=torch.uint8
            )
            frame = decode_jpeg(buffer, device=self._device).permute(1, 2, 0)
        else:
            image = Image.open(BytesIO(grpc_image.image_bytes))
            frame = np.array(session.rectify_image(grpc_image.logical_id, image))

        height, width = frame.shape[0], frame.shape[1]
        minimum = self._model.MIN_FRAME_HW
        if minimum is not None and (height < minimum[0] or width < minimum[1]):
            # Small renders do not fail on their own: Alpamayo's vision
            # processor fits frames to a pixel budget that a 320x512 render
            # already meets, so it passes through untouched and costs vision
            # tokens with no trace in the logs.
            raise ValueError(
                f"Camera {grpc_image.logical_id} renders {height}x{width}, below "
                f"the {minimum[0]}x{minimum[1]} this model needs."
            )

        self._maybe_save_debug_image(
            frame,
            session.debug_scene_id,
            grpc_image.logical_id,
            grpc_image.frame_end_us,
        )
        session.add_image(
            grpc_image.logical_id,
            frame,
            grpc_image.frame_end_us,
        )

        return Empty()

    @log_call
    def submit_egomotion_observation(
        self, request: RolloutEgoTrajectory, context: grpc.ServicerContext
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

    @log_call
    def submit_route(
        self, request: RouteRequest, context: grpc.ServicerContext
    ) -> Empty:
        logger.debug("submit_route: waypoint count=%s", len(request.route.waypoints))
        session = self._sessions[request.session_uuid]
        session.route = request.route
        if self._cfg.route is not None:
            session.update_command_from_route(
                request.route,
                self._cfg.route.use_waypoint_commands,
                self._cfg.route.command_distance_threshold,
                self._cfg.route.min_lookahead_distance,
            )
        else:
            session.update_command_from_route(
                request.route,
                use_waypoint_commands=False,
            )
        return Empty()

    @log_call
    def submit_recording_ground_truth(
        self, request: GroundTruthRequest, context: grpc.ServicerContext
    ) -> Empty:
        logger.debug("Ground truth received but not used by driver")
        return Empty()

    def _check_frames_ready(self, session: Session) -> bool:
        """Check if all cameras have enough frames for inference."""
        return session.all_cameras_ready()

    @log_call
    def drive(
        self, request: DriveRequest, context: grpc.ServicerContext
    ) -> DriveResponse:
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

        future: Future[ModelPrediction] = Future()
        job = DriveJob(
            session_id=request.session_uuid,
            session=session,
            command=session.current_command,
            pose=pose_snapshot,
            timestamp_us=request.time_now_us,
            result=future,
        )
        with self._worker_lifecycle_lock:
            if self._worker_stop.is_set():
                if context is not None:
                    context.abort(
                        grpc.StatusCode.UNAVAILABLE,
                        "Driver inference worker is stopping",
                    )
                raise RuntimeError("Driver inference worker is stopping")
            self._job_queue.put_nowait(job)

        try:
            prediction = future.result()
        except ModelInputValidationError as exc:
            logger.error("Driver input validation failed: %s", exc)
            if context is not None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

        # Convert model prediction to Alpasim trajectory format
        alpasim_traj: Trajectory = self._convert_prediction_to_alpasim_trajectory(
            prediction, job.pose, job.timestamp_us
        )
        reasoning_text: str | None = prediction.reasoning_text

        debug_data = {
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
        debug_info = DriveResponse.DebugInfo(
            unstructured_debug_info=pickle.dumps(debug_data),
            sampled_trajectories=self._sampled_alpasim_trajectories(
                prediction, job.pose, job.timestamp_us
            ),
        )
        response = DriveResponse(trajectory=alpasim_traj, debug_info=debug_info)

        logger.debug("Returning drive response at time %s", request.time_now_us)
        return response

    def _convert_prediction_to_alpasim_trajectory(
        self,
        prediction: ModelPrediction,
        current_pose: PoseAtTime,
        time_now_us: int,
    ) -> Trajectory:
        """Convert the driven waypoints of a prediction to Alpasim trajectory format.

        Args:
            prediction: Model prediction with waypoint poses in the rig frame.
            current_pose: Current vehicle pose in local frame.
            time_now_us: Current time in microseconds.

        Returns:
            Alpasim Trajectory protobuf message.
        """
        positions = prediction.selected_positions
        frequency_hz = self._model.output_frequency_hz

        # Apply trajectory optimization in rig frame if enabled.  It moves the
        # waypoints within the ground plane and leaves their orientation to the
        # model, so only x and y are taken from its result.
        if self._trajectory_optimizer is not None and len(positions) >= 2:
            # Add heading to create [N, 3] trajectory for optimizer
            rig_trajectory = add_heading_to_trajectory(positions[:, :2])

            # Run optimization
            opt_cfg = self._cfg.trajectory_optimizer
            result = self._trajectory_optimizer.optimize(
                trajectory=rig_trajectory,
                time_step=1.0 / frequency_hz,
                vehicle_constraints=self._vehicle_constraints,
                retime_in_frenet=opt_cfg.retime_in_frenet,
                retime_alpha=opt_cfg.retime_alpha,
            )

            if result.success:
                positions = positions.copy()
                positions[:, :2] = result.trajectory[:, :2]
                logger.debug(
                    "Trajectory optimization succeeded: iterations=%s, cost=%.4f",
                    result.iterations,
                    result.final_cost,
                )
            else:
                logger.warning("Trajectory optimization failed: %s", result.message)

        pose_local_to_rig_t0, model_t0_us, waypoint_timestamps_us = (
            self._prediction_reference(prediction, current_pose, time_now_us)
        )
        return _rig_est_waypoints_to_local_trajectory(
            positions,
            prediction.selected_rotations,
            pose_local_to_rig_t0,
            model_t0_us,
            waypoint_timestamps_us,
        )

    def _prediction_reference(
        self,
        prediction: ModelPrediction,
        current_pose: PoseAtTime,
        time_now_us: int,
    ) -> tuple[GeometryPose, int, np.ndarray]:
        """Resolve the exact frame and timestamps for response conversion.

        Alpamayo models report the camera-derived t0 and its interpolated pose.
        Other model plugins retain the historical request-time fallback.
        """
        reference_fields = (
            prediction.model_t0_us,
            prediction.pose_local_to_rig_t0,
            prediction.waypoint_timestamps_us,
        )
        if all(value is None for value in reference_fields):
            model_t0_us = time_now_us
            step_us = int(1_000_000 / self._model.output_frequency_hz)
            waypoint_timestamps_us = (
                model_t0_us
                + np.arange(
                    1, prediction.candidate_positions.shape[1] + 1, dtype=np.uint64
                )
                * step_us
            )
            return (
                pose_from_grpc(current_pose.pose),
                model_t0_us,
                waypoint_timestamps_us,
            )
        if any(value is None for value in reference_fields):
            raise ValueError(
                "ModelPrediction must provide model_t0_us, "
                "pose_local_to_rig_t0, and waypoint_timestamps_us together."
            )

        assert prediction.model_t0_us is not None
        assert prediction.pose_local_to_rig_t0 is not None
        assert prediction.waypoint_timestamps_us is not None
        return (
            prediction.pose_local_to_rig_t0,
            int(prediction.model_t0_us),
            np.asarray(prediction.waypoint_timestamps_us, dtype=np.uint64),
        )

    def _sampled_alpasim_trajectories(
        self,
        prediction: ModelPrediction,
        current_pose: PoseAtTime,
        time_now_us: int,
    ) -> list[Trajectory]:
        """Convert every sampled candidate of a prediction to Alpasim trajectories.

        Reported alongside the driven trajectory so that scorers can measure the
        spread of the samples the model drew, e.g. minADE.
        """
        pose_local_to_rig_t0, model_t0_us, waypoint_timestamps_us = (
            self._prediction_reference(prediction, current_pose, time_now_us)
        )
        return [
            _rig_est_waypoints_to_local_trajectory(
                positions,
                rotations,
                pose_local_to_rig_t0,
                model_t0_us,
                waypoint_timestamps_us,
            )
            for positions, rotations in zip(
                prediction.candidate_positions,
                prediction.candidate_rotations,
                strict=True,
            )
        ]


def serve(cfg: DriverConfig, ready_event: threading.Event | None = None) -> None:
    """Start the gRPC server with the driver service.

    Args:
        cfg: Driver configuration.
        ready_event: Optional event to signal when the service is initialized.
            Used when the server runs in a background thread (GUI mode).
    """
    server = grpc.server(ThreadPoolExecutor())

    service = EgoDriverService(cfg=cfg)
    add_EgodriverServiceServicer_to_server(service, server)

    address = f"{cfg.host}:{cfg.port}"
    server.add_insecure_port(address)

    server.start()
    external_ip = _get_external_ip()
    logger.info(
        "Starting %s driver on %s (external IP: %s:%d)",
        cfg.model.model_type,
        address,
        external_ip,
        cfg.port,
    )

    if ready_event is not None:
        ready_event.set()

    try:
        server.wait_for_termination()
    finally:
        server_stop = server.stop(grace=None)
        service.stop_worker()
        server_stop.wait()


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
            target=serve,
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

    serve(cfg)


if __name__ == "__main__":
    main()
