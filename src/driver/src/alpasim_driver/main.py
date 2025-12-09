# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""VAM Driver implementation for Alpasim."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import pickle
import queue
import threading
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from importlib.metadata import version
from io import BytesIO
from typing import Any, Callable, Optional, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf.dictconfig
import omegaconf.listconfig
import torch
import torch.serialization
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
from omegaconf import OmegaConf
from PIL import Image
from vam.action_expert import VideoActionModelInference
from vam.datalib.transforms import NeuroNCAPTransform

import grpc
import grpc.aio

from .frame_cache import FrameCache, FrameEntry
from .rectification import (
    FthetaToPinholeRectifier,
    build_ftheta_rectifier_for_resolution,
)
from .schema import RectificationTargetConfig, VAMDriverConfig
from .trajectory_optimizer import (
    TrajectoryOptimizer,
    VehicleConstraints,
    add_heading_to_trajectory,
)

logger = logging.getLogger(__name__)


class DriveCommand(IntEnum):
    """Discrete high-level maneuver commands passed to the VAM."""

    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2


torch.serialization.add_safe_globals(
    # Let torch.load's safe unpickler recreate OmegaConf containers embedded in checkpoints.
    [
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
    ]
)


def load_inference_VAM(
    checkpoint_path: str,
    device: torch.device | str = "cuda",
    tempdir: Optional[str] = None,
) -> VideoActionModelInference:
    """Custom loader that handles PyTorch 2.6+ weights_only issue."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = ckpt["hyper_parameters"]["vam_conf"].copy()
    config.pop("_target_", None)
    config.pop("_recursive_", None)
    config["gpt_checkpoint_path"] = None
    config["action_checkpoint_path"] = None
    config["gpt_mup_base_shapes"] = None
    config["action_mup_base_shapes"] = None

    logging.info("Loading VAM checkpoint from %s", checkpoint_path)
    logging.info("VAM config: %s", config)

    vam = VideoActionModelInference(**config)
    state_dict = OrderedDict()
    for key, value in ckpt["state_dict"].items():
        state_dict[key.replace("vam.", "")] = value
    vam.load_state_dict(state_dict, strict=True)
    vam = vam.eval().to(device)
    return vam


def _format_trajs(trajs: torch.Tensor) -> np.ndarray:
    """Normalize VAM trajectory tensor shape to (T, 2)."""

    array = trajs.detach().float().cpu().numpy()
    while array.ndim > 2 and array.shape[0] == 1:
        array = array.squeeze(0)

    if array.ndim != 2:
        raise ValueError(f"Unexpected trajectory shape {array.shape}")

    return array


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


@dataclass
class DriveJob:
    """Unit of work processed by the background inference worker."""

    session_id: str
    session: "Session"
    command: DriveCommand
    pose: Optional[PoseAtTime]
    timestamp_us: int
    result: asyncio.Future[DriveResponse]


@dataclass
class Session:
    """Represents a VAM session."""

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
        cfg: VAMDriverConfig,
        context_length: int,
        subsample_factor: int = 1,
    ) -> Session:
        """Create a new VAM session."""
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

        # VAM model currently requires exactly one camera
        if len(cfg.inference.use_cameras) != 1:
            raise ValueError(
                f"VAM model requires exactly one camera, got {len(cfg.inference.use_cameras)}: "
                f"{cfg.inference.use_cameras}. Multi-camera support requires a different model."
            )

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

    def pending_frames_all_cameras(self) -> list[FrameEntry]:
        """Return pending frames across all cameras."""
        result = []
        for cache in self.frame_caches.values():
            result.extend(cache.pending_frames())
        return result

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
            return
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
        """Derive command from waypoints using VAM-style logic.
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

        target_waypoint = None
        for wp in route.waypoints:
            distance = np.hypot(wp.x, wp.y)

            if distance >= min_lookahead_distance:
                target_waypoint = wp
                break

        if target_waypoint is None:
            return

        dy_rig = target_waypoint.y  # already in rig frame (positive is left)

        if dy_rig > command_distance_threshold:
            self.current_command = DriveCommand.LEFT
        elif dy_rig < -command_distance_threshold:
            self.current_command = DriveCommand.RIGHT
        else:
            self.current_command = DriveCommand.STRAIGHT

        logger.debug(
            "Command: %s (lateral displacement: %.2fm)",
            self.current_command.name,
            dy_rig,
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


class VAMPolicyService(EgodriverServiceServicer):
    """VAM Policy service implementing the Alpasim ego driver interface."""

    def __init__(
        self,
        cfg: VAMDriverConfig,
        loop: asyncio.AbstractEventLoop,
        grpc_server: grpc.aio.Server,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the VAM Policy service.

        Sets up the VAM model, image tokenizer, preprocessing pipeline, and starts
        a background worker thread for batched inference processing.

        Args:
            cfg: Hydra configuration containing model paths and inference settings
            loop: Asyncio event loop for coordinating async operations and scheduling
                futures from the worker thread back to the async gRPC handlers
            grpc_server: gRPC server instance for service registration
            device: PyTorch device (CPU or CUDA) for model execution
            dtype: PyTorch data type for tensor operations
        """

        # Private members
        self._cfg = cfg
        self._loop = loop
        self._grpc_server = grpc_server

        self._device = device
        self._dtype = dtype
        self._image_tokenizer = torch.jit.load(
            cfg.model.tokenizer_path, map_location=self._device
        )
        self._image_tokenizer.to(self._device)
        self._image_tokenizer.eval()

        self._vam = load_inference_VAM(cfg.model.checkpoint_path, self._device)
        self._preproc_pipeline = NeuroNCAPTransform()
        self._use_autocast = self._device.type == "cuda"

        self._max_batch_size = cfg.inference.max_batch_size
        self._job_queue: queue.Queue[DriveJob | object] = queue.Queue()
        self._worker_stop = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_main,
            name="vam-policy-worker",
            daemon=True,
        )
        self._context_length = self._cfg.inference.context_length
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
                logger.debug("Running VAM batch of size %s", len(batch))
                responses = self._run_batch(batch)
                batch_count += 1
                total_items += len(batch)
                if batch_count % 100 == 0:
                    logger.info(
                        "VAM batches: %d processed, %d total items, avg size %.1f",
                        batch_count,
                        total_items,
                        total_items / batch_count,
                    )
            except Exception as exc:
                logger.exception("VAM batch failed")
                for pending_job in batch:
                    self._loop.call_soon_threadsafe(
                        pending_job.result.set_exception, exc
                    )
            else:
                logger.debug("VAM batch succeeded")
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

    def _tokenize_frames(self, batch: list[DriveJob]) -> None:
        """Tokenize frames for all cameras in the given batch."""
        frames_to_tokenize: list[FrameEntry] = []

        # Collect pending frames from all cameras across all jobs
        for job in batch:
            frames_to_tokenize.extend(job.session.pending_frames_all_cameras())

        if frames_to_tokenize:
            images = [frame.image for frame in frames_to_tokenize]
            token_batches = self._tokenize_batch(images)

            # Assign tokens back to frame entries
            for frame, tokens in zip(frames_to_tokenize, token_batches, strict=True):
                frame.tokens = tokens

    def _tokenize_batch(self, images: list[np.ndarray]) -> list[torch.Tensor]:
        if not images:
            return []

        tensors = [self._preproc_pipeline(image) for image in images]
        batch = torch.stack(tensors, dim=0).to(self._device)
        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=self._dtype)
            if self._use_autocast
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                token_batch = self._image_tokenizer(batch)
        return [tokens.detach().cpu() for tokens in token_batch]

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

    def _run_batch(self, batch: list[DriveJob]) -> list[DriveResponse]:
        self._tokenize_frames(batch)

        inputs: list[torch.Tensor] = []
        commands: list[DriveCommand] = []
        timestamps: list[int] = []

        for job in batch:
            # Collect tokens from all cameras (VAM validates single camera at session creation)
            camera_tokens = []
            for frame_cache in job.session.frame_caches.values():
                token_window = frame_cache.latest_token_window()
                camera_tokens.append(torch.stack(token_window, dim=0))  # (T, C, H, W)

            # # Concatenate along channel dimension: (T, N*C, H, W)
            # tensor = torch.cat(camera_tokens, dim=1)
            # For now assume a single camera
            if len(camera_tokens) != 1:
                raise ValueError(f"Expected 1 camera token, got {len(camera_tokens)}")
            tensor = camera_tokens[0]  # NOTE: Remove when we want more cameras!
            inputs.append(tensor)
            commands.append(job.command)
            timestamps.append(job.timestamp_us)

        visual_tokens = torch.stack(inputs, dim=0).to(self._device)  # (B, T, C, H, W)
        command_tensor = torch.tensor(  # (B, 1)
            [int(cmd) for cmd in commands], device=self._device, dtype=torch.long
        ).unsqueeze(-1)

        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=self._dtype)
            if self._use_autocast
            else nullcontext()
        )

        with torch.no_grad():
            with autocast_ctx:
                trajectories = self._vam(visual_tokens, command_tensor, self._dtype)
        trajectories = trajectories.detach().cpu()

        responses: list[DriveResponse] = []
        vam_trajectories: list[np.ndarray] = []
        alpasim_trajectories: list[Trajectory] = []

        for job, traj, now_us in zip(batch, trajectories, timestamps, strict=True):
            np_traj = _format_trajs(traj)
            alpasim_traj = self._convert_vam_trajectory_to_alpasim(
                np_traj, job.pose, now_us
            )
            responses.append(
                DriveResponse(
                    trajectory=alpasim_traj,
                )
            )
            vam_trajectories.append(np_traj)
            alpasim_trajectories.append(alpasim_traj)

        return responses

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

        logger.info(f"Starting VAM session {request.session_uuid}")
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
        return VersionId(
            version_id=f"vam-driver-{driver_version}",
            git_hash="unknown",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    def _resize_and_crop_image(self, image: Image.Image) -> Image.Image:
        img_w, img_h = image.size

        if img_h != self._cfg.inference.image_height:
            resize_factor = self._cfg.inference.image_height / img_h
            target_width = int(img_w * resize_factor)
            image = image.resize((target_width, self._cfg.inference.image_height))
            img_w, img_h = image.size

        if img_w > self._cfg.inference.image_width:
            left = (img_w - self._cfg.inference.image_width) // 2
            image = image.crop((left, 0, left + self._cfg.inference.image_width, img_h))
        elif img_w < self._cfg.inference.image_width:
            raise ValueError(
                f"Image width {img_w} is less than expected {self._cfg.inference.image_width}"
            )

        return image

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
        resized_rectified_image = self._resize_and_crop_image(rectified_image)
        session.add_image(
            grpc_image.logical_id,
            np.array(resized_rectified_image),
            grpc_image.frame_end_us,
        )

        return Empty()

    @async_log_call
    async def submit_egomotion_observation(
        self, request: RolloutEgoTrajectory, context: grpc.aio.ServicerContext
    ) -> Empty:
        session = self._sessions[request.session_uuid]

        # Guard: We currently assume a single pose per egomotion observation.
        # The dynamic_state has no timestamp and is assumed to correspond to
        # the (single) pose's timestamp. If multiple poses are sent, remove
        # this check and ensure proper handling of multi-pose trajectories.
        if len(request.trajectory.poses) != 1:
            raise ValueError(
                f"Expected exactly 1 pose in egomotion trajectory, got {len(request.trajectory.poses)}. "
                "The driver assumes dynamic_state corresponds to the single pose's timestamp. "
                "If multi-pose trajectories are intentional, update the driver to handle them correctly."
            )

        session.add_egoposes(request.trajectory)

        # Track dynamic state if provided (velocities, accelerations in rig frame)
        if request.HasField("dynamic_state") and request.trajectory.poses:
            # Use the latest pose timestamp for the dynamic state
            latest_timestamp_us = max(
                pose.timestamp_us for pose in request.trajectory.poses
            )
            session.add_dynamic_state(latest_timestamp_us, request.dynamic_state)

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
        logger.debug("Ground truth received but not used by VAM")
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

        future: asyncio.Future[DriveResponse] = self._loop.create_future()
        job = DriveJob(
            session_id=request.session_uuid,
            session=session,
            command=session.current_command,
            pose=pose_snapshot,
            timestamp_us=request.time_now_us,
            result=future,
        )
        self._job_queue.put_nowait(job)

        response = await future

        debug_data = {
            "command": int(session.current_command),
            "command_name": session.current_command.name,
            "num_frames": {
                cam_id: cache.frame_count()
                for cam_id, cache in session.frame_caches.items()
            },
            "num_cameras": len(session.frame_caches),
            "num_poses": len(session.poses),
            "trajectory_points": len(response.trajectory.poses),
        }
        response.debug_info.unstructured_debug_info = pickle.dumps(debug_data)

        logger.debug("Returning drive response at time %s", request.time_now_us)
        return response

    def _convert_vam_trajectory_to_alpasim(
        self,
        vam_trajectory: np.ndarray,  # Shape: (N, 2) - x,y offsets in rig frame
        current_pose: PoseAtTime,
        time_now_us: int,
    ) -> Trajectory:
        trajectory = Trajectory()
        trajectory.poses.append(current_pose)

        if vam_trajectory is None or len(vam_trajectory) == 0:
            return trajectory

        curr_z = current_pose.pose.vec.z
        frequency_hz = self._cfg.trajectory.frequency_hz
        time_delta_us = int(1_000_000 / frequency_hz)
        time_step = 1.0 / frequency_hz

        # Apply trajectory optimization in rig frame if enabled
        optimized_vam_trajectory = vam_trajectory
        if self._trajectory_optimizer is not None and len(vam_trajectory) >= 2:
            # Add heading to create [N, 3] trajectory for optimizer
            rig_trajectory = add_heading_to_trajectory(vam_trajectory)

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
                optimized_vam_trajectory = result.trajectory[:, :2]
                logger.debug(
                    "Trajectory optimization succeeded: iterations=%s, cost=%.4f",
                    result.iterations,
                    result.final_cost,
                )
            else:
                logger.warning("Trajectory optimization failed: %s", result.message)

        # Convert rig offsets to local frame positions
        local_positions = _rig_est_offsets_to_local_positions(
            current_pose, optimized_vam_trajectory
        )
        num_positions = local_positions.shape[0]

        if num_positions == 0:
            return trajectory

        # Pre-compute timestamps and XY deltas between consecutive positions.
        steps = np.arange(1, num_positions + 1, dtype=np.int64)
        timestamps_us = (time_now_us + steps * time_delta_us).tolist()

        previous_positions = np.vstack(
            (
                np.array(
                    [current_pose.pose.vec.x, current_pose.pose.vec.y], dtype=float
                ),
                local_positions[:-1],
            )
        )
        deltas = local_positions - previous_positions
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        yaws = np.arctan2(deltas[:, 1], deltas[:, 0])

        prev_quat = Quat()
        prev_quat.CopyFrom(current_pose.pose.quat)

        for local_xy, distance, yaw, timestamp_us in zip(
            local_positions,
            distances,
            yaws,
            timestamps_us,
            strict=True,
        ):
            local_x, local_y = map(float, local_xy)
            local_z = curr_z

            if distance > 1e-4:
                quat = _yaw_to_quat(float(yaw))
            else:
                quat = Quat()
                quat.CopyFrom(prev_quat)

            trajectory.poses.append(
                PoseAtTime(
                    pose=Pose(
                        vec=Vec3(x=local_x, y=local_y, z=local_z),
                        quat=quat,
                    ),
                    timestamp_us=timestamp_us,
                )
            )

            prev_quat = quat

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


async def serve(cfg: VAMDriverConfig) -> None:
    server = grpc.aio.server()
    loop = asyncio.get_running_loop()

    device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if cfg.model.dtype == "float16" else torch.float32

    service = VAMPolicyService(
        cfg=cfg,
        loop=loop,
        grpc_server=server,
        device=device,
        dtype=dtype,
    )
    add_EgodriverServiceServicer_to_server(service, server)

    address = f"{cfg.host}:{cfg.port}"
    server.add_insecure_port(address)

    await server.start()
    logger.info("Starting VAM driver on %s", address)

    try:
        await server.wait_for_termination()
    finally:
        await service.stop_worker()


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="vam_driver",
)
def main(cfg: VAMDriverConfig) -> None:
    schema = OmegaConf.structured(VAMDriverConfig)
    cfg: VAMDriverConfig = cast(VAMDriverConfig, OmegaConf.merge(schema, cfg))

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(
            cfg, os.path.join(cfg.output_dir, "vam-driver.yaml"), resolve=True
        )

    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
