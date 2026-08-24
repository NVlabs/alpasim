# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
import math
import os
import signal
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral, Real
from typing import Protocol

import numpy as np
import torch
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
from PIL import Image

import grpc

from .batch_worker import BatchPolicy, BatchWorker
from .navigation import DriveCommand, command_from_route, command_one_hot
from .policy import DiffusionDrivePolicy, InferenceInput, Prediction
from .preprocessing import CAMERA_IDS
from .trajectory import (
    CachedPlan,
    build_trajectory_from_plan,
    cached_plan_covers_query,
    make_cached_plan,
    yaw_from_quat,
)

LOGGER = logging.getLogger(__name__)

_RUNTIME_WRITE_DIR_DEFAULTS = {
    "HOME": "/tmp/home",
    "TMPDIR": "/tmp",
    "XDG_CACHE_HOME": "/tmp/.cache",
    "TORCH_HOME": "/tmp/torch",
    "HF_HOME": "/tmp/huggingface",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "CUDA_CACHE_PATH": "/tmp/nv",
    "NUMBA_CACHE_DIR": "/tmp/numba",
    "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions",
    "ALPASIM_DRIVER_LOG_DIR": "/run/alpasim-driver",
}


def _configure_runtime_write_dirs() -> None:
    for key, default in _RUNTIME_WRITE_DIR_DEFAULTS.items():
        os.environ.setdefault(key, default)
    for key in _RUNTIME_WRITE_DIR_DEFAULTS:
        path = os.environ[key]
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            if key != "ALPASIM_DRIVER_LOG_DIR":
                raise
            fallback = "/tmp/alpasim-driver"
            os.environ[key] = fallback
            os.makedirs(fallback, exist_ok=True)


def _configure_torch_threads() -> None:
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
    torch.set_num_interop_threads(
        max(1, int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")))
    )


def _pose_xy(pose: common_pb2.PoseAtTime) -> np.ndarray:
    return np.array([pose.pose.vec.x, pose.pose.vec.y], dtype=np.float64)


def _local_vector_to_rig(
    vector_xy: np.ndarray,
    reference_pose: common_pb2.PoseAtTime,
) -> np.ndarray:
    yaw = yaw_from_quat(reference_pose.pose.quat)
    c = math.cos(yaw)
    s = math.sin(yaw)
    local_to_rig = np.array([[c, s], [-s, c]], dtype=np.float64)
    return (local_to_rig @ vector_xy).astype(np.float32)


def _derive_dynamics_from_poses(
    poses: list[common_pb2.PoseAtTime],
) -> tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros(2, dtype=np.float32)
    if len(poses) < 2:
        return zeros.copy(), zeros.copy()

    previous, current = poses[-2:]
    dt_current_s = (current.timestamp_us - previous.timestamp_us) / 1_000_000.0
    if dt_current_s <= 1e-6:
        return zeros.copy(), zeros.copy()

    current_velocity_local = (_pose_xy(current) - _pose_xy(previous)) / dt_current_s
    velocity_rig = _local_vector_to_rig(current_velocity_local, current)
    if len(poses) < 3:
        return velocity_rig, zeros.copy()

    older = poses[-3]
    dt_previous_s = (previous.timestamp_us - older.timestamp_us) / 1_000_000.0
    if dt_previous_s <= 1e-6:
        return velocity_rig, zeros.copy()

    previous_velocity_local = (_pose_xy(previous) - _pose_xy(older)) / dt_previous_s
    midpoint_dt_s = 0.5 * (dt_previous_s + dt_current_s)
    acceleration_local = (
        current_velocity_local - previous_velocity_local
    ) / midpoint_dt_s
    acceleration_rig = _local_vector_to_rig(acceleration_local, current)
    return velocity_rig, acceleration_rig


@dataclass
class SessionCounters:
    diffusiondrive_inference: int = 0
    cached_plan: int = 0
    straight_fallback: int = 0
    dynamic_state_fallback: int = 0
    inference_error: int = 0


@dataclass
class SessionState:
    images: dict[str, OrderedDict[int, np.ndarray]] = field(
        default_factory=lambda: {camera_id: OrderedDict() for camera_id in CAMERA_IDS}
    )
    poses: list[common_pb2.PoseAtTime] = field(default_factory=list)
    dynamics: list[tuple[int, common_pb2.DynamicState]] = field(default_factory=list)
    command_one_hot: np.ndarray = field(
        default_factory=lambda: command_one_hot(DriveCommand.UNKNOWN)
    )
    cached_plan: CachedPlan | None = None
    last_inference_timestamp_us: int | None = None
    inference_inflight_timestamp_us: int | None = None
    random_seed: int = 0
    noise_index: int = 0
    counters: SessionCounters = field(default_factory=SessionCounters)

    def add_image(
        self,
        camera_id: str,
        timestamp_us: int,
        image: np.ndarray,
    ) -> None:
        if camera_id not in self.images:
            raise ValueError(f"unknown camera_id: {camera_id}")

        cache = self.images[camera_id]
        cache[int(timestamp_us)] = image
        retained = sorted(cache.items())[-4:]
        cache.clear()
        cache.update(retained)

    def synchronized_images(
        self,
        time_now_us: int,
    ) -> tuple[int, dict[str, np.ndarray]]:
        eligible = {
            camera_id: [
                timestamp
                for timestamp in self.images[camera_id]
                if timestamp <= time_now_us
            ]
            for camera_id in CAMERA_IDS
        }
        if any(not timestamps for timestamps in eligible.values()):
            raise LookupError("no complete synchronized camera set")

        common_timestamps = set(eligible[CAMERA_IDS[0]])
        for camera_id in CAMERA_IDS[1:]:
            common_timestamps.intersection_update(eligible[camera_id])
        if common_timestamps:
            timestamp_us = max(common_timestamps)
            return timestamp_us, {
                camera_id: self.images[camera_id][timestamp_us]
                for camera_id in CAMERA_IDS
            }

        latest = {camera_id: max(eligible[camera_id]) for camera_id in CAMERA_IDS}
        if max(latest.values()) - min(latest.values()) > 1_000:
            raise LookupError("camera timestamps differ by more than 1 ms")

        timestamp_us = max(latest.values())
        return timestamp_us, {
            camera_id: self.images[camera_id][latest[camera_id]]
            for camera_id in CAMERA_IDS
        }

    def add_egomotion(
        self,
        poses: list[common_pb2.PoseAtTime],
        dynamic_states: list[common_pb2.DynamicState],
    ) -> None:
        if dynamic_states and len(dynamic_states) != len(poses):
            raise ValueError(
                "dynamic_states must be empty or correspond 1:1 with poses"
            )

        incoming_poses = {int(pose.timestamp_us): pose for pose in poses}
        pose_by_time = {int(pose.timestamp_us): pose for pose in self.poses}
        pose_by_time.update(incoming_poses)
        self.poses = [
            pose_by_time[timestamp_us] for timestamp_us in sorted(pose_by_time)[-32:]
        ]

        state_by_time = dict(self.dynamics)
        for timestamp_us in incoming_poses:
            state_by_time.pop(timestamp_us, None)
        if dynamic_states:
            state_by_time.update(
                {
                    int(pose.timestamp_us): state
                    for pose, state in zip(poses, dynamic_states, strict=True)
                }
            )
        self.dynamics = [
            (timestamp_us, state_by_time[timestamp_us])
            for timestamp_us in sorted(state_by_time)[-32:]
        ]

    def ego_snapshot(
        self,
        time_now_us: int,
    ) -> tuple[common_pb2.PoseAtTime, np.ndarray, np.ndarray]:
        eligible_poses = [
            pose for pose in self.poses if pose.timestamp_us <= time_now_us
        ]
        if not eligible_poses:
            raise LookupError("no ego pose at or before Drive time")

        pose = eligible_poses[-1]
        state = dict(self.dynamics).get(int(pose.timestamp_us))
        if state is not None:
            velocity = np.array(
                [state.linear_velocity.x, state.linear_velocity.y],
                dtype=np.float32,
            )
            acceleration = np.array(
                [state.linear_acceleration.x, state.linear_acceleration.y],
                dtype=np.float32,
            )
            return pose, velocity, acceleration

        self.counters.dynamic_state_fallback += 1
        velocity, acceleration = _derive_dynamics_from_poses(eligible_poses)
        return pose, velocity, acceleration


class PredictionWorker(Protocol):
    def predict(
        self,
        request: InferenceInput,
        timeout: float | None = None,
    ) -> Prediction: ...


class PolicyHandleLike(Protocol):
    def start(self) -> None: ...

    def wait_ready(self, timeout: float | None) -> bool: ...

    def load_error(self) -> BaseException | None: ...

    def worker(self) -> PredictionWorker | None: ...

    def stop(self) -> None: ...


class PolicyHandle:
    def __init__(
        self,
        *,
        loader: Callable[[], BatchPolicy],
        max_batch_size: int = 1,
        batch_window_s: float = 0.002,
    ) -> None:
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, Integral)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if (
            isinstance(batch_window_s, bool)
            or not isinstance(batch_window_s, Real)
            or not math.isfinite(batch_window_s)
            or batch_window_s < 0
        ):
            raise ValueError("batch_window_s must be non-negative")

        self._loader = loader
        self._max_batch_size = int(max_batch_size)
        self._batch_window_s = float(batch_window_s)
        self._lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._ready = threading.Event()
        self._worker: BatchWorker | None = None
        self._load_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None or self._stop_requested:
                return
            self._thread = threading.Thread(
                target=self._load,
                name="diffusiondrive-policy-loader",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def wait_ready(self, timeout: float | None) -> bool:
        return self._ready.wait(timeout)

    def load_error(self) -> BaseException | None:
        with self._lock:
            return self._load_error

    def worker(self) -> BatchWorker | None:
        with self._lock:
            if self._stop_requested:
                return None
            return self._worker

    def stop(self) -> None:
        with self._stop_lock:
            with self._lock:
                self._stop_requested = True
                worker = self._worker
            if worker is None:
                return

            worker.stop()
            with self._lock:
                if self._worker is worker:
                    self._worker = None

    def _load(self) -> None:
        try:
            with self._lock:
                if self._stop_requested:
                    return
            policy = self._loader()
            with self._lock:
                if self._stop_requested:
                    return

            worker = BatchWorker(
                policy,
                max_batch_size=self._max_batch_size,
                batch_window_s=self._batch_window_s,
            )
            worker.start()
            with self._lock:
                self._worker = worker
                if self._stop_requested:
                    publish = False
                else:
                    publish = True
            if not publish:
                self.stop()
        except BaseException as exc:
            with self._lock:
                self._load_error = exc
            LOGGER.exception("DiffusionDrive policy load failed")
        finally:
            self._ready.set()


def _context_timeout(context: grpc.ServicerContext) -> float | None:
    time_remaining = getattr(context, "time_remaining", None)
    if not callable(time_remaining):
        return None
    remaining = time_remaining()
    if remaining is None:
        return None
    try:
        timeout = float(remaining)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(timeout) or timeout >= threading.TIMEOUT_MAX:
        return None
    return max(0.0, timeout)


class NavsimDiffusionDriveDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self, policy_handle: PolicyHandleLike) -> None:
        self._policy_handle = policy_handle
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def start_session(
        self,
        request: egodriver_pb2.DriveSessionRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.SessionRequestStatus:
        available = {
            camera.logical_id
            for camera in request.rollout_spec.vehicle.available_cameras
        }
        missing = sorted(set(CAMERA_IDS) - available)
        if missing:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"missing required NAVSIM cameras: {missing}",
            )
            raise AssertionError("unreachable")

        with self._lock:
            previous = self._sessions.get(request.session_uuid)
            if previous is not None:
                previous.inference_inflight_timestamp_us = None
            self._sessions[request.session_uuid] = SessionState(
                random_seed=int(request.random_seed)
            )
        LOGGER.info("started session %s", request.session_uuid)
        return common_pb2.SessionRequestStatus()

    def close_session(
        self,
        request: egodriver_pb2.DriveSessionCloseRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        with self._lock:
            session = self._sessions.pop(request.session_uuid, None)
            if session is not None:
                session.inference_inflight_timestamp_us = None
        LOGGER.info("closed session %s", request.session_uuid)
        return common_pb2.Empty()

    def submit_image_observation(
        self,
        request: egodriver_pb2.RolloutCameraImage,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        session = self._get_session(request.session_uuid, context)
        grpc_image = request.camera_image
        if grpc_image.logical_id not in CAMERA_IDS:
            return common_pb2.Empty()

        try:
            with Image.open(BytesIO(grpc_image.image_bytes)) as decoded:
                if decoded.size != (1920, 1080):
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"{grpc_image.logical_id} must decode to 1920x1080 RGB; "
                        f"got size {decoded.size}",
                    )
                    raise AssertionError("unreachable")
                image = np.asarray(decoded.convert("RGB")).copy()
        except (Image.DecompressionBombError, OSError, ValueError) as exc:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"failed to decode {grpc_image.logical_id}: {exc}",
            )
            raise AssertionError("unreachable") from exc
        if image.shape != (1080, 1920, 3):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"{grpc_image.logical_id} must decode to 1920x1080 RGB; "
                f"got {image.shape}",
            )
            raise AssertionError("unreachable")
        image.setflags(write=False)

        with self._lock:
            if self._sessions.get(request.session_uuid) is session:
                session.add_image(
                    grpc_image.logical_id,
                    int(grpc_image.frame_end_us),
                    image,
                )
        return common_pb2.Empty()

    def submit_egomotion_observation(
        self,
        request: egodriver_pb2.RolloutEgoTrajectory,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        session = self._get_session(request.session_uuid, context)
        poses = list(request.trajectory.poses)
        dynamic_states = list(request.dynamic_states)
        try:
            with self._lock:
                if self._sessions.get(request.session_uuid) is session:
                    session.add_egomotion(poses, dynamic_states)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise AssertionError("unreachable") from exc
        return common_pb2.Empty()

    def submit_route(
        self,
        request: egodriver_pb2.RouteRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        session = self._get_session(request.session_uuid, context)
        command = command_from_route(request.route)
        with self._lock:
            if self._sessions.get(request.session_uuid) is session:
                session.command_one_hot = command
        return common_pb2.Empty()

    def submit_recording_ground_truth(
        self,
        request: egodriver_pb2.GroundTruthRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        self._get_session(request.session_uuid, context)
        return common_pb2.Empty()

    def drive(
        self,
        request: egodriver_pb2.DriveRequest,
        context: grpc.ServicerContext,
    ) -> egodriver_pb2.DriveResponse:
        session = self._get_session(request.session_uuid, context)
        time_now_us = int(request.time_now_us)
        time_query_us = int(request.time_query_us)
        worker = self._policy_handle.worker()

        with self._lock:
            try:
                current_pose, current_velocity, current_acceleration = (
                    session.ego_snapshot(time_now_us)
                )
            except LookupError as exc:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
                raise AssertionError("unreachable") from exc

            image_timestamp_us: int | None = None
            inference_input: InferenceInput | None = None
            inference_pose: common_pb2.PoseAtTime | None = None
            try:
                image_timestamp_us, images = session.synchronized_images(time_now_us)
                if image_timestamp_us == current_pose.timestamp_us:
                    inference_pose = current_pose
                    velocity = current_velocity
                    acceleration = current_acceleration
                else:
                    inference_pose, velocity, acceleration = session.ego_snapshot(
                        image_timestamp_us
                    )
                inference_input = InferenceInput(
                    images=dict(images),
                    command_one_hot=session.command_one_hot.copy(),
                    velocity_xy=velocity.copy(),
                    acceleration_xy=acceleration.copy(),
                    noise_seed=session.random_seed,
                    noise_index=session.noise_index,
                )
            except LookupError:
                inference_input = None

            cached_before = session.cached_plan
            timestamp_is_new = image_timestamp_us is not None and (
                session.last_inference_timestamp_us is None
                or image_timestamp_us > session.last_inference_timestamp_us
            )
            should_infer = (
                worker is not None
                and inference_input is not None
                and inference_pose is not None
                and timestamp_is_new
                and session.inference_inflight_timestamp_us is None
            )
            if should_infer:
                session.last_inference_timestamp_us = image_timestamp_us
                session.inference_inflight_timestamp_us = image_timestamp_us
                session.noise_index += 1
                reserved_timestamp_us = image_timestamp_us
            else:
                reserved_timestamp_us = None

        fresh_plan: CachedPlan | None = None
        inference_error: Exception | None = None
        if reserved_timestamp_us is not None:
            assert worker is not None
            assert inference_input is not None
            assert inference_pose is not None
            try:
                try:
                    prediction = worker.predict(
                        inference_input,
                        timeout=_context_timeout(context),
                    )
                    fresh_plan = make_cached_plan(
                        reserved_timestamp_us,
                        inference_pose,
                        prediction.trajectory,
                    )
                except Exception as exc:
                    inference_error = exc
                    LOGGER.exception(
                        "DiffusionDrive inference failed for session %s timestamp_us=%d",
                        request.session_uuid,
                        reserved_timestamp_us,
                    )
            finally:
                with self._lock:
                    if session.inference_inflight_timestamp_us == reserved_timestamp_us:
                        session.inference_inflight_timestamp_us = None

        counter_values: tuple[int, int, int, int, int] | None = None
        with self._lock:
            active_session = self._sessions.get(request.session_uuid)
            if active_session is session:
                if inference_error is not None:
                    session.counters.inference_error += 1
                if fresh_plan is not None:
                    session.cached_plan = fresh_plan
                    session.counters.diffusiondrive_inference += 1

                response_plan = session.cached_plan
                if fresh_plan is None:
                    if cached_plan_covers_query(
                        response_plan, time_now_us, time_query_us
                    ):
                        session.counters.cached_plan += 1
                    else:
                        session.counters.straight_fallback += 1
                counter_values = (
                    session.counters.diffusiondrive_inference,
                    session.counters.cached_plan,
                    session.counters.straight_fallback,
                    session.counters.dynamic_state_fallback,
                    session.counters.inference_error,
                )
            else:
                response_plan = fresh_plan if fresh_plan is not None else cached_before

        trajectory = build_trajectory_from_plan(
            response_plan,
            current_pose,
            time_now_us,
            time_query_us,
            fallback_speed_mps=float(np.linalg.norm(current_velocity)),
        )
        if counter_values is not None:
            LOGGER.info(
                "session=%s diffusiondrive_inference=%d cached_plan=%d "
                "straight_fallback=%d dynamic_state_fallback=%d "
                "inference_error=%d",
                request.session_uuid,
                *counter_values,
            )
        return egodriver_pb2.DriveResponse(trajectory=trajectory)

    def get_version(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.VersionId:
        timeout = _context_timeout(context)
        if timeout is None:
            timeout = 30.0
        else:
            timeout = min(timeout, 30.0)
        if not self._policy_handle.wait_ready(timeout):
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "DiffusionDrive policy readiness timed out",
            )
            raise AssertionError("unreachable")

        load_error = self._policy_handle.load_error()
        if load_error is not None:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"DiffusionDrive policy load failed: {load_error}",
            )
            raise AssertionError("unreachable")
        if self._policy_handle.worker() is None:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "DiffusionDrive policy reported ready without a batch worker",
            )
            raise AssertionError("unreachable")

        return common_pb2.VersionId(
            version_id="simscale-diffusiondrive-navhard-e2e",
            git_hash=os.environ.get("DIFFUSIONDRIVE_GIT_HASH", "local"),
            grpc_api_version=API_VERSION_MESSAGE,
        )

    def stop(self) -> None:
        self._policy_handle.stop()

    def _get_session(
        self,
        session_uuid: str,
        context: grpc.ServicerContext,
    ) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_uuid)
        if session is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {session_uuid}")
            raise AssertionError("unreachable")
        return session


def main() -> None:
    _configure_runtime_write_dirs()
    _configure_torch_threads()
    logging.basicConfig(
        level=os.environ.get("ALPASIM_DRIVER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    host = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0")
    port = int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    checkpoint_path = os.environ.get(
        "DIFFUSIONDRIVE_CHECKPOINT_PATH",
        "/app/assets/diffusiondrive/diffusiondrive_sim_navhard.ckpt",
    )
    device = os.environ.get("DIFFUSIONDRIVE_DEVICE", "cuda")
    max_batch_size = int(os.environ.get("DIFFUSIONDRIVE_MAX_BATCH_SIZE", "1"))
    batch_window_s = (
        float(os.environ.get("DIFFUSIONDRIVE_BATCH_WINDOW_MS", "2")) / 1000.0
    )

    policy_handle = PolicyHandle(
        loader=lambda: DiffusionDrivePolicy(checkpoint_path, device=device),
        max_batch_size=max_batch_size,
        batch_window_s=batch_window_s,
    )
    service = NavsimDiffusionDriveDriver(policy_handle)
    grpc_workers = max(
        1,
        int(os.environ.get("ALPASIM_DRIVER_GRPC_WORKERS", "4")),
    )
    server = grpc.server(ThreadPoolExecutor(max_workers=grpc_workers))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(service, server)

    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        server.stop(grace=0.0)
        service.stop()
        raise RuntimeError(f"failed to bind {host}:{port}")

    def request_stop(signum: int, frame: object) -> None:
        LOGGER.info("received signal %s, stopping", signum)
        server.stop(grace=0.0)
        service.stop()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        server.start()
        LOGGER.info(
            "SimScale DiffusionDrive driver listening on %s:%d",
            host,
            bound_port,
        )
        policy_handle.start()
        server.wait_for_termination()
    finally:
        server.stop(grace=0.0)
        service.stop()


if __name__ == "__main__":
    main()
