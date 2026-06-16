#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""AlpaSim e2e challenge driver backed by VAVAM."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
import torch
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc, sensorsim_pb2
from PIL import Image

import grpc

from .rectification import (
    FthetaToPinholeRectifier,
    RectificationTargetConfig,
    build_ftheta_rectifier_for_resolution,
)
from .trajectory import CachedPlan, build_trajectory_from_plan, make_cached_plan

_RUNTIME_WRITE_DIR_DEFAULTS = {
    "XDG_CACHE_HOME": "/tmp/.cache",
    "TORCH_HOME": "/tmp/torch",
    "HF_HOME": "/tmp/huggingface",
    "TRANSFORMERS_CACHE": "/tmp/huggingface/transformers",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "CUDA_CACHE_PATH": "/tmp/nv",
    "NUMBA_CACHE_DIR": "/tmp/numba",
    "ALPASIM_DRIVER_LOG_DIR": "/run/alpasim-driver",
}
for _key, _value in _RUNTIME_WRITE_DIR_DEFAULTS.items():
    os.environ.setdefault(_key, _value)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")

LOGGER = logging.getLogger("vavam_challenge_driver")


def _configure_torch_threads() -> None:
    torch.set_num_threads(max(1, int(os.environ["TORCH_NUM_THREADS"])))
    torch.set_num_interop_threads(max(1, int(os.environ["TORCH_NUM_INTEROP_THREADS"])))


@dataclass
class SessionState:
    camera_id: str | None = None
    latest_pose: common_pb2.PoseAtTime | None = None
    latest_image: np.ndarray | None = None
    latest_image_timestamp_us: int = 0
    poses: list[common_pb2.PoseAtTime] = field(default_factory=list)
    dynamic_states: list[tuple[int, common_pb2.DynamicState]] = field(
        default_factory=list
    )
    command: int = 2
    cached_plan: CachedPlan | None = None
    camera_specs: dict[str, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera] = (
        field(default_factory=dict)
    )
    rectifier: FthetaToPinholeRectifier | None = None
    rectification_disabled: bool = False


class VavamPolicyHandle:
    """Loads the heavyweight policy after gRPC startup and exposes readiness."""

    def __init__(
        self, *, checkpoint_path: str, tokenizer_path: str, device: str
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._tokenizer_path = tokenizer_path
        self._device = device
        self._lock = threading.Lock()
        self._policy: Any | None = None
        self._load_error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._load_policy,
            name="vavam-policy-loader",
            daemon=True,
        )
        self._thread.start()

    def get(self) -> Any | None:
        with self._lock:
            return self._policy

    def load_error(self) -> BaseException | None:
        with self._lock:
            return self._load_error

    def ready(self) -> bool:
        with self._lock:
            return self._policy is not None

    def _load_policy(self) -> None:
        try:
            LOGGER.info(
                "loading VAVAM policy checkpoint=%s tokenizer=%s device=%s",
                self._checkpoint_path,
                self._tokenizer_path,
                self._device,
            )
            from .vavam_policy import VavamPolicy

            policy = VavamPolicy(
                checkpoint_path=self._checkpoint_path,
                tokenizer_path=self._tokenizer_path,
                device=self._device,
            )
        except BaseException as exc:
            with self._lock:
                self._load_error = exc
            LOGGER.exception("VAVAM policy load failed")
            return

        with self._lock:
            self._policy = policy
        LOGGER.info("VAVAM policy load complete")


def default_rectification_config() -> RectificationTargetConfig:
    """NuScenes-style pinhole target used by the existing AlpaSim VAVAM config."""

    return RectificationTargetConfig(
        focal_length=(1545.0, 1545.0),
        principal_point=(960.0, 560.0),
        resolution_hw=(1080, 1920),
        radial=(-0.356123, 0.172545, -0.05231, 0.0, 0.0, 0.0),
        tangential=(-0.00213, 0.000464),
        thin_prism=(0.0, 0.0, 0.0, 0.0),
        max_overscan_scale=2.0,
        safety_margin_px=10,
    )


class VavamChallengeDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    """gRPC service that runs VAVAM inference inline during ``drive``."""

    def __init__(
        self,
        *,
        policy_handle: VavamPolicyHandle,
        camera_id: str | None,
        camera_candidates: tuple[str, ...],
        inference_interval_us: int,
        enable_rectification: bool,
    ) -> None:
        self._policy_handle = policy_handle
        self._camera_id = camera_id
        self._camera_candidates = camera_candidates
        self._inference_interval_us = inference_interval_us
        self._rectification_cfg = (
            default_rectification_config() if enable_rectification else None
        )
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()
        # Serializes GPU inference across concurrent Drive calls (one model, one
        # GPU): a second session waits its turn rather than running in parallel.
        self._inference_lock = threading.Lock()
        self._server: grpc.Server | None = None

    def attach_server(self, server: grpc.Server) -> None:
        self._server = server

    def start_session(
        self,
        request: egodriver_pb2.DriveSessionRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.SessionRequestStatus:
        camera_specs: dict[
            str, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera
        ] = {}
        vehicle = request.rollout_spec.vehicle
        if vehicle is not None:
            for camera in vehicle.available_cameras:
                if camera.logical_id:
                    camera_specs[camera.logical_id] = camera

        camera_id = self._select_camera_id(camera_specs)

        with self._lock:
            self._sessions[request.session_uuid] = SessionState(
                camera_id=camera_id,
                camera_specs=camera_specs,
            )

        LOGGER.info(
            "started session %s, keeping camera %s from %s available cameras",
            request.session_uuid,
            camera_id,
            len(camera_specs),
        )
        return common_pb2.SessionRequestStatus()

    def close_session(
        self,
        request: egodriver_pb2.DriveSessionCloseRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        with self._lock:
            self._sessions.pop(request.session_uuid, None)
        LOGGER.info("closed session %s", request.session_uuid)
        return common_pb2.Empty()

    def submit_image_observation(
        self,
        request: egodriver_pb2.RolloutCameraImage,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        grpc_image = request.camera_image
        session = self._get_session(request.session_uuid, context)
        if grpc_image.logical_id != session.camera_id:
            return common_pb2.Empty()

        image = Image.open(BytesIO(grpc_image.image_bytes)).convert("RGB")

        with self._lock:
            rectifier = self._get_or_build_rectifier_locked(session, image)
        if rectifier is not None:
            image_array = rectifier.rectify(np.array(image))
        else:
            image_array = np.array(image)

        with self._lock:
            session.latest_image = image_array
            session.latest_image_timestamp_us = int(grpc_image.frame_end_us)

        return common_pb2.Empty()

    def submit_egomotion_observation(
        self,
        request: egodriver_pb2.RolloutEgoTrajectory,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        session = self._get_session(request.session_uuid, context)
        with self._lock:
            if request.trajectory.poses:
                session.poses.extend(request.trajectory.poses)
                session.poses.sort(key=lambda pose: pose.timestamp_us)
                session.latest_pose = session.poses[-1]
            for idx, dynamic_state in enumerate(request.dynamic_states):
                if idx < len(request.trajectory.poses):
                    timestamp_us = int(request.trajectory.poses[idx].timestamp_us)
                    session.dynamic_states.append((timestamp_us, dynamic_state))
            session.dynamic_states.sort(key=lambda item: item[0])
            if len(session.poses) > 32:
                session.poses = session.poses[-32:]
            if len(session.dynamic_states) > 32:
                session.dynamic_states = session.dynamic_states[-32:]
        return common_pb2.Empty()

    def submit_route(
        self,
        request: egodriver_pb2.RouteRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        session = self._get_session(request.session_uuid, context)
        command = _command_from_route(request.route)
        with self._lock:
            session.command = command
        return common_pb2.Empty()

    def submit_recording_ground_truth(
        self,
        request: egodriver_pb2.GroundTruthRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        return common_pb2.Empty()

    def drive(
        self,
        request: egodriver_pb2.DriveRequest,
        context: grpc.ServicerContext,
    ) -> egodriver_pb2.DriveResponse:
        session = self._get_session(request.session_uuid, context)
        time_now_us = int(request.time_now_us)
        self._maybe_run_inference(session, time_now_us)

        with self._lock:
            pose = session.latest_pose
            plan = session.cached_plan
            speed = _estimate_speed_mps(session)

        trajectory = build_trajectory_from_plan(
            plan,
            pose,
            time_now_us,
            fallback_speed_mps=max(2.0, speed),
        )
        return egodriver_pb2.DriveResponse(trajectory=trajectory)

    def get_version(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.VersionId:
        if os.environ.get("VAVAM_REQUIRE_POLICY_FOR_VERSION", "1") == "1":
            load_error = self._policy_handle.load_error()
            if load_error is not None:
                context.abort(
                    grpc.StatusCode.UNAVAILABLE,
                    f"VAVAM policy load failed: {load_error}",
                )
            if not self._policy_handle.ready():
                context.abort(
                    grpc.StatusCode.UNAVAILABLE, "VAVAM policy is still loading"
                )
        return common_pb2.VersionId(
            version_id="vavam-e2e-driver",
            git_hash="local",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    def shut_down(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        if self._server is not None:
            threading.Thread(target=self._stop_server, daemon=True).start()
        return common_pb2.Empty()

    def stop(self) -> None:
        return

    def _stop_server(self) -> None:
        time.sleep(0.05)
        if self._server is not None:
            self._server.stop(grace=0.0)

    def _should_run_inference(self, session: SessionState, time_now_us: int) -> bool:
        """Decide, using sim time only, whether a new plan is due this step."""
        if session.latest_image is None or session.latest_pose is None:
            return False
        plan = session.cached_plan
        if (
            plan is not None
            and time_now_us - plan.created_time_us < self._inference_interval_us
        ):
            return False
        return True

    def _maybe_run_inference(self, session: SessionState, time_now_us: int) -> None:
        """Run VAVAM inline and cache the resulting plan, if one is due.

        Inference happens here, inside the Drive RPC, so the trajectory we
        return reflects the latest observations.  The cached plan is anchored at
        the current ego pose, so a fresh plan begins exactly at the ego.
        """
        policy = self._policy_handle.get()
        if policy is None:
            return

        with self._lock:
            if not self._should_run_inference(session, time_now_us):
                return
            image = session.latest_image
            anchor_pose = session.latest_pose
            command = session.command

        with self._inference_lock:
            try:
                prediction = policy.predict(image, command)
            except Exception:
                LOGGER.exception("VAVAM inference failed")
                return

        plan = make_cached_plan(
            created_time_us=time_now_us,
            anchor_pose=anchor_pose,
            trajectory_xy=prediction.trajectory_xy,
            headings=prediction.headings,
            source_frequency_hz=policy.output_frequency_hz,
        )
        if plan is not None:
            with self._lock:
                session.cached_plan = plan

    def _get_or_build_rectifier_locked(
        self,
        session: SessionState,
        image: Image.Image,
    ) -> FthetaToPinholeRectifier | None:
        if self._rectification_cfg is None:
            return None
        if session.rectification_disabled:
            return None
        if session.rectifier is not None:
            return session.rectifier
        camera_id = session.camera_id
        if camera_id is None:
            LOGGER.warning(
                "no compatible camera found for session; skipping rectification"
            )
            session.rectification_disabled = True
            return None
        camera = session.camera_specs.get(camera_id)
        if camera is None:
            LOGGER.warning(
                "camera spec for %s missing; skipping rectification", camera_id
            )
            session.rectification_disabled = True
            return None
        if camera.intrinsics.WhichOneof("camera_param") != "ftheta_param":
            LOGGER.info("camera %s is not f-theta; skipping rectification", camera_id)
            session.rectification_disabled = True
            return None
        session.rectifier = build_ftheta_rectifier_for_resolution(
            camera_proto=camera,
            target_cfg=self._rectification_cfg,
            source_resolution_hw=(image.height, image.width),
        )
        LOGGER.info(
            "enabled rectification for %s at %sx%s",
            camera_id,
            image.width,
            image.height,
        )
        return session.rectifier

    def _select_camera_id(
        self,
        camera_specs: dict[str, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera],
    ) -> str | None:
        if self._camera_id:
            if self._camera_id not in camera_specs:
                LOGGER.warning(
                    "requested camera %s is not available; image observations will be ignored",
                    self._camera_id,
                )
            return self._camera_id

        for candidate in self._camera_candidates:
            if candidate in camera_specs:
                return candidate

        for logical_id in camera_specs:
            lowered = logical_id.lower()
            if "front" in lowered or lowered in {"cam_f0", "cam_front"}:
                return logical_id

        if camera_specs:
            fallback = next(iter(camera_specs))
            LOGGER.warning(
                "no preferred front camera found; falling back to first available camera %s",
                fallback,
            )
            return fallback

        LOGGER.warning("session has no available cameras")
        return None

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


def _command_from_route(route: egodriver_pb2.Route) -> int:
    threshold_m = float(os.environ.get("VAVAM_ROUTE_LATERAL_THRESHOLD_M", "3.0"))
    min_lookahead_m = float(os.environ.get("VAVAM_ROUTE_MIN_LOOKAHEAD_M", "20.0"))
    candidates = [wp for wp in route.waypoints if wp.x >= min_lookahead_m]
    waypoint = (
        candidates[0]
        if candidates
        else (route.waypoints[-1] if route.waypoints else None)
    )
    if waypoint is None:
        return 2
    if waypoint.y > threshold_m:
        return 1
    if waypoint.y < -threshold_m:
        return 0
    return 2


def _estimate_speed_mps(session: SessionState) -> float:
    if session.dynamic_states:
        state = session.dynamic_states[-1][1]
        return float(np.hypot(state.linear_velocity.x, state.linear_velocity.y))
    if len(session.poses) >= 2:
        a, b = session.poses[-2], session.poses[-1]
        dt = (int(b.timestamp_us) - int(a.timestamp_us)) / 1_000_000.0
        if dt > 1e-6:
            return float(
                np.hypot(b.pose.vec.x - a.pose.vec.x, b.pose.vec.y - a.pose.vec.y) / dt
            )
    return 5.0


def _camera_candidates_from_env() -> tuple[str, ...]:
    raw = os.environ.get(
        "VAVAM_CAMERA_CANDIDATES",
        "CAM_F0,camera_front_wide_120fov,camera_front_tele_30fov",
    )
    return tuple(candidate.strip() for candidate in raw.split(",") if candidate.strip())


def _configure_runtime_write_dirs() -> None:
    """Keep framework caches on writable tmpfs paths in the challenge container."""

    for key, value in _RUNTIME_WRITE_DIR_DEFAULTS.items():
        os.environ.setdefault(key, value)
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


def main() -> None:
    _configure_torch_threads()
    _configure_runtime_write_dirs()
    logging.basicConfig(
        level=os.environ.get("ALPASIM_DRIVER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    host = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0")
    port = int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    camera_id = os.environ.get("VAVAM_CAMERA_ID")
    if camera_id is not None:
        camera_id = camera_id.strip() or None
    camera_candidates = _camera_candidates_from_env()
    checkpoint_path = os.environ.get(
        "VAVAM_CHECKPOINT_PATH",
        "/app/assets/vavam/VAM_width_1024_pretrained_139k.pt",
    )
    tokenizer_path = os.environ.get(
        "VAVAM_TOKENIZER_PATH",
        "/app/assets/vavam/VQ_ds16_16384_llamagen_encoder.jit",
    )
    policy_handle = VavamPolicyHandle(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=os.environ.get("VAVAM_DEVICE", "cuda"),
    )
    service = VavamChallengeDriver(
        policy_handle=policy_handle,
        camera_id=camera_id,
        camera_candidates=camera_candidates,
        inference_interval_us=int(
            os.environ.get("VAVAM_INFERENCE_INTERVAL_US", "500000")
        ),
        enable_rectification=os.environ.get("VAVAM_DISABLE_RECTIFICATION", "0") != "1",
    )

    grpc_workers = max(1, int(os.environ.get("ALPASIM_DRIVER_GRPC_WORKERS", "4")))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=grpc_workers))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(service, server)
    service.attach_server(server)

    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")

    def request_stop(signum: int, frame: object) -> None:
        LOGGER.info("received signal %s, stopping", signum)
        server.stop(grace=0.0)
        service.stop()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    server.start()
    LOGGER.info("VAVAM driver listening on %s:%d", host, bound_port)
    policy_handle.start()
    try:
        server.wait_for_termination()
    finally:
        service.stop()


if __name__ == "__main__":
    main()
