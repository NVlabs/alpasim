# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import navsim_gtrs_dense_challenge.driver as driver_module
import numpy as np
import pytest
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, sensorsim_pb2
from navsim_gtrs_dense_challenge.driver import (
    NavsimGTRSDenseDriver,
    PolicyHandle,
    SessionCounters,
    SessionState,
    _context_timeout,
    _derive_dynamics_from_poses,
)
from navsim_gtrs_dense_challenge.navigation import DriveCommand, command_one_hot
from navsim_gtrs_dense_challenge.policy import InferenceInput, Prediction
from navsim_gtrs_dense_challenge.preprocessing import CAMERA_IDS
from PIL import Image

import grpc


def _pose(
    timestamp_us: int,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
) -> common_pb2.PoseAtTime:
    return common_pb2.PoseAtTime(
        timestamp_us=timestamp_us,
        pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=x, y=y),
            quat=common_pb2.Quat(
                w=math.cos(yaw / 2),
                z=math.sin(yaw / 2),
            ),
        ),
    )


def _state(
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    acceleration_x: float = 0.0,
    acceleration_y: float = 0.0,
) -> common_pb2.DynamicState:
    return common_pb2.DynamicState(
        linear_velocity=common_pb2.Vec3(x=velocity_x, y=velocity_y),
        linear_acceleration=common_pb2.Vec3(
            x=acceleration_x,
            y=acceleration_y,
        ),
    )


def _image(marker: int) -> np.ndarray:
    return np.full((2, 2, 3), marker, dtype=np.uint8)


def test_session_defaults_are_independent_and_unknown_command() -> None:
    first = SessionState()
    second = SessionState()

    assert first.counters == SessionCounters()
    np.testing.assert_array_equal(
        first.command_one_hot,
        command_one_hot(DriveCommand.UNKNOWN),
    )
    assert first.cached_plan is None
    assert first.last_inference_timestamp_us is None
    assert first.images is not second.images
    assert first.counters is not second.counters


def test_session_selects_latest_complete_camera_timestamp_before_now() -> None:
    state = SessionState()
    for camera_id in CAMERA_IDS:
        state.add_image(camera_id, 1_000_000, _image(1))
        state.add_image(camera_id, 2_000_000, _image(2))
    state.add_image("CAM_F0", 1_500_000, _image(15))

    timestamp_us, images = state.synchronized_images(1_750_000)

    assert timestamp_us == 1_000_000
    assert set(images) == set(CAMERA_IDS)
    assert {int(image[0, 0, 0]) for image in images.values()} == {1}


def test_session_accepts_camera_timestamps_within_one_millisecond() -> None:
    state = SessionState()
    timestamps = dict(zip(CAMERA_IDS, (1_000_000, 1_000_500, 1_001_000)))
    for marker, camera_id in enumerate(CAMERA_IDS):
        state.add_image(camera_id, timestamps[camera_id], _image(marker))

    timestamp_us, images = state.synchronized_images(1_001_000)

    assert timestamp_us == 1_001_000
    assert [int(images[camera_id][0, 0, 0]) for camera_id in CAMERA_IDS] == [
        0,
        1,
        2,
    ]


def test_session_rejects_camera_timestamps_over_one_millisecond_apart() -> None:
    state = SessionState()
    timestamps = dict(zip(CAMERA_IDS, (1_000_000, 1_000_500, 1_001_001)))
    for camera_id in CAMERA_IDS:
        state.add_image(camera_id, timestamps[camera_id], _image(1))

    with pytest.raises(
        LookupError,
        match="^camera timestamps differ by more than 1 ms$",
    ):
        state.synchronized_images(1_001_001)


def test_session_rejects_missing_or_future_only_camera_sets() -> None:
    missing = SessionState()
    missing.add_image("CAM_L0", 1_000_000, _image(1))
    missing.add_image("CAM_F0", 1_000_000, _image(1))

    with pytest.raises(
        LookupError,
        match="^no complete synchronized camera set$",
    ):
        missing.synchronized_images(1_000_000)

    future = SessionState()
    for camera_id in CAMERA_IDS:
        future.add_image(camera_id, 2_000_000, _image(2))

    with pytest.raises(
        LookupError,
        match="^no complete synchronized camera set$",
    ):
        future.synchronized_images(1_000_000)


def test_session_rejects_unknown_camera() -> None:
    state = SessionState()

    with pytest.raises(ValueError, match="^unknown camera_id: CAM_UNKNOWN$"):
        state.add_image("CAM_UNKNOWN", 1_000_000, _image(1))


def test_image_cache_keeps_latest_four_sorted_and_replaces_duplicates() -> None:
    state = SessionState()
    for timestamp_us in (5, 1, 4, 2, 3):
        state.add_image("CAM_F0", timestamp_us, _image(timestamp_us))

    cache = state.images["CAM_F0"]
    assert list(cache) == [2, 3, 4, 5]

    replacement = _image(40)
    state.add_image("CAM_F0", 4, replacement)

    assert list(cache) == [2, 3, 4, 5]
    assert cache[4] is replacement


def test_session_uses_latest_pose_and_exact_dynamic_state_before_now() -> None:
    state = SessionState()
    state.add_egomotion(
        [
            _pose(1_000_000, 1.0),
            _pose(1_500_000, 2.0),
            _pose(2_000_000, 3.0),
        ],
        [
            _state(velocity_x=3.0, acceleration_x=0.5),
            _state(velocity_x=4.0, acceleration_x=1.0),
            _state(velocity_x=5.0, acceleration_x=1.5),
        ],
    )

    pose, velocity, acceleration = state.ego_snapshot(1_250_000)

    assert pose.timestamp_us == 1_000_000
    np.testing.assert_array_equal(velocity, [3.0, 0.0])
    np.testing.assert_array_equal(acceleration, [0.5, 0.0])
    assert velocity.dtype == np.float32
    assert acceleration.dtype == np.float32
    assert state.counters.dynamic_state_fallback == 0


def test_session_defaults_to_exact_rpc_dynamics() -> None:
    state = SessionState()
    state.add_egomotion(
        [
            _pose(1_000_000, 0.0),
            _pose(1_500_000, 1.0),
            _pose(2_000_000, 3.0),
        ],
        [
            _state(velocity_x=97.0, acceleration_x=7.0),
            _state(velocity_x=98.0, acceleration_x=8.0),
            _state(velocity_x=99.0, acceleration_x=9.0),
        ],
    )

    _, velocity, acceleration = state.ego_snapshot(2_000_000)

    np.testing.assert_array_equal(velocity, [99.0, 0.0])
    np.testing.assert_array_equal(acceleration, [9.0, 0.0])
    assert state.counters.dynamic_state_fallback == 0


def test_session_derives_rig_velocity_from_pose_history() -> None:
    state = SessionState()
    state.add_egomotion(
        [_pose(1_000_000, 0.0), _pose(1_500_000, 1.0)],
        [],
    )

    _, velocity, acceleration = state.ego_snapshot(1_500_000)

    np.testing.assert_allclose(velocity, [2.0, 0.0], atol=1e-6)
    np.testing.assert_array_equal(acceleration, [0.0, 0.0])
    assert velocity.dtype == np.float32
    assert state.counters.dynamic_state_fallback == 1


def test_session_derives_rig_acceleration_from_three_poses() -> None:
    state = SessionState()
    state.add_egomotion(
        [
            _pose(1_000_000, 0.0),
            _pose(1_500_000, 1.0),
            _pose(2_000_000, 3.0),
        ],
        [],
    )

    _, velocity, acceleration = state.ego_snapshot(2_000_000)

    np.testing.assert_allclose(velocity, [4.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(acceleration, [4.0, 0.0], atol=1e-6)


def test_session_rotates_local_pose_delta_into_current_rig_frame() -> None:
    state = SessionState()
    state.add_egomotion(
        [
            _pose(1_000_000, y=0.0, yaw=math.pi / 2),
            _pose(1_500_000, y=1.0, yaw=math.pi / 2),
        ],
        [],
    )

    _, velocity, _ = state.ego_snapshot(1_500_000)

    np.testing.assert_allclose(velocity, [2.0, 0.0], atol=1e-6)


def test_session_does_not_reuse_state_from_an_older_pose() -> None:
    state = SessionState()
    state.add_egomotion(
        [_pose(1_000_000, 0.0)],
        [_state(velocity_x=99.0)],
    )
    state.add_egomotion([_pose(1_500_000, 1.0)], [])

    pose, velocity, _ = state.ego_snapshot(1_500_000)

    assert pose.timestamp_us == 1_500_000
    np.testing.assert_allclose(velocity, [2.0, 0.0], atol=1e-6)
    assert state.counters.dynamic_state_fallback == 1


def test_replacing_pose_without_state_clears_same_timestamp_state() -> None:
    state = SessionState()
    state.add_egomotion(
        [_pose(1_000_000, 0.0)],
        [_state(velocity_x=99.0)],
    )
    replacement = _pose(1_000_000, 10.0)
    state.add_egomotion([replacement], [])

    pose, velocity, acceleration = state.ego_snapshot(1_000_000)

    assert pose is replacement
    np.testing.assert_array_equal(velocity, [0.0, 0.0])
    np.testing.assert_array_equal(acceleration, [0.0, 0.0])
    assert state.counters.dynamic_state_fallback == 1


def test_session_rejects_partial_dynamic_state_lists() -> None:
    state = SessionState()

    with pytest.raises(
        ValueError,
        match="^dynamic_states must be empty or correspond 1:1 with poses$",
    ):
        state.add_egomotion(
            [_pose(1_000_000), _pose(1_500_000)],
            [_state()],
        )


def test_egomotion_caches_keep_latest_32_sorted_and_replace_duplicates() -> None:
    state = SessionState()
    poses = [_pose(index * 1_000, x=float(index)) for index in range(1, 36)]
    dynamics = [_state(velocity_x=float(index)) for index in range(1, 36)]
    state.add_egomotion(list(reversed(poses)), list(reversed(dynamics)))

    expected_timestamps = [index * 1_000 for index in range(4, 36)]
    assert [pose.timestamp_us for pose in state.poses] == expected_timestamps
    assert [timestamp_us for timestamp_us, _ in state.dynamics] == expected_timestamps

    replacement_pose = _pose(20_000, x=999.0)
    replacement_state = _state(velocity_x=777.0)
    state.add_egomotion([replacement_pose], [replacement_state])

    assert (
        next(pose for pose in state.poses if pose.timestamp_us == 20_000)
        is replacement_pose
    )
    assert dict(state.dynamics)[20_000] is replacement_state


@pytest.mark.parametrize(
    "poses",
    [
        [_pose(1_000_000, 0.0), _pose(1_000_000, 1.0)],
        [_pose(1_000_000, 0.0), _pose(1_000_001, 1.0)],
        [_pose(2_000_000, 0.0), _pose(1_000_000, 1.0)],
    ],
)
def test_pose_dynamics_returns_zeros_for_non_increasing_or_tiny_current_dt(
    poses: list[common_pb2.PoseAtTime],
) -> None:
    velocity, acceleration = _derive_dynamics_from_poses(poses)

    np.testing.assert_array_equal(velocity, [0.0, 0.0])
    np.testing.assert_array_equal(acceleration, [0.0, 0.0])
    assert velocity.dtype == np.float32
    assert acceleration.dtype == np.float32


def test_pose_dynamics_keeps_velocity_but_zeroes_invalid_previous_dt() -> None:
    velocity, acceleration = _derive_dynamics_from_poses(
        [
            _pose(1_000_000, 0.0),
            _pose(999_999, 0.0),
            _pose(1_499_999, 1.0),
        ]
    )

    np.testing.assert_allclose(velocity, [2.0, 0.0], atol=1e-6)
    np.testing.assert_array_equal(acceleration, [0.0, 0.0])


def test_session_rejects_snapshot_without_pose_at_or_before_now() -> None:
    state = SessionState()
    state.add_egomotion([_pose(2_000_000)], [])

    with pytest.raises(
        LookupError,
        match="^no ego pose at or before Drive time$",
    ):
        state.ego_snapshot(1_000_000)


class AbortError(Exception):
    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__(details)
        self.code = code
        self.details = details


class FakeContext:
    def __init__(self, time_remaining: float | None = 1.0):
        self._time_remaining = time_remaining

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise AbortError(code, details)

    def time_remaining(self) -> float | None:
        return self._time_remaining


class ContextWithoutDeadline:
    def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise AbortError(code, details)


class OverflowingTimeout:
    def __float__(self) -> float:
        raise OverflowError("synthetic timeout overflow")


class FakeWorker:
    def __init__(self, trajectory: np.ndarray):
        self.trajectory = np.asarray(trajectory, dtype=np.float32)
        self.calls: list[InferenceInput] = []
        self.timeouts: list[float | None] = []
        self.error: Exception | None = None
        self.lock = threading.Lock()

    def predict(
        self,
        request: InferenceInput,
        timeout: float | None = None,
    ) -> Prediction:
        with self.lock:
            self.calls.append(request)
            self.timeouts.append(timeout)
            error = self.error
        if error is not None:
            raise error
        return Prediction(trajectory=self.trajectory.copy())


class BlockingWorker(FakeWorker):
    def __init__(self, trajectory: np.ndarray):
        super().__init__(trajectory)
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict(
        self,
        request: InferenceInput,
        timeout: float | None = None,
    ) -> Prediction:
        with self.lock:
            self.calls.append(request)
            self.timeouts.append(timeout)
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test worker was not released")
        return Prediction(trajectory=self.trajectory.copy())


class FakePolicyHandle:
    def __init__(
        self,
        worker: FakeWorker | None = None,
        *,
        ready: bool = True,
        load_error: BaseException | None = None,
    ) -> None:
        self.current_worker = worker
        self.ready = ready
        self.current_load_error = load_error
        self.stopped = False
        self.start_count = 0
        self.stop_count = 0
        self.wait_timeouts: list[float | None] = []

    def start(self) -> None:
        self.start_count += 1

    def wait_ready(self, timeout: float | None) -> bool:
        self.wait_timeouts.append(timeout)
        return self.ready

    def load_error(self) -> BaseException | None:
        return self.current_load_error

    def worker(self) -> FakeWorker | None:
        return self.current_worker

    def stop(self) -> None:
        self.stopped = True
        self.stop_count += 1


def _prediction() -> np.ndarray:
    return np.column_stack([np.arange(1, 41) * 0.1, np.zeros(40), np.zeros(40)]).astype(
        np.float32
    )


def _service(worker: FakeWorker | None = None) -> NavsimGTRSDenseDriver:
    if worker is None:
        worker = FakeWorker(_prediction())
    return NavsimGTRSDenseDriver(FakePolicyHandle(worker))


def _session_request(
    session_uuid: str = "session",
    camera_ids: tuple[str, ...] = CAMERA_IDS,
) -> egodriver_pb2.DriveSessionRequest:
    cameras = [
        sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(logical_id=camera_id)
        for camera_id in camera_ids
    ]
    return egodriver_pb2.DriveSessionRequest(
        session_uuid=session_uuid,
        rollout_spec=egodriver_pb2.DriveSessionRequest.RolloutSpec(
            vehicle=egodriver_pb2.DriveSessionRequest.RolloutSpec.VehicleDefinition(
                available_cameras=cameras
            )
        ),
    )


@lru_cache(maxsize=None)
def _jpeg(value: int = 0, *, width: int = 1920, height: int = 1080) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(
        buffer, format="JPEG"
    )
    return buffer.getvalue()


def _submit_images(
    service: NavsimGTRSDenseDriver,
    session_uuid: str = "session",
    timestamp_us: int = 1_000_000,
) -> None:
    payload = _jpeg()
    for camera_id in ("CAM_R0", "CAM_B0", "CAM_L0", "CAM_F0"):
        service.submit_image_observation(
            egodriver_pb2.RolloutCameraImage(
                session_uuid=session_uuid,
                camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
                    frame_start_us=timestamp_us - 30_000,
                    frame_end_us=timestamp_us,
                    logical_id=camera_id,
                    image_bytes=payload,
                ),
            ),
            FakeContext(),
        )


def _submit_egomotion(
    service: NavsimGTRSDenseDriver,
    session_uuid: str = "session",
    timestamp_us: int = 1_000_000,
    *,
    x: float = 0.0,
    velocity: tuple[float, float] = (4.0, -0.5),
    acceleration: tuple[float, float] = (0.2, 0.1),
) -> None:
    service.submit_egomotion_observation(
        egodriver_pb2.RolloutEgoTrajectory(
            session_uuid=session_uuid,
            trajectory=common_pb2.Trajectory(poses=[_pose(timestamp_us, x=x)]),
            dynamic_states=[
                _state(
                    velocity_x=velocity[0],
                    velocity_y=velocity[1],
                    acceleration_x=acceleration[0],
                    acceleration_y=acceleration[1],
                )
            ],
        ),
        FakeContext(),
    )


def _submit_route(
    service: NavsimGTRSDenseDriver,
    session_uuid: str = "session",
    lateral_y: float = 3.0,
) -> None:
    service.submit_route(
        egodriver_pb2.RouteRequest(
            session_uuid=session_uuid,
            route=egodriver_pb2.Route(
                timestamp_us=1_000_000,
                waypoints=[common_pb2.Vec3(x=10.0, y=lateral_y)],
            ),
        ),
        FakeContext(),
    )


def _drive(
    service: NavsimGTRSDenseDriver,
    session_uuid: str = "session",
    time_now_us: int = 1_000_000,
    context: FakeContext | ContextWithoutDeadline | None = None,
) -> egodriver_pb2.DriveResponse:
    if context is None:
        context = FakeContext()
    return service.drive(
        egodriver_pb2.DriveRequest(
            session_uuid=session_uuid,
            time_now_us=time_now_us,
            time_query_us=time_now_us + 500_000,
        ),
        context,
    )


def test_start_session_rejects_missing_camera() -> None:
    service = _service()

    with pytest.raises(AbortError) as exc_info:
        service.start_session(
            _session_request(camera_ids=("CAM_L0", "CAM_F0")),
            FakeContext(),
        )

    assert exc_info.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "CAM_R0" in exc_info.value.details


def test_observation_rpcs_build_one_complete_inference_input() -> None:
    worker = FakeWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_route(service)
    _submit_images(service)
    _submit_egomotion(service)

    response = _drive(service)

    assert len(worker.calls) == 1
    submitted = worker.calls[0]
    assert set(submitted.images) == set(CAMERA_IDS)
    assert all(not image.flags.writeable for image in submitted.images.values())
    np.testing.assert_array_equal(submitted.command_one_hot, [1, 0, 0, 0])
    np.testing.assert_allclose(submitted.velocity_xy, [4.0, -0.5])
    np.testing.assert_allclose(submitted.acceleration_xy, [0.2, 0.1])
    timestamps = [pose.timestamp_us for pose in response.trajectory.poses]
    assert timestamps[0] == 1_000_000
    assert timestamps[-1] >= 1_500_000


@pytest.mark.parametrize(
    "image_bytes",
    [
        b"not a jpeg",
        _jpeg(width=1280, height=720),
    ],
)
def test_image_rpc_rejects_bad_data_or_wrong_resolution(
    image_bytes: bytes,
) -> None:
    service = _service()
    service.start_session(_session_request(), FakeContext())
    request = egodriver_pb2.RolloutCameraImage(
        session_uuid="session",
        camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
            frame_end_us=1_000_000,
            logical_id="CAM_L0",
            image_bytes=image_bytes,
        ),
    )

    with pytest.raises(AbortError) as exc_info:
        service.submit_image_observation(request, FakeContext())

    assert exc_info.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_wrong_size_image_is_rejected_before_convert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service.start_session(_session_request(), FakeContext())

    def unexpected_convert(*args: object, **kwargs: object) -> Image.Image:
        raise AssertionError("convert must not run for a wrong-size image")

    monkeypatch.setattr(driver_module.Image.Image, "convert", unexpected_convert)
    request = egodriver_pb2.RolloutCameraImage(
        session_uuid="session",
        camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
            frame_end_us=1_000_000,
            logical_id="CAM_L0",
            image_bytes=_jpeg(width=1280, height=720),
        ),
    )

    with pytest.raises(AbortError) as exc_info:
        service.submit_image_observation(request, FakeContext())

    assert exc_info.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_image_rpc_maps_decompression_bomb_to_invalid_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service.start_session(_session_request(), FakeContext())

    def bomb(*args: object, **kwargs: object) -> Image.Image:
        raise Image.DecompressionBombError("synthetic bomb")

    monkeypatch.setattr(driver_module.Image, "open", bomb)
    request = egodriver_pb2.RolloutCameraImage(
        session_uuid="session",
        camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
            frame_end_us=1_000_000,
            logical_id="CAM_L0",
            image_bytes=b"header only",
        ),
    )

    with pytest.raises(AbortError) as exc_info:
        service.submit_image_observation(request, FakeContext())

    assert exc_info.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_egomotion_rpc_rejects_partial_dynamic_states() -> None:
    service = _service()
    service.start_session(_session_request(), FakeContext())
    request = egodriver_pb2.RolloutEgoTrajectory(
        session_uuid="session",
        trajectory=common_pb2.Trajectory(poses=[_pose(1_000_000), _pose(1_500_000)]),
        dynamic_states=[common_pb2.DynamicState()],
    )

    with pytest.raises(AbortError) as exc_info:
        service.submit_egomotion_observation(request, FakeContext())

    assert exc_info.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_repeated_frame_uses_cache_and_new_frame_runs_inference() -> None:
    worker = FakeWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)

    _drive(service)
    _drive(service, time_now_us=1_100_000)

    assert len(worker.calls) == 1
    assert service._sessions["session"].counters.cached_plan == 1

    _submit_egomotion(service, timestamp_us=1_200_000, x=0.5)
    _submit_images(service, timestamp_us=1_200_000)
    _drive(service, time_now_us=1_200_000)

    assert len(worker.calls) == 2
    assert service._sessions["session"].counters.gtrs_inference == 2


def test_expired_cached_plan_counts_straight_fallback() -> None:
    worker = FakeWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)
    _drive(service)

    response = _drive(service, time_now_us=5_000_000)

    counters = service._sessions["session"].counters
    assert len(worker.calls) == 1
    assert response.trajectory.poses[-1].timestamp_us >= 5_500_000
    assert counters.cached_plan == 0
    assert counters.straight_fallback == 1


def test_first_drive_without_images_returns_straight_fallback() -> None:
    worker = FakeWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)

    response = _drive(service)

    assert worker.calls == []
    assert response.trajectory.poses[-1].timestamp_us >= 1_500_000
    assert service._sessions["session"].counters.straight_fallback == 1


def test_inference_error_reuses_existing_plan() -> None:
    worker = FakeWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)
    _drive(service)

    worker.error = RuntimeError("synthetic inference failure")
    _submit_egomotion(service, timestamp_us=1_200_000, x=0.5)
    _submit_images(service, timestamp_us=1_200_000)
    response = _drive(service, time_now_us=1_200_000)

    counters = service._sessions["session"].counters
    assert response.trajectory.poses[-1].timestamp_us >= 1_700_000
    assert counters.inference_error == 1
    assert counters.cached_plan == 1


def test_inference_error_without_plan_returns_straight_fallback() -> None:
    worker = FakeWorker(_prediction())
    worker.error = RuntimeError("first inference failed")
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)

    response = _drive(service)

    counters = service._sessions["session"].counters
    assert response.trajectory.poses[-1].timestamp_us >= 1_500_000
    assert counters.inference_error == 1
    assert counters.straight_fallback == 1


def test_two_concurrent_sessions_remain_isolated() -> None:
    worker = FakeWorker(_prediction())
    service = _service(worker)
    for session_uuid, x in (("first", 0.0), ("second", 100.0)):
        service.start_session(_session_request(session_uuid), FakeContext())
        _submit_egomotion(service, session_uuid, x=x)
        _submit_images(service, session_uuid)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_drive, service, "first")
        second = executor.submit(_drive, service, "second")
        responses = [first.result(timeout=2.0), second.result(timeout=2.0)]

    assert len(worker.calls) == 2
    assert responses[0].trajectory.poses[0].pose.vec.x == pytest.approx(0.0)
    assert responses[1].trajectory.poses[0].pose.vec.x == pytest.approx(100.0)


def test_close_session_does_not_restore_inflight_state() -> None:
    worker = BlockingWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request("closing"), FakeContext())
    _submit_egomotion(service, "closing")
    _submit_images(service, "closing")
    old_session = service._sessions["closing"]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_drive, service, "closing")
        assert worker.entered.wait(timeout=1.0)
        try:
            service.close_session(
                egodriver_pb2.DriveSessionCloseRequest(session_uuid="closing"),
                FakeContext(),
            )
        finally:
            worker.release.set()
        response = future.result(timeout=2.0)

    assert "closing" not in service._sessions
    assert old_session.inference_inflight_timestamp_us is None
    assert old_session.cached_plan is None
    assert response.trajectory.poses[-1].timestamp_us >= 1_500_000
    pose_at_half_second = next(
        pose for pose in response.trajectory.poses if pose.timestamp_us == 1_500_000
    )
    assert pose_at_half_second.pose.vec.x == pytest.approx(0.5)


def test_drive_without_ego_pose_is_failed_precondition() -> None:
    service = _service()
    service.start_session(_session_request(), FakeContext())
    _submit_images(service)

    with pytest.raises(AbortError) as exc_info:
        _drive(service)

    assert exc_info.value.code == grpc.StatusCode.FAILED_PRECONDITION


def test_unknown_session_is_not_found() -> None:
    with pytest.raises(AbortError) as exc_info:
        _drive(_service(), "missing")

    assert exc_info.value.code == grpc.StatusCode.NOT_FOUND


def test_non_target_camera_is_ignored_without_decoding() -> None:
    service = _service()
    service.start_session(_session_request(), FakeContext())

    service.submit_image_observation(
        egodriver_pb2.RolloutCameraImage(
            session_uuid="session",
            camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
                frame_end_us=1_000_000,
                logical_id="CAM_B0",
                image_bytes=b"not an image",
            ),
        ),
        FakeContext(),
    )

    assert all(not cache for cache in service._sessions["session"].images.values())


def test_route_updates_session_command() -> None:
    service = _service()
    service.start_session(_session_request(), FakeContext())

    _submit_route(service, lateral_y=-3.0)

    np.testing.assert_array_equal(
        service._sessions["session"].command_one_hot,
        command_one_hot(DriveCommand.RIGHT),
    )


def test_same_session_concurrent_drive_reserves_frame_once() -> None:
    worker = BlockingWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(_drive, service)
        assert worker.entered.wait(timeout=1.0)

        second_response = _drive(service)
        assert len(worker.calls) == 1
        assert second_response.trajectory.poses[-1].timestamp_us >= 1_500_000
        assert service._sessions["session"].counters.straight_fallback == 1

        worker.release.set()
        first_response = first_future.result(timeout=2.0)

    assert first_response.trajectory.poses[-1].timestamp_us >= 1_500_000
    assert service._sessions["session"].counters.gtrs_inference == 1

    _drive(service)
    assert len(worker.calls) == 1

    _submit_egomotion(service, timestamp_us=1_200_000, x=0.5)
    _submit_images(service, timestamp_us=1_200_000)
    _drive(service, time_now_us=1_200_000)

    assert len(worker.calls) == 2
    assert service._sessions["session"].counters.gtrs_inference == 2


def test_inference_error_consumes_frame_but_new_frame_retries() -> None:
    worker = FakeWorker(_prediction())
    worker.error = RuntimeError("synthetic failure")
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)

    _drive(service)
    _drive(service)

    assert len(worker.calls) == 1
    assert service._sessions["session"].counters.inference_error == 1

    _submit_egomotion(service, timestamp_us=1_200_000, x=0.5)
    _submit_images(service, timestamp_us=1_200_000)
    _drive(service, time_now_us=1_200_000)

    assert len(worker.calls) == 2
    assert service._sessions["session"].counters.inference_error == 2


def test_base_exception_clears_reservation_and_propagates() -> None:
    class SystemExitThenSuccessWorker(FakeWorker):
        def predict(
            self,
            request: InferenceInput,
            timeout: float | None = None,
        ) -> Prediction:
            with self.lock:
                self.calls.append(request)
                self.timeouts.append(timeout)
                call_count = len(self.calls)
            if call_count == 1:
                raise SystemExit("synthetic shutdown")
            return Prediction(trajectory=self.trajectory.copy())

    worker = SystemExitThenSuccessWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)

    with pytest.raises(SystemExit, match="^synthetic shutdown$"):
        _drive(service)

    session = service._sessions["session"]
    assert session.inference_inflight_timestamp_us is None
    assert session.last_inference_timestamp_us == 1_000_000
    assert session.counters.inference_error == 0

    _drive(service)
    assert len(worker.calls) == 1

    _submit_egomotion(service, timestamp_us=1_200_000, x=0.5)
    _submit_images(service, timestamp_us=1_200_000)
    _drive(service, time_now_us=1_200_000)

    assert len(worker.calls) == 2
    assert session.counters.gtrs_inference == 1
    assert session.counters.inference_error == 0


@pytest.mark.parametrize(
    ("context", "expected_timeout"),
    [
        (FakeContext(None), None),
        (FakeContext(-2.0), 0.0),
        (FakeContext(1.25), 1.25),
        (ContextWithoutDeadline(), None),
    ],
)
def test_context_timeout_is_passed_to_worker(
    context: FakeContext | ContextWithoutDeadline,
    expected_timeout: float | None,
) -> None:
    worker = FakeWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)

    _drive(service, context=context)

    assert worker.timeouts == [expected_timeout]


def test_context_timeout_handles_non_callable_attribute() -> None:
    context = FakeContext()
    context.time_remaining = None  # type: ignore[method-assign]

    assert _context_timeout(context) is None


@pytest.mark.parametrize(
    ("remaining", "expected_timeout"),
    [
        (None, None),
        (float("inf"), None),
        (float("-inf"), 0.0),
        (float("nan"), None),
        (threading.TIMEOUT_MAX, None),
        (threading.TIMEOUT_MAX * 2, None),
        (
            math.nextafter(threading.TIMEOUT_MAX, 0.0),
            math.nextafter(threading.TIMEOUT_MAX, 0.0),
        ),
        (-2.0, 0.0),
        (1.25, 1.25),
    ],
)
def test_context_timeout_normalizes_platform_unsafe_values(
    remaining: float | None,
    expected_timeout: float | None,
) -> None:
    assert _context_timeout(FakeContext(remaining)) == expected_timeout


@pytest.mark.parametrize("remaining", ["invalid", object(), OverflowingTimeout()])
def test_context_timeout_handles_float_conversion_errors(remaining: object) -> None:
    assert _context_timeout(FakeContext(remaining)) is None  # type: ignore[arg-type]


def test_drive_runs_inference_when_context_timeout_conversion_fails() -> None:
    worker = FakeWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)

    _drive(service, context=FakeContext(OverflowingTimeout()))  # type: ignore[arg-type]

    assert worker.timeouts == [None]
    assert service._sessions["session"].counters.gtrs_inference == 1


def test_drive_passes_no_timeout_for_platform_unsafe_context_deadline() -> None:
    worker = FakeWorker(_prediction())
    service = _service(worker)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)

    _drive(service, context=FakeContext(threading.TIMEOUT_MAX * 2))

    assert worker.timeouts == [None]


def test_worker_unavailable_does_not_consume_frame() -> None:
    handle = FakePolicyHandle(None)
    service = NavsimGTRSDenseDriver(handle)
    service.start_session(_session_request(), FakeContext())
    _submit_egomotion(service)
    _submit_images(service)

    _drive(service)
    assert service._sessions["session"].last_inference_timestamp_us is None

    worker = FakeWorker(_prediction())
    handle.current_worker = worker
    _drive(service)

    assert len(worker.calls) == 1
    assert service._sessions["session"].counters.gtrs_inference == 1


def test_image_handler_does_not_write_into_replacement_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service.start_session(_session_request(), FakeContext())
    entered = threading.Event()
    release = threading.Event()
    original_open = driver_module.Image.open
    payload = _jpeg()

    def blocking_open(*args: object, **kwargs: object) -> Image.Image:
        entered.set()
        if not release.wait(timeout=1.0):
            raise TimeoutError("image handler was not released")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(driver_module.Image, "open", blocking_open)
    request = egodriver_pb2.RolloutCameraImage(
        session_uuid="session",
        camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
            frame_end_us=1_000_000,
            logical_id="CAM_L0",
            image_bytes=payload,
        ),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            service.submit_image_observation,
            request,
            FakeContext(),
        )
        assert entered.wait(timeout=1.0)
        service.start_session(_session_request(), FakeContext())
        release.set()
        future.result(timeout=2.0)

    assert not service._sessions["session"].images["CAM_L0"]


def test_egomotion_handler_does_not_restore_closed_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service.start_session(_session_request(), FakeContext())
    entered = threading.Event()
    release = threading.Event()
    original_get_session = service._get_session

    def blocking_get_session(
        session_uuid: str,
        context: FakeContext,
    ) -> SessionState:
        session = original_get_session(session_uuid, context)
        entered.set()
        if not release.wait(timeout=1.0):
            raise TimeoutError("egomotion handler was not released")
        return session

    monkeypatch.setattr(service, "_get_session", blocking_get_session)
    request = egodriver_pb2.RolloutEgoTrajectory(
        session_uuid="session",
        trajectory=common_pb2.Trajectory(poses=[_pose(1_000_000)]),
        dynamic_states=[_state()],
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            service.submit_egomotion_observation,
            request,
            FakeContext(),
        )
        assert entered.wait(timeout=1.0)
        service.close_session(
            egodriver_pb2.DriveSessionCloseRequest(session_uuid="session"),
            FakeContext(),
        )
        release.set()
        future.result(timeout=2.0)

    assert "session" not in service._sessions


class FakeBatchPolicy:
    def predict_batch(
        self,
        requests: list[InferenceInput],
    ) -> list[Prediction]:
        return [Prediction(trajectory=_prediction()) for _request_item in requests]


def test_policy_handle_becomes_ready_and_starts_worker() -> None:
    handle = PolicyHandle(
        loader=FakeBatchPolicy,
        max_batch_size=2,
        batch_window_s=0.002,
    )

    handle.start()

    assert handle.wait_ready(timeout=1.0)
    assert handle.load_error() is None
    worker = handle.worker()
    assert worker is not None
    worker_thread = worker._thread
    assert worker_thread is not None
    assert worker_thread.is_alive()

    handle.stop()

    assert handle.worker() is None
    assert not worker_thread.is_alive()


def test_policy_handle_exposes_loader_failure() -> None:
    def fail_loader() -> FakeBatchPolicy:
        raise RuntimeError("bad ckpt")

    handle = PolicyHandle(loader=fail_loader)
    handle.start()

    assert handle.wait_ready(timeout=1.0)
    assert isinstance(handle.load_error(), RuntimeError)
    assert handle.worker() is None
    assert handle._thread is not None
    handle._thread.join(timeout=1.0)
    assert not handle._thread.is_alive()


def test_policy_handle_stop_wins_slow_loader_race() -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow_loader() -> FakeBatchPolicy:
        entered.set()
        if not release.wait(timeout=2.0):
            raise TimeoutError("test loader was not released")
        return FakeBatchPolicy()

    handle = PolicyHandle(loader=slow_loader)
    handle.start()
    assert entered.wait(timeout=1.0)

    handle.stop()
    release.set()

    assert handle.wait_ready(timeout=1.0)
    assert handle.worker() is None
    assert handle.load_error() is None
    assert handle._thread is not None
    handle._thread.join(timeout=1.0)
    assert not handle._thread.is_alive()


def test_policy_handle_start_is_idempotent_and_stop_prevents_restart() -> None:
    entered = threading.Event()
    release = threading.Event()
    loader_calls = 0
    loader_lock = threading.Lock()

    def loader() -> FakeBatchPolicy:
        nonlocal loader_calls
        with loader_lock:
            loader_calls += 1
        entered.set()
        if not release.wait(timeout=2.0):
            raise TimeoutError("test loader was not released")
        return FakeBatchPolicy()

    handle = PolicyHandle(loader=loader)
    handle.start()
    handle.start()
    assert entered.wait(timeout=1.0)
    assert handle._thread is not None
    assert handle._thread.name == "gtrs-policy-loader"
    assert handle._thread.daemon

    release.set()
    assert handle.wait_ready(timeout=1.0)
    handle.start()

    assert loader_calls == 1
    handle.stop()
    handle.start()
    assert loader_calls == 1
    assert handle.worker() is None

    stopped_before_start_calls = 0

    def never_loader() -> FakeBatchPolicy:
        nonlocal stopped_before_start_calls
        stopped_before_start_calls += 1
        return FakeBatchPolicy()

    stopped_before_start = PolicyHandle(loader=never_loader)
    stopped_before_start.stop()
    stopped_before_start.start()

    assert stopped_before_start_calls == 0
    assert stopped_before_start._thread is None


def test_policy_handle_retries_failed_published_worker_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = PolicyHandle(loader=FakeBatchPolicy)
    handle.start()
    assert handle.wait_ready(timeout=1.0)
    worker = handle.worker()
    assert worker is not None
    worker_thread = worker._thread
    assert worker_thread is not None
    original_stop = worker.stop
    stop_calls = 0

    def flaky_stop() -> None:
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 1:
            raise RuntimeError("synthetic stop timeout")
        original_stop()

    monkeypatch.setattr(worker, "stop", flaky_stop)
    try:
        with pytest.raises(RuntimeError, match="^synthetic stop timeout$"):
            handle.stop()

        assert handle.worker() is None
        assert handle._worker is worker
        assert worker_thread.is_alive()

        handle.stop()

        assert handle._worker is None
        assert stop_calls == 2
        assert not worker_thread.is_alive()
    finally:
        if worker_thread.is_alive():
            original_stop()


def test_policy_handle_retains_temporary_worker_when_stop_retry_is_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_entered = threading.Event()
    release_start = threading.Event()
    created_workers: list[object] = []

    class FlakyTemporaryWorker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.stop_calls = 0
            self.stopped = False
            created_workers.append(self)

        def start(self) -> None:
            start_entered.set()
            if not release_start.wait(timeout=2.0):
                raise TimeoutError("temporary worker start was not released")

        def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("synthetic temporary stop timeout")
            self.stopped = True

    monkeypatch.setattr(driver_module, "BatchWorker", FlakyTemporaryWorker)
    handle = PolicyHandle(loader=FakeBatchPolicy)
    handle.start()
    assert start_entered.wait(timeout=1.0)

    handle.stop()
    release_start.set()

    assert handle.wait_ready(timeout=1.0)
    assert len(created_workers) == 1
    temporary_worker = created_workers[0]
    assert temporary_worker.stop_calls == 1
    assert handle.worker() is None
    assert handle._worker is temporary_worker
    assert isinstance(handle.load_error(), RuntimeError)

    handle.stop()

    assert temporary_worker.stop_calls == 2
    assert temporary_worker.stopped
    assert handle._worker is None
    assert handle._thread is not None
    handle._thread.join(timeout=1.0)
    assert not handle._thread.is_alive()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size must be a positive integer"),
        ({"max_batch_size": 1.5}, "max_batch_size must be a positive integer"),
        ({"max_batch_size": True}, "max_batch_size must be a positive integer"),
        ({"batch_window_s": -0.1}, "batch_window_s must be non-negative"),
        ({"batch_window_s": float("nan")}, "batch_window_s must be non-negative"),
        ({"batch_window_s": float("inf")}, "batch_window_s must be non-negative"),
        ({"batch_window_s": True}, "batch_window_s must be non-negative"),
    ],
)
def test_policy_handle_rejects_invalid_batch_settings_before_loader(
    kwargs: dict[str, object],
    message: str,
) -> None:
    loader_called = False

    def loader() -> FakeBatchPolicy:
        nonlocal loader_called
        loader_called = True
        return FakeBatchPolicy()

    with pytest.raises(ValueError, match=f"^{message}$"):
        PolicyHandle(loader=loader, **kwargs)

    assert not loader_called


def test_get_version_reports_ready_policy_and_git_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = FakePolicyHandle(FakeWorker(_prediction()))
    service = NavsimGTRSDenseDriver(handle)
    monkeypatch.setenv("GTRS_GIT_HASH", "test-hash")

    version = service.get_version(common_pb2.Empty(), FakeContext(None))

    assert version.version_id == "simscale-gtrs-dense-e2e"
    assert version.git_hash == "test-hash"
    assert version.grpc_api_version == API_VERSION_MESSAGE
    assert handle.wait_timeouts == [30.0]


def test_get_version_accepts_fixed_service_version_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = FakePolicyHandle(FakeWorker(_prediction()))
    service = NavsimGTRSDenseDriver(handle)
    expected = "simscale-gtrs-dense-resnet-expert-e2e"
    monkeypatch.setenv("GTRS_SERVICE_VERSION", expected)

    version = service.get_version(common_pb2.Empty(), FakeContext(None))

    assert version.version_id == expected


@pytest.mark.parametrize(
    ("handle", "message"),
    [
        (
            FakePolicyHandle(FakeWorker(_prediction()), ready=False),
            "readiness timed out",
        ),
        (
            FakePolicyHandle(
                FakeWorker(_prediction()),
                load_error=RuntimeError("bad ckpt"),
            ),
            "policy load failed: bad ckpt",
        ),
        (
            FakePolicyHandle(None),
            "ready without a batch worker",
        ),
    ],
)
def test_get_version_rejects_unready_policy(
    handle: FakePolicyHandle,
    message: str,
) -> None:
    service = NavsimGTRSDenseDriver(handle)

    with pytest.raises(AbortError) as exc_info:
        service.get_version(common_pb2.Empty(), FakeContext(0.01))

    assert exc_info.value.code == grpc.StatusCode.UNAVAILABLE
    assert message in str(exc_info.value)
    assert handle.wait_timeouts == [0.01]


@pytest.mark.parametrize(
    ("remaining", "expected_timeout"),
    [(-2.0, 0.0), (120.0, 30.0)],
)
def test_get_version_bounds_context_timeout(
    remaining: float,
    expected_timeout: float,
) -> None:
    handle = FakePolicyHandle(FakeWorker(_prediction()))
    service = NavsimGTRSDenseDriver(handle)

    service.get_version(common_pb2.Empty(), FakeContext(remaining))

    assert handle.wait_timeouts == [expected_timeout]


def test_runtime_write_dir_defaults_are_exact() -> None:
    assert driver_module._RUNTIME_WRITE_DIR_DEFAULTS == {
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


def test_configure_runtime_write_dirs_preserves_env_and_creates_paths(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    default_path = root / "default"
    existing_path = root / "existing"
    monkeypatch.setattr(
        driver_module,
        "_RUNTIME_WRITE_DIR_DEFAULTS",
        {
            "TEST_DEFAULT_DIR": str(default_path),
            "TEST_EXISTING_DIR": str(root / "unused"),
        },
    )
    monkeypatch.delenv("TEST_DEFAULT_DIR", raising=False)
    monkeypatch.setenv("TEST_EXISTING_DIR", str(existing_path))

    driver_module._configure_runtime_write_dirs()

    assert driver_module.os.environ["TEST_DEFAULT_DIR"] == str(default_path)
    assert driver_module.os.environ["TEST_EXISTING_DIR"] == str(existing_path)
    assert default_path.is_dir()
    assert existing_path.is_dir()


def test_configure_runtime_write_dirs_falls_back_for_unwritable_log_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = "/run/alpasim-driver"
    fallback = "/tmp/alpasim-driver"
    created: list[tuple[str, bool]] = []

    def make_dir(path: str, *, exist_ok: bool) -> None:
        if path == configured:
            raise PermissionError(path)
        created.append((path, exist_ok))

    monkeypatch.setattr(
        driver_module,
        "_RUNTIME_WRITE_DIR_DEFAULTS",
        {"ALPASIM_DRIVER_LOG_DIR": configured},
    )
    monkeypatch.delenv("ALPASIM_DRIVER_LOG_DIR", raising=False)
    monkeypatch.setattr(driver_module.os, "makedirs", make_dir)

    driver_module._configure_runtime_write_dirs()

    assert driver_module.os.environ["ALPASIM_DRIVER_LOG_DIR"] == fallback
    assert created == [(fallback, True)]


def test_configure_torch_threads_uses_defaults_and_positive_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_threads: list[int] = []
    interop_threads: list[int] = []
    monkeypatch.setattr(driver_module.torch, "set_num_threads", num_threads.append)
    monkeypatch.setattr(
        driver_module.torch,
        "set_num_interop_threads",
        interop_threads.append,
    )
    monkeypatch.delenv("TORCH_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NUM_INTEROP_THREADS", raising=False)

    driver_module._configure_torch_threads()
    monkeypatch.setenv("TORCH_NUM_THREADS", "0")
    monkeypatch.setenv("TORCH_NUM_INTEROP_THREADS", "3")
    driver_module._configure_torch_threads()

    assert num_threads == [1, 1]
    assert interop_threads == [1, 3]


class FakeMainServer:
    def __init__(self, bound_port: int, events: list[object]):
        self.bound_port = bound_port
        self.events = events

    def add_insecure_port(self, address: str) -> int:
        self.events.append(("bind", address))
        return self.bound_port

    def start(self) -> None:
        self.events.append("server_start")

    def wait_for_termination(self) -> None:
        self.events.append("server_wait")

    def stop(self, grace: float) -> None:
        self.events.append(("server_stop", grace))


class FakeMainPolicyHandle(FakePolicyHandle):
    def __init__(self, events: list[object], **kwargs: object):
        super().__init__(FakeWorker(_prediction()))
        self.events = events
        self.kwargs = kwargs

    def start(self) -> None:
        super().start()
        self.events.append("handle_start")

    def stop(self) -> None:
        super().stop()
        self.events.append("handle_stop")


def _install_main_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bound_port: int,
) -> tuple[FakeMainServer, FakeMainPolicyHandle, list[object], dict[str, object]]:
    events: list[object] = []
    captures: dict[str, object] = {}
    server = FakeMainServer(bound_port, events)

    def make_handle(**kwargs: object) -> FakeMainPolicyHandle:
        handle = FakeMainPolicyHandle(events, **kwargs)
        captures["handle"] = handle
        return handle

    def make_executor(*, max_workers: int) -> object:
        captures["grpc_workers"] = max_workers
        return object()

    def register(service: object, registered_server: object) -> None:
        captures["service"] = service
        captures["registered_server"] = registered_server

    monkeypatch.setattr(driver_module, "PolicyHandle", make_handle)
    monkeypatch.setattr(driver_module, "ThreadPoolExecutor", make_executor)
    monkeypatch.setattr(driver_module.grpc, "server", lambda executor: server)
    monkeypatch.setattr(
        driver_module.egodriver_pb2_grpc,
        "add_EgodriverServiceServicer_to_server",
        register,
    )
    monkeypatch.setattr(
        driver_module,
        "_configure_runtime_write_dirs",
        lambda: events.append("configure_dirs"),
    )
    monkeypatch.setattr(
        driver_module,
        "_configure_torch_threads",
        lambda: events.append("configure_threads"),
    )
    monkeypatch.setattr(
        driver_module.logging,
        "basicConfig",
        lambda **kwargs: captures.setdefault("logging", kwargs),
    )
    monkeypatch.setattr(
        driver_module.signal,
        "signal",
        lambda signum, handler: captures.setdefault(f"signal_{signum}", handler),
    )

    def create_policy(
        checkpoint_path: Path,
        vocabulary_path: Path,
        device: str,
        scorer_mode: str = "release",
        ep_exponent: float = 1.0,
        speed_top_k: int = 0,
        speed_weight: float = 0.0,
        speed_proxy: str = "longitudinal",
        curvature_weight: float = 0.0,
        heading_change_weight: float = 0.0,
        backbone_type: str = "resnet",
    ) -> tuple[object, ...]:
        return (
            checkpoint_path,
            vocabulary_path,
            device,
            scorer_mode,
            ep_exponent,
            speed_top_k,
            speed_weight,
            speed_proxy,
            curvature_weight,
            heading_change_weight,
            backbone_type,
        )

    monkeypatch.setattr(
        driver_module,
        "GTRSDensePolicy",
        create_policy,
    )
    return server, captures.get("handle"), events, captures


def test_speed_enhancement_is_enabled_by_default() -> None:
    settings = driver_module._resolve_scorer_settings({})

    assert settings.speed_enhancement is True
    assert settings.scorer_mode == "nc_dac_ep"
    assert settings.ep_exponent == 3.0
    assert settings.speed_top_k == 64
    assert settings.speed_weight == 3.0


def test_speed_enhancement_can_be_disabled() -> None:
    settings = driver_module._resolve_scorer_settings({"GTRS_SPEED_ENHANCEMENT": "0"})

    assert settings.speed_enhancement is False
    assert settings.scorer_mode == "nc_dac_ep"
    assert settings.ep_exponent == 1.0
    assert settings.speed_top_k == 0
    assert settings.speed_weight == 0.0


def test_speed_enhancement_profile_allows_advanced_overrides() -> None:
    settings = driver_module._resolve_scorer_settings(
        {
            "GTRS_EP_EXPONENT": "10",
            "GTRS_SPEED_TOP_K": "32",
            "GTRS_SPEED_WEIGHT": "0.1",
        }
    )

    assert (settings.ep_exponent, settings.speed_top_k, settings.speed_weight) == (
        10.0,
        32,
        0.1,
    )


def test_speed_enhancement_rejects_invalid_switch() -> None:
    with pytest.raises(
        ValueError,
        match="^GTRS_SPEED_ENHANCEMENT must be 0 or 1$",
    ):
        driver_module._resolve_scorer_settings({"GTRS_SPEED_ENHANCEMENT": "true"})


def test_main_wires_server_policy_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPASIM_DRIVER_HOST", "127.0.0.9")
    monkeypatch.setenv("ALPASIM_DRIVER_PORT", "7777")
    monkeypatch.setenv("ALPASIM_DRIVER_GRPC_WORKERS", "6")
    monkeypatch.setenv("GTRS_CHECKPOINT_PATH", "/tmp/test.ckpt")
    monkeypatch.setenv("GTRS_VOCAB_PATH", "/tmp/navhard_8192.npy")
    monkeypatch.setenv("GTRS_DEVICE", "cpu")
    monkeypatch.setenv("GTRS_MAX_BATCH_SIZE", "3")
    monkeypatch.setenv("GTRS_BATCH_WINDOW_MS", "7.5")
    server, _, events, captures = _install_main_fakes(
        monkeypatch,
        bound_port=7777,
    )

    driver_module.main()

    handle = captures["handle"]
    assert isinstance(handle, FakeMainPolicyHandle)
    assert captures["grpc_workers"] == 6
    assert captures["registered_server"] is server
    assert ("bind", "127.0.0.9:7777") in events
    assert events[:2] == ["configure_dirs", "configure_threads"]
    assert "server_start" in events
    assert "server_wait" in events
    assert handle.start_count == 1
    assert handle.stop_count == 1
    assert handle.kwargs["max_batch_size"] == 3
    assert handle.kwargs["batch_window_s"] == pytest.approx(0.0075)
    loader = handle.kwargs["loader"]
    assert callable(loader)
    assert loader() == (
        "/tmp/test.ckpt",
        "/tmp/navhard_8192.npy",
        "cpu",
        "nc_dac_ep",
        3.0,
        64,
        3.0,
        "longitudinal",
        0.0,
        0.0,
        "resnet",
    )
    service = captures["service"]
    assert isinstance(service, NavsimGTRSDenseDriver)
    assert f"signal_{driver_module.signal.SIGTERM}" in captures
    assert f"signal_{driver_module.signal.SIGINT}" in captures


def test_main_wires_experimental_scorer_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GTRS_SCORER_MODE", "nc_dac_ep")
    monkeypatch.setenv("GTRS_EP_EXPONENT", "10")
    monkeypatch.setenv("GTRS_SPEED_TOP_K", "32")
    monkeypatch.setenv("GTRS_SPEED_WEIGHT", "0.1")
    monkeypatch.setenv("GTRS_SPEED_PROXY", "longitudinal_0p5s")
    monkeypatch.setenv("GTRS_CURVATURE_WEIGHT", "0.05")
    monkeypatch.setenv("GTRS_HEADING_CHANGE_WEIGHT", "0.05")
    monkeypatch.setenv("GTRS_TRAJECTORY_TIME_SCALE", "1.10")
    _, _, _, captures = _install_main_fakes(monkeypatch, bound_port=6789)

    driver_module.main()

    assert captures["handle"].kwargs["loader"]()[-8:] == (
        "nc_dac_ep",
        10.0,
        32,
        0.1,
        "longitudinal_0p5s",
        0.05,
        0.05,
        "resnet",
    )
    service = captures["service"]
    assert isinstance(service, NavsimGTRSDenseDriver)
    assert service._trajectory_time_scale == pytest.approx(1.10)


def test_main_wires_safety_gated_progress_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GTRS_SCORER_MODE", "safety_gate_ep")
    monkeypatch.setenv("GTRS_EP_EXPONENT", "3")
    monkeypatch.setenv("GTRS_SPEED_TOP_K", "64")
    monkeypatch.setenv("GTRS_SPEED_WEIGHT", "3")
    monkeypatch.setenv("GTRS_SPEED_PROXY", "longitudinal")
    _, _, _, captures = _install_main_fakes(monkeypatch, bound_port=6789)

    driver_module.main()

    assert captures["handle"].kwargs["loader"]()[-8:] == (
        "safety_gate_ep",
        3.0,
        64,
        3.0,
        "longitudinal",
        0.0,
        0.0,
        "resnet",
    )


def test_main_uses_resnet_checkpoint_default_when_environment_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GTRS_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("GTRS_VOCAB_PATH", raising=False)
    monkeypatch.delenv("GTRS_DEVICE", raising=False)
    _, _, _, captures = _install_main_fakes(monkeypatch, bound_port=6789)

    driver_module.main()

    handle = captures["handle"]
    assert isinstance(handle, FakeMainPolicyHandle)
    loader = handle.kwargs["loader"]
    assert callable(loader)
    assert loader() == (
        "/app/assets/gtrs_dense/gtrs_dense_resnet_sim_reward_navhard.ckpt",
        "/app/assets/gtrs_dense/navsim_16384.npy",
        "cuda",
        "nc_dac_ep",
        3.0,
        64,
        3.0,
        "longitudinal",
        0.0,
        0.0,
        "resnet",
    )
    service = captures["service"]
    assert isinstance(service, NavsimGTRSDenseDriver)
    assert service._trajectory_time_scale == 1.0


def test_main_uses_vov_reward_checkpoint_default_for_vov_backbone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GTRS_BACKBONE", "vov")
    monkeypatch.delenv("GTRS_CHECKPOINT_PATH", raising=False)
    _, _, _, captures = _install_main_fakes(monkeypatch, bound_port=6789)

    driver_module.main()

    policy_args = captures["handle"].kwargs["loader"]()
    assert policy_args[0] == (
        "/app/assets/gtrs_dense/gtrs_dense_vov_sim_reward_navhard.ckpt"
    )
    assert policy_args[-1] == "vov"


def test_main_rejects_invalid_scorer_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GTRS_SCORER_MODE", "invalid")
    _install_main_fakes(monkeypatch, bound_port=6789)

    with pytest.raises(ValueError, match="scorer_mode must be one of"):
        driver_module.main()


@pytest.mark.parametrize("scale", ["nan", "inf", "0.9", "1.3", "fast"])
def test_main_rejects_invalid_trajectory_time_scale(
    monkeypatch: pytest.MonkeyPatch,
    scale: str,
) -> None:
    monkeypatch.setenv("GTRS_TRAJECTORY_TIME_SCALE", scale)
    _install_main_fakes(monkeypatch, bound_port=6789)

    with pytest.raises(ValueError, match="GTRS_TRAJECTORY_TIME_SCALE"):
        driver_module.main()


def test_main_bind_failure_cleans_up_without_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _, events, captures = _install_main_fakes(
        monkeypatch,
        bound_port=0,
    )

    with pytest.raises(RuntimeError, match="^failed to bind 0.0.0.0:6789$"):
        driver_module.main()

    handle = captures["handle"]
    assert isinstance(handle, FakeMainPolicyHandle)
    assert captures["registered_server"] is server
    assert handle.start_count == 0
    assert handle.stop_count == 1
    assert "server_start" not in events
    assert "server_wait" not in events
    assert ("server_stop", 0.0) in events
