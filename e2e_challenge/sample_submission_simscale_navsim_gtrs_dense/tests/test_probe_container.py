# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

SAMPLE = Path(__file__).resolve().parents[1]
SCRIPT = SAMPLE / "scripts/probe_container.py"
sys.path.insert(0, str(SAMPLE / "scripts"))

import probe_container  # noqa: E402


def test_probe_requires_resnet_release_version() -> None:
    assert probe_container.VERSION_ID == "simscale-gtrs-dense-e2e"


def test_probe_accepts_fixed_version_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "simscale-gtrs-dense-resnet-expert-e2e"
    monkeypatch.setenv("GTRS_PROBE_VERSION_ID", expected)

    assert runpy.run_path(str(SCRIPT))["VERSION_ID"] == expected


class FakeStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, float]] = []

    def _record(self, name: str, request: Any, timeout: float) -> None:
        self.calls.append((name, request, timeout))

    def start_session(self, request: Any, *, timeout: float) -> None:
        self._record("start_session", request, timeout)

    def submit_egomotion_observation(
        self,
        request: Any,
        *,
        timeout: float,
    ) -> None:
        self._record("submit_egomotion_observation", request, timeout)

    def submit_route(self, request: Any, *, timeout: float) -> None:
        self._record("submit_route", request, timeout)

    def submit_image_observation(
        self,
        request: Any,
        *,
        timeout: float,
    ) -> None:
        self._record("submit_image_observation", request, timeout)


def test_jpeg_bytes_has_expected_dimensions() -> None:
    with Image.open(BytesIO(probe_container.jpeg_bytes(32))) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (1920, 1080)


def test_start_request_contains_all_camera_intrinsics() -> None:
    request = probe_container.start_request("session", 17)

    assert request.session_uuid == "session"
    assert request.random_seed == 17
    cameras = request.rollout_spec.vehicle.available_cameras
    assert tuple(camera.logical_id for camera in cameras) == probe_container.CAMERAS
    for camera in cameras:
        assert camera.intrinsics.logical_id == camera.logical_id
        assert camera.intrinsics.resolution_h == 1080
        assert camera.intrinsics.resolution_w == 1920


def test_seed_session_submits_complete_observation_set() -> None:
    stub = FakeStub()

    probe_container.seed_session(
        stub,
        "session",
        100.0,
        64,
        -5.0,
        12.5,
    )

    assert [name for name, _, _ in stub.calls] == [
        "start_session",
        "submit_egomotion_observation",
        "submit_route",
        *("submit_image_observation" for _ in probe_container.CAMERAS),
    ]
    assert all(timeout == 12.5 for _, _, timeout in stub.calls)

    start = stub.calls[0][1]
    assert start.session_uuid == "session"
    assert start.random_seed == 64

    egomotion = stub.calls[1][1]
    pose = egomotion.trajectory.poses[0]
    assert pose.timestamp_us == probe_container.NOW_US
    assert pose.pose.vec.x == 100.0
    assert pose.pose.quat.w == 1.0
    assert egomotion.dynamic_states[0].linear_velocity.x == 5.0

    route = stub.calls[2][1].route
    assert route.timestamp_us == probe_container.NOW_US
    assert route.waypoints[0].x == 20.0
    assert route.waypoints[0].y == -5.0

    image_calls = stub.calls[3:]
    assert (
        tuple(request.camera_image.logical_id for _, request, _ in image_calls)
        == probe_container.CAMERAS
    )
    for _, request, _ in image_calls:
        assert request.session_uuid == "session"
        assert request.camera_image.frame_start_us == probe_container.NOW_US - 30_000
        assert request.camera_image.frame_end_us == probe_container.NOW_US
        assert request.camera_image.image_bytes


def _run_probe(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"PYTHONPATH": str(SAMPLE)}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_probe_cli_help_does_not_contact_server() -> None:
    result = _run_probe("--help")

    assert result.returncode == 0
    assert "--address" in result.stdout
    assert "--timeout" in result.stdout


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "-inf"])
def test_probe_cli_rejects_non_positive_timeout(timeout: str) -> None:
    result = _run_probe(f"--timeout={timeout}")

    assert result.returncode == 2
    assert "timeout must be finite and greater than 0" in result.stderr


def test_require_raises_runtime_error() -> None:
    probe_container._require(True, "unused")

    with pytest.raises(RuntimeError, match="^stable probe failure$"):
        probe_container._require(False, "stable probe failure")


def _drive_response(
    timestamps: list[int],
    *,
    first_x: float = 0.0,
) -> probe_container.egodriver_pb2.DriveResponse:
    return probe_container.egodriver_pb2.DriveResponse(
        trajectory=probe_container.common_pb2.Trajectory(
            poses=[
                probe_container.common_pb2.PoseAtTime(
                    timestamp_us=timestamp,
                    pose=probe_container.common_pb2.Pose(
                        vec=probe_container.common_pb2.Vec3(
                            x=first_x if index == 0 else float(index)
                        )
                    ),
                )
                for index, timestamp in enumerate(timestamps)
            ]
        )
    )


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        (
            [probe_container.NOW_US, probe_container.QUERY_US],
            "exactly 41 points",
        ),
        (
            [
                *(probe_container.NOW_US + index * 100_000 for index in range(39)),
                5_000_000,
            ],
            "exactly 41 points",
        ),
        (
            [probe_container.NOW_US + index * 97_500 for index in range(41)],
            "end at 5000000us",
        ),
    ],
    ids=["two-points", "forty-points", "wrong-final-time"],
)
def test_trajectory_summary_rejects_wrong_observed_contract(
    timestamps: list[int],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        probe_container.trajectory_summary(
            _drive_response(timestamps),
            "probe-a",
            0.0,
        )


def test_trajectory_summary_accepts_exact_observed_contract() -> None:
    timestamps = [probe_container.NOW_US + index * 100_000 for index in range(41)]

    assert probe_container.trajectory_summary(
        _drive_response(timestamps),
        "probe-a",
        0.0,
    ) == {
        "session": "probe-a",
        "points": 41,
        "last_us": 5_000_000,
    }


def test_trajectory_summary_rejects_non_finite_pose() -> None:
    timestamps = [probe_container.NOW_US + index * 100_000 for index in range(41)]
    response = _drive_response(
        timestamps,
        first_x=float("nan"),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        probe_container.trajectory_summary(response, "probe-a", 0.0)


def test_probe_main_success_emits_json_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeChannel:
        closed = False

        def close(self) -> None:
            self.closed = True

    class ReadyFuture:
        def result(self, *, timeout: float) -> None:
            assert timeout == 7.5

    class LiveStub(FakeStub):
        def __init__(self) -> None:
            super().__init__()
            self.version_timeout: float | None = None
            self.started: list[str] = []
            self.closed: list[str] = []
            self.image_sessions: list[str] = []
            self.egomotion_sessions: list[str] = []
            self.route_sessions: list[str] = []

        def get_version(
            self,
            request: Any,
            *,
            wait_for_ready: bool,
            timeout: float,
        ) -> Any:
            assert wait_for_ready
            self.version_timeout = timeout
            return probe_container.common_pb2.VersionId(
                version_id=probe_container.VERSION_ID,
                grpc_api_version=probe_container.API_VERSION_MESSAGE,
            )

        def start_session(self, request: Any, *, timeout: float) -> None:
            super().start_session(request, timeout=timeout)
            self.started.append(request.session_uuid)

        def submit_egomotion_observation(
            self,
            request: Any,
            *,
            timeout: float,
        ) -> None:
            super().submit_egomotion_observation(request, timeout=timeout)
            self.egomotion_sessions.append(request.session_uuid)

        def submit_route(self, request: Any, *, timeout: float) -> None:
            super().submit_route(request, timeout=timeout)
            self.route_sessions.append(request.session_uuid)

        def submit_image_observation(
            self,
            request: Any,
            *,
            timeout: float,
        ) -> None:
            super().submit_image_observation(request, timeout=timeout)
            self.image_sessions.append(request.session_uuid)

        def drive(self, request: Any, *, timeout: float) -> Any:
            assert timeout == 7.5
            expected_x = 0.0 if request.session_uuid == "probe-a" else 100.0
            return _drive_response(
                [probe_container.NOW_US + index * 100_000 for index in range(41)],
                first_x=expected_x,
            )

        def close_session(self, request: Any, *, timeout: float) -> None:
            assert timeout == 10.0
            self.closed.append(request.session_uuid)

    channel = FakeChannel()
    stub = LiveStub()
    monkeypatch.setattr(
        probe_container.grpc,
        "insecure_channel",
        lambda address: channel,
    )
    monkeypatch.setattr(
        probe_container.grpc,
        "channel_ready_future",
        lambda actual: ReadyFuture(),
    )
    monkeypatch.setattr(
        probe_container.egodriver_pb2_grpc,
        "EgodriverServiceStub",
        lambda actual: stub,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--address", "fake:123", "--timeout", "7.5"],
    )

    probe_container.main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "sessions": [
            {"last_us": 5_000_000, "points": 41, "session": "probe-a"},
            {"last_us": 5_000_000, "points": 41, "session": "probe-b"},
        ],
        "version": probe_container.VERSION_ID,
    }
    assert stub.version_timeout == 7.5
    assert stub.started == ["probe-a", "probe-b"]
    assert stub.egomotion_sessions == ["probe-a", "probe-b"]
    assert stub.route_sessions == ["probe-a", "probe-b"]
    assert stub.image_sessions == ["probe-a"] * 8 + ["probe-b"] * 8
    assert stub.closed == ["probe-a", "probe-b"]
    assert channel.closed


def test_probe_closes_channel_when_readiness_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeChannel:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FailedFuture:
        def result(self, *, timeout: float) -> None:
            assert timeout == 3.0
            raise RuntimeError("not ready")

    channel = FakeChannel()
    monkeypatch.setattr(
        probe_container.grpc,
        "insecure_channel",
        lambda address: channel,
    )
    monkeypatch.setattr(
        probe_container.grpc,
        "channel_ready_future",
        lambda actual: FailedFuture(),
    )
    monkeypatch.setattr(
        probe_container.egodriver_pb2_grpc,
        "EgodriverServiceStub",
        lambda actual: object(),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--timeout", "3"])

    with pytest.raises(RuntimeError, match="not ready"):
        probe_container.main()

    assert channel.closed
