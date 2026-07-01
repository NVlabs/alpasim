# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Black-box gRPC tests for the CATK traffic service."""

from __future__ import annotations

import threading
from concurrent import futures
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
import torch
from alpasim_grpc.v0 import common_pb2, traffic_pb2, traffic_pb2_grpc
from alpasim_runtime.errors import UnknownSceneError
from alpasim_trafficsim.grpc.config import CatkLoaderConfig
from alpasim_trafficsim.grpc.servicer import TrafficServiceServicer
from alpasim_trafficsim.grpc.session.history import merge_trajectory
from alpasim_utils.geometry import trajectory_from_grpc

import grpc

DT_US = 100_000
MIN_HISTORY = 16
DEFAULT_SCENE_ID = "clipgt-test-scene"


class FakeSceneLoader:
    """Returns scene data source handles for a fixed set of scene IDs."""

    def __init__(self, scene_ids: list[str] | None = None) -> None:
        self.scene_ids = set(scene_ids or [DEFAULT_SCENE_ID])

    def get_data_source(self, scene_id: str) -> str:
        if scene_id not in self.scene_ids:
            raise UnknownSceneError(scene_id)
        return scene_id


class FakeSceneAdapter:
    """Returns a fixed base env_data for any accepted data source."""

    def __init__(self, env_data: dict[str, Any]) -> None:
        self._env_data = env_data

    def load(self, data_source: Any) -> dict[str, Any]:
        del data_source
        return self._env_data


class NoopPredictor:
    """Never predicts; used to exercise the CATK-unavailable error path."""

    def __init__(self, *, predict_static: bool = True) -> None:
        self.predict_static = predict_static

    def run_inference(
        self, env_data: dict[str, Any], *, predict_steps: int
    ) -> dict[str, Any] | None:
        del env_data, predict_steps
        return None


class RecordingPredictor:
    """Records each run_inference (predict_steps, curr_t); empty predictions."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.predict_static = True

    def run_inference(
        self, env_data: dict[str, Any], *, predict_steps: int
    ) -> dict[str, Any] | None:
        self.calls.append((predict_steps, int(env_data["env"].get("curr_t", 0))))
        return {
            "agent_future_xyz": torch.zeros((0, predict_steps, 3)),
            "agent_future_heading": torch.zeros((0, predict_steps)),
            "agent_future_valid_mask": torch.zeros(
                (0, predict_steps), dtype=torch.bool
            ),
        }

    @staticmethod
    def apply_predictions_to_env(
        session_state: Any, *, future_step_indices: list[int], actions: dict[str, Any]
    ) -> list[int]:
        del actions
        env_data = session_state.env_data
        agents = env_data["agents"]
        total_agents = int(agents["xyz"].shape[0])
        if total_agents == 0:
            return future_step_indices
        for step_idx in future_step_indices:
            prev_step_idx = max(step_idx - 1, 0)
            agents["xyz"][:, step_idx, :] = agents["xyz"][:, prev_step_idx, :]
            agents["heading"][:, step_idx] = agents["heading"][:, prev_step_idx]
            agents["valid_mask"][:, step_idx] = agents["valid_mask"][:, prev_step_idx]
        return future_step_indices


class LinearPredictor(RecordingPredictor):
    """Predicts agent x == future-timestamp-in-seconds (a unique value per step)."""

    def run_inference(
        self, env_data: dict[str, Any], *, predict_steps: int
    ) -> dict[str, Any] | None:
        curr_t = int(env_data["env"].get("curr_t", 0))
        self.calls.append((predict_steps, curr_t))
        sample_start_t_us = int(env_data["env"]["sample_start_t_us"])
        current_ts_us = sample_start_t_us + curr_t * DT_US
        future_ts_s = torch.tensor(
            [
                (current_ts_us + ((offset + 1) * DT_US)) / 1_000_000.0
                for offset in range(predict_steps)
            ],
            dtype=torch.float32,
        )
        xyz = torch.zeros((1, predict_steps, 3), dtype=torch.float32)
        xyz[0, :, 0] = future_ts_s
        return {
            "agent_future_xyz": xyz,
            "agent_future_heading": torch.zeros((1, predict_steps)),
            "agent_future_valid_mask": torch.ones((1, predict_steps), dtype=torch.bool),
        }

    @staticmethod
    def apply_predictions_to_env(
        session_state: Any, *, future_step_indices: list[int], actions: dict[str, Any]
    ) -> None:
        env_data = session_state.env_data
        for offset, step_idx in enumerate(future_step_indices):
            env_data["agents"]["xyz"][0, step_idx, :] = actions["agent_future_xyz"][
                0, offset, :
            ]
            env_data["agents"]["heading"][0, step_idx] = actions[
                "agent_future_heading"
            ][0, offset]
            env_data["agents"]["valid_mask"][0, step_idx] = actions[
                "agent_future_valid_mask"
            ][0, offset]


class BlockingPredictor(RecordingPredictor):
    """Blocks the first inference until released so concurrency can be observed."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.first_started = threading.Event()
        self.second_started = threading.Event()
        self.release_first = threading.Event()

    def run_inference(
        self, env_data: dict[str, Any], *, predict_steps: int
    ) -> dict[str, Any] | None:
        with self._lock:
            self.calls.append((predict_steps, int(env_data["env"].get("curr_t", 0))))
            call_count = len(self.calls)
        if call_count == 1:
            self.first_started.set()
            if not self.release_first.wait(timeout=2.0):
                raise TimeoutError("timed out waiting to release first inference")
        elif call_count == 2:
            self.second_started.set()
        return {
            "agent_future_xyz": torch.zeros((0, predict_steps, 3)),
            "agent_future_heading": torch.zeros((0, predict_steps)),
            "agent_future_valid_mask": torch.zeros(
                (0, predict_steps), dtype=torch.bool
            ),
        }


class ExplodingPredictor(NoopPredictor):
    """Raises during inference to exercise the simulate INTERNAL error path."""

    def run_inference(
        self, env_data: dict[str, Any], *, predict_steps: int
    ) -> dict[str, Any] | None:
        raise RuntimeError("boom: model inference failed")


def _make_env_data(*, num_agents: int = 0, steps: int = 1) -> dict[str, Any]:
    return {
        "env": {},
        "metadata": {
            "t0_us": 0,
            "obstacle_class_name_2_id": {"car": 1, "others": 0},
        },
        "ego": {
            "xyz": torch.zeros((steps, 3), dtype=torch.float32),
            "heading": torch.zeros((steps,), dtype=torch.float32),
            "lwh": torch.tensor([4.5, 2.0, 1.7], dtype=torch.float32),
        },
        "agents": {
            "xyz": torch.zeros((num_agents, steps, 3), dtype=torch.float32),
            "heading": torch.zeros((num_agents, steps), dtype=torch.float32),
            "valid_mask": torch.zeros((num_agents, steps), dtype=torch.bool),
            "lwh": torch.ones((num_agents, 3), dtype=torch.float32),
            "track_ids": torch.arange(1, num_agents + 1, dtype=torch.long),
            "class_ids": torch.ones((num_agents,), dtype=torch.long),
            "num_obstacles": num_agents,
        },
        "map": {},
    }


def _pose(
    timestamp_us: int, *, x: float | None = None, y: float = 0.0, z: float = 0.0
) -> common_pb2.PoseAtTime:
    x = float(timestamp_us) / 1_000_000.0 if x is None else float(x)
    return common_pb2.PoseAtTime(
        timestamp_us=timestamp_us,
        pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=x, y=y, z=z),
            quat=common_pb2.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
    )


def _ego_trajectory(*, end_us: int = 3_000_000) -> common_pb2.Trajectory:
    trajectory = common_pb2.Trajectory()
    trajectory.poses.extend(_pose(ts) for ts in range(0, end_us + 1, DT_US))
    return trajectory


def _ego_object() -> traffic_pb2.ObjectTrajectory:
    return traffic_pb2.ObjectTrajectory(
        object_id="EGO",
        aabb=common_pb2.AABB(size_x=4.5, size_y=2.0, size_z=1.7),
        trajectory=_ego_trajectory(),
    )


def _moving_object(object_id: str = "moving-1") -> traffic_pb2.ObjectTrajectory:
    return traffic_pb2.ObjectTrajectory(
        object_id=object_id,
        aabb=common_pb2.AABB(size_x=4.5, size_y=2.0, size_z=1.7),
        trajectory=_ego_trajectory(),
        is_static=False,
    )


def _static_object(
    *, object_id: str = "static-1", x: float = 10.0, y: float = 1.0, z: float = 0.5
) -> traffic_pb2.ObjectTrajectory:
    return traffic_pb2.ObjectTrajectory(
        object_id=object_id,
        aabb=common_pb2.AABB(size_x=4.5, size_y=2.0, size_z=1.7),
        trajectory=common_pb2.Trajectory(poses=[_pose(0, x=x, y=y, z=z)]),
        is_static=True,
    )


def _ego_update(
    timestamp_us: int,
    *,
    end_us: int | None = None,
) -> traffic_pb2.ObjectTrajectoryUpdate:
    poses = [_pose(timestamp_us)]
    if end_us is not None and end_us != timestamp_us:
        poses.append(_pose(end_us))
    return traffic_pb2.ObjectTrajectoryUpdate(
        object_id="EGO",
        trajectory=common_pb2.Trajectory(poses=poses),
    )


def test_merge_trajectory_replaces_existing_future_segment() -> None:
    existing = trajectory_from_grpc(
        common_pb2.Trajectory(
            poses=[
                _pose(100_000, x=1.0),
                _pose(200_000, x=2.0),
                _pose(300_000, x=3.0),
            ]
        )
    )
    incoming = trajectory_from_grpc(
        common_pb2.Trajectory(
            poses=[
                _pose(250_000, x=25.0),
                _pose(350_000, x=35.0),
            ]
        )
    )
    trajectories = {"agent-1": existing}

    merge_trajectory(trajectories, object_id="agent-1", trajectory=incoming)

    merged = trajectories["agent-1"]
    assert merged.timestamps_us.tolist() == [100_000, 200_000, 250_000, 350_000]
    assert merged.positions[:, 0].tolist() == pytest.approx([1.0, 2.0, 25.0, 35.0])


def _session_request(
    *,
    logged: list[traffic_pb2.ObjectTrajectory],
    handover_time_us: int = 1_500_000,
    session_uuid: str = "session-1",
    scene_id: str = DEFAULT_SCENE_ID,
) -> traffic_pb2.TrafficSessionRequest:
    return traffic_pb2.TrafficSessionRequest(
        session_uuid=session_uuid,
        scene_id=scene_id,
        random_seed=7,
        logged_object_trajectories=logged,
        handover_time_us=handover_time_us,
    )


def _simulate_request(
    time_query_us: int,
    *,
    session_uuid: str = "session-1",
    ego_update_end_us: int | None = None,
) -> traffic_pb2.TrafficRequest:
    return traffic_pb2.TrafficRequest(
        session_uuid=session_uuid,
        time_query_us=time_query_us,
        object_trajectory_updates=[
            _ego_update(time_query_us, end_us=ego_update_end_us)
        ],
    )


class RunningService:
    """A started gRPC service: a real client stub plus the servicer for asserts."""

    def __init__(self, stub: Any, servicer: TrafficServiceServicer) -> None:
        self.stub = stub
        self.servicer = servicer

    def session_state(self, session_uuid: str = "session-1") -> Any:
        return self.servicer._sessions[session_uuid]


def _build_servicer(
    *,
    predictor: Any,
    env_data: dict[str, Any] | None = None,
    scene_loader: Any | None = None,
) -> TrafficServiceServicer:
    """Construct a servicer with injected fakes, bypassing loader/model setup."""
    servicer = TrafficServiceServicer.__new__(TrafficServiceServicer)
    servicer._server = None
    servicer._lock = threading.Lock()
    servicer._sessions = {}
    servicer._service_version = "test-traffic-service"
    servicer._time_step_s = DT_US / 1e6
    servicer._dt_us = DT_US
    servicer._minimum_history_length = MIN_HISTORY
    servicer._minimum_future_steps = 5
    servicer._loader_cfg = CatkLoaderConfig(usdz_folder="unused")
    servicer._scene_loader = scene_loader or FakeSceneLoader()
    servicer._scene_adapter = FakeSceneAdapter(env_data or _make_env_data())
    servicer._catk_predictor = predictor
    return servicer


@contextmanager
def _serve(
    predictor: Any | None = None,
    *,
    max_workers: int = 4,
    scene_loader: Any | None = None,
) -> Iterator[RunningService]:
    servicer = _build_servicer(
        predictor=predictor or RecordingPredictor(),
        scene_loader=scene_loader,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    traffic_pb2_grpc.add_TrafficServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = traffic_pb2_grpc.TrafficServiceStub(channel)
        yield RunningService(stub, servicer)
    finally:
        channel.close()
        server.stop(0)


@pytest.fixture
def service() -> Iterator[RunningService]:
    with _serve() as running:
        yield running


def test_get_metadata_reports_service_version(
    service: RunningService,
) -> None:
    metadata = service.stub.get_metadata(common_pb2.Empty())
    assert metadata.version_id.version_id == "test-traffic-service"
    assert metadata.minimum_history_length_us == MIN_HISTORY * DT_US


def test_get_available_scenes_reports_scene_ids(service: RunningService) -> None:
    response = service.stub.get_available_scenes(common_pb2.Empty())
    assert DEFAULT_SCENE_ID in response.scene_ids


def test_start_then_close_session_round_trips(service: RunningService) -> None:
    service.stub.start_session(_session_request(logged=[_ego_object()]))
    assert "session-1" in service.servicer._sessions

    service.stub.close_session(
        traffic_pb2.TrafficSessionCloseRequest(session_uuid="session-1")
    )
    assert "session-1" not in service.servicer._sessions


def test_start_session_rejects_empty_uuid(service: RunningService) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        service.stub.start_session(
            _session_request(logged=[_ego_object()], session_uuid="")
        )
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_close_unknown_session_returns_not_found(service: RunningService) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        service.stub.close_session(
            traffic_pb2.TrafficSessionCloseRequest(session_uuid="ghost")
        )
    assert exc.value.code() == grpc.StatusCode.NOT_FOUND


def test_simulate_unknown_session_returns_not_found(service: RunningService) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        service.stub.simulate(_simulate_request(1_500_000, session_uuid="ghost"))
    assert exc.value.code() == grpc.StatusCode.NOT_FOUND


def test_start_session_rejects_empty_scene_id(service: RunningService) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        service.stub.start_session(
            _session_request(logged=[_ego_object()], scene_id="")
        )
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_start_session_rejects_no_logged_trajectories(
    service: RunningService,
) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        service.stub.start_session(_session_request(logged=[]))
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_start_session_rejects_nonpositive_handover_time(
    service: RunningService,
) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        service.stub.start_session(
            _session_request(
                logged=[_ego_object()],
                handover_time_us=0,
            )
        )
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "handover_time_us" in exc.value.details()


def test_start_session_requires_logged_ego_trajectory(
    service: RunningService,
) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        service.stub.start_session(_session_request(logged=[_moving_object()]))
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "EGO" in exc.value.details()


def test_start_session_rejects_empty_logged_ego_trajectory(
    service: RunningService,
) -> None:
    empty_ego = traffic_pb2.ObjectTrajectory(
        object_id="EGO",
        aabb=common_pb2.AABB(size_x=4.5, size_y=2.0, size_z=1.7),
        trajectory=common_pb2.Trajectory(),
    )

    with pytest.raises(grpc.RpcError) as exc:
        service.stub.start_session(_session_request(logged=[empty_ego]))
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "EGO" in exc.value.details()


def test_start_session_unknown_scene_returns_not_found() -> None:
    with _serve(scene_loader=FakeSceneLoader(["clipgt-known-scene"])) as service:
        with pytest.raises(grpc.RpcError) as exc:
            service.stub.start_session(
                _session_request(
                    logged=[_ego_object()],
                    scene_id="clipgt-does-not-exist",
                )
            )
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND


def test_simulate_inference_failure_returns_internal() -> None:
    with _serve(ExplodingPredictor()) as service:
        service.stub.start_session(_session_request(logged=[_ego_object()]))
        with pytest.raises(grpc.RpcError) as exc:
            service.stub.simulate(_simulate_request(2_000_000))
        assert exc.value.code() == grpc.StatusCode.INTERNAL


def test_simulate_missing_catk_predictions_returns_failed_precondition() -> None:
    with _serve(NoopPredictor()) as service:
        service.stub.start_session(_session_request(logged=[_ego_object()]))
        with pytest.raises(grpc.RpcError) as exc:
            service.stub.simulate(_simulate_request(2_000_000))
        assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "CATK did not produce predictions" in exc.value.details()


def test_first_simulate_accepts_initial_history_time(service: RunningService) -> None:
    service.stub.start_session(_session_request(logged=[_ego_object()]))
    initial_ts_us = service.session_state().current_ts_us
    assert initial_ts_us == 1_500_000

    service.stub.simulate(_simulate_request(initial_ts_us))
    assert service.session_state().env_data["env"]["curr_t"] == 15


def test_static_agent_with_one_pose_survives_model_simulate(
    service: RunningService,
) -> None:
    service.stub.start_session(
        _session_request(logged=[_ego_object(), _static_object()])
    )
    response = service.stub.simulate(_simulate_request(1_600_000))

    assert service.session_state().env_data["agents"]["valid_mask"][0, :17].all()
    assert response.object_trajectory_updates[0].object_id == "static-1"
    pose = response.object_trajectory_updates[0].trajectory.poses[0]
    assert pose.timestamp_us == 1_600_000
    assert pose.pose.vec.x == pytest.approx(10.0)
    assert pose.pose.vec.y == pytest.approx(1.0)


def test_dynamic_response_includes_available_prediction_horizon() -> None:
    with _serve(LinearPredictor()) as service:
        service.stub.start_session(
            _session_request(logged=[_ego_object(), _moving_object()])
        )
        response = service.stub.simulate(_simulate_request(1_600_000))

        assert len(response.object_trajectory_updates) == 1
        trajectory = response.object_trajectory_updates[0].trajectory
        assert [pose.timestamp_us for pose in trajectory.poses] == [
            1_600_000,
            1_700_000,
            1_800_000,
            1_900_000,
            2_000_000,
        ]
        assert [pose.pose.vec.x for pose in trajectory.poses] == pytest.approx(
            [1.6, 1.7, 1.8, 1.9, 2.0]
        )


def test_logged_replay_does_not_extrapolate_past_track_end(
    service: RunningService,
) -> None:
    short_moving = traffic_pb2.ObjectTrajectory(
        object_id="moving-1",
        aabb=common_pb2.AABB(size_x=4.5, size_y=2.0, size_z=1.7),
        trajectory=common_pb2.Trajectory(
            poses=[_pose(0, x=10.0), _pose(100_000, x=11.0)]
        ),
        is_static=False,
    )
    service.stub.start_session(
        _session_request(
            logged=[_ego_object(), short_moving], handover_time_us=2_000_000
        )
    )
    response = service.stub.simulate(_simulate_request(1_600_000))
    assert list(response.object_trajectory_updates) == []


def test_logged_traffic_used_until_handover_then_model_runs() -> None:
    predictor = RecordingPredictor()
    with _serve(predictor) as service:
        service.stub.start_session(
            _session_request(
                logged=[_ego_object(), _moving_object()], handover_time_us=2_000_000
            )
        )

        before = service.stub.simulate(_simulate_request(1_600_000))
        assert predictor.calls == []
        assert before.object_trajectory_updates[0].object_id == "moving-1"
        pose = before.object_trajectory_updates[0].trajectory.poses[0]
        assert pose.pose.vec.x == pytest.approx(1.6)

        service.stub.simulate(_simulate_request(2_100_000))
        assert predictor.calls == [(5, MIN_HISTORY - 1)]


def test_off_grid_handover_uses_exact_anchor_for_catk() -> None:
    predictor = RecordingPredictor()
    with _serve(predictor) as service:
        service.stub.start_session(
            _session_request(
                logged=[_ego_object(), _moving_object()], handover_time_us=2_050_000
            )
        )
        state = service.session_state()

        service.stub.simulate(_simulate_request(2_060_000))
        assert predictor.calls == [(5, MIN_HISTORY - 1)]
        assert state.current_ts_us == 2_060_000
        assert state.env_data["env"]["sample_start_t_us"] == 550_000
        assert state.env_data["env"]["curr_t"] == MIN_HISTORY

        service.stub.simulate(_simulate_request(2_200_000))
        assert predictor.calls == [(5, MIN_HISTORY - 1), (5, MIN_HISTORY - 1)]
        assert state.env_data["env"]["sample_start_t_us"] == 560_000
        assert state.env_data["env"]["curr_t"] == MIN_HISTORY + 1


def test_off_grid_request_keeps_query_time_as_session_time(
    service: RunningService,
) -> None:
    service.stub.start_session(_session_request(logged=[_ego_object()]))
    service.stub.simulate(_simulate_request(2_020_000, ego_update_end_us=2_100_000))

    state = service.session_state()
    assert state.env_data["env"]["curr_t"] == 21
    assert state.current_ts_us == 2_020_000
    assert state.env_data["ego"]["xyz"][21, 0].item() == pytest.approx(2.1)


def test_off_grid_request_accepts_single_ego_pose_at_query_time(
    service: RunningService,
) -> None:
    service.stub.start_session(_session_request(logged=[_ego_object()]))
    service.stub.simulate(_simulate_request(2_020_000))

    state = service.session_state()
    assert state.env_data["env"]["curr_t"] == 21
    assert state.current_ts_us == 2_020_000
    assert state.env_data["ego"]["xyz"][20, 0].item() == pytest.approx(2.0)
    assert state.env_data["ego"]["xyz"][21, 0].item() == pytest.approx(2.02)


def test_off_grid_response_is_interpolated_at_requested_time() -> None:
    with _serve(LinearPredictor()) as service:
        service.stub.start_session(
            _session_request(logged=[_ego_object(), _moving_object()])
        )
        response = service.stub.simulate(
            _simulate_request(2_020_000, ego_update_end_us=2_100_000)
        )

        assert len(response.object_trajectory_updates) == 1
        pose = response.object_trajectory_updates[0].trajectory.poses[0]
        assert pose.timestamp_us == 2_020_000
        assert pose.pose.vec.x == pytest.approx(2.02)


def test_consecutive_off_grid_requests_resample_from_latest_query_time() -> None:
    with _serve(LinearPredictor()) as service:
        service.stub.start_session(
            _session_request(logged=[_ego_object(), _moving_object()])
        )
        state = service.session_state()

        for query_ts_us in (2_020_000, 2_530_000, 3_040_000):
            previous_current_ts_us = state.current_ts_us
            assert previous_current_ts_us is not None
            ego_update_end_us = ((query_ts_us + DT_US - 1) // DT_US) * DT_US
            service.stub.simulate(
                _simulate_request(query_ts_us, ego_update_end_us=ego_update_end_us)
            )
            assert state.current_ts_us == query_ts_us
            assert state.env_data["env"]["sample_start_t_us"] == (
                previous_current_ts_us - ((MIN_HISTORY - 1) * DT_US)
            )

        assert state.current_ts_us == 3_040_000


def test_simulation_extends_current_sample_window() -> None:
    with _serve(LinearPredictor()) as service:
        service.stub.start_session(
            _session_request(logged=[_ego_object(), _moving_object()])
        )
        state = service.session_state()
        assert state.env_data["env"]["sample_start_t_us"] == 0

        service.stub.simulate(_simulate_request(2_500_000))

        assert state.env_data["env"]["curr_t"] == 25
        assert state.env_data["env"]["sample_start_t_us"] == 0
        assert state.env_data["agents"]["xyz"].shape[1] >= 26
        assert state.env_data["agents"]["xyz"][0, 25, 0].item() == pytest.approx(2.5)


def test_handover_resamples_logged_history_before_catk() -> None:
    with _serve(LinearPredictor()) as service:
        service.stub.start_session(
            _session_request(
                logged=[_ego_object(), _moving_object()], handover_time_us=2_000_000
            )
        )
        state = service.session_state()
        assert state.env_data["env"]["sample_start_t_us"] == 0
        assert state.env_data["env"]["curr_t"] == 15
        assert state.env_data["agents"]["xyz"].shape[1] == MIN_HISTORY

        service.stub.simulate(_simulate_request(2_100_000))

        assert state.env_data["env"]["sample_start_t_us"] == 500_000
        assert state.env_data["env"]["curr_t"] == MIN_HISTORY
        assert state.env_data["agents"]["xyz"][0, 15, 0].item() == pytest.approx(2.0)
        assert state.env_data["agents"]["xyz"][0, 16, 0].item() == pytest.approx(2.1)


def test_multi_token_horizon_uses_single_inference_call() -> None:
    predictor = RecordingPredictor()
    with _serve(predictor) as service:
        service.stub.start_session(_session_request(logged=[_ego_object()]))
        service.stub.simulate(_simulate_request(2_300_000))

        assert len(predictor.calls) == 1
        assert predictor.calls[0] == (8, 15)
        assert service.session_state().env_data["env"]["curr_t"] == 23


def test_concurrent_simulate_for_same_session_is_serialized() -> None:
    predictor = BlockingPredictor()
    with _serve(predictor, max_workers=4) as service:
        service.stub.start_session(_session_request(logged=[_ego_object()]))

        errors: list[BaseException] = []

        def call(timestamp_us: int) -> None:
            try:
                service.stub.simulate(_simulate_request(timestamp_us))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        first = threading.Thread(target=call, args=(2_000_000,))
        second = threading.Thread(target=call, args=(2_100_000,))

        first.start()
        assert predictor.first_started.wait(timeout=2.0)
        second.start()
        assert not predictor.second_started.wait(timeout=0.3)

        predictor.release_first.set()
        first.join(timeout=3.0)
        second.join(timeout=3.0)

        assert not first.is_alive() and not second.is_alive()
        assert errors == []
        assert predictor.calls == [(5, 15), (5, 15)]
        assert service.session_state().current_ts_us == 2_100_000
