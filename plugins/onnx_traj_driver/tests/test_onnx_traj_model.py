# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""OnnxTrajModel against a tiny stand-in graph (see fixtures/make_dummy_graphs.py)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from alpasim_driver.models.base import CameraFrame, DriveCommand, PredictionInput
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Vec3
from alpasim_grpc.v0.egodriver_pb2 import Route
from alpasim_onnx_traj.onnx_traj_model import KV_RING_KEY, OnnxTrajModel
from alpasim_onnx_traj.rsim_port import frame_rope_cos_sin

FIXTURES = Path(__file__).parent / "fixtures"
T0_US = 1_000_000
# The ego drives 5 m along local x between the route's timestamp and t0.
EGO_ADVANCE_M = 5.0


@pytest.fixture(scope="module")
def model() -> OnnxTrajModel:
    return OnnxTrajModel(
        checkpoint_path=str(FIXTURES / "dummy_decoder.onnx"),
        device=torch.device("cpu"),
        camera_ids=["cam_front_left"],
    )


def _pose(timestamp_us: int, x: float) -> PoseAtTime:
    return PoseAtTime(
        timestamp_us=timestamp_us,
        pose=Pose(vec=Vec3(x=x, y=0.0, z=0.0), quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0)),
    )


def _input(
    session_state: dict,
    speed: float = 10.0,
    t0_us: int = T0_US,
    image_hw: tuple[int, int] = (1080, 1920),
) -> PredictionInput:
    # Straight route ahead of the rig at timestamp 0, ending 80 m out.
    route = Route(
        timestamp_us=0,
        waypoints=[Vec3(x=float(x), y=0.0, z=0.0) for x in np.arange(0.0, 84.0, 4.0)],
    )
    return PredictionInput(
        camera_images={
            "cam_front_left": [
                CameraFrame(t0_us, np.full((*image_hw, 3), 128, dtype=np.uint8))
            ]
        },
        command=DriveCommand.STRAIGHT,
        speed=speed,
        acceleration=0.0,
        ego_pose_history=[
            _pose(0, 0.0),
            _pose(t0_us, EGO_ADVANCE_M * t0_us / T0_US),
            _pose(t0_us + 1, EGO_ADVANCE_M * t0_us / T0_US),
        ],
        inference_seed=0,
        previous_plan=None,
        route=route,
        session_state=session_state,
    )


@pytest.fixture
def feeds(model: OnnxTrajModel, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Records every feed the model sends to onnxruntime."""
    recorded: list[dict] = []
    run = model._session.run

    def spy(output_names, feed):  # noqa: ANN001, ANN202
        recorded.append({k: v.copy() for k, v in feed.items()})
        return run(output_names, feed)

    monkeypatch.setattr(model._session, "run", spy)
    return recorded


def test_drives_argmax_mode_in_rig_frame_at_camera_t0(model: OnnxTrajModel) -> None:
    prediction = model.predict(_input({}))

    # Mode logits argmax is 3; its poses are (x_right=1, y_forward=2(p+1)).
    assert prediction.selected_index == 3
    assert prediction.candidate_positions.shape == (5, 5, 3)
    np.testing.assert_allclose(
        prediction.selected_positions[:, :2],
        [[2.0 * (p + 1), -1.0] for p in range(5)],
    )
    np.testing.assert_allclose(
        np.arctan2(
            prediction.selected_rotations[1:, 1, 0],
            prediction.selected_rotations[1:, 0, 0],
        ),
        0.0,
    )
    assert prediction.model_t0_us == T0_US
    np.testing.assert_array_equal(
        prediction.waypoint_timestamps_us, T0_US + 333_333 * np.arange(1, 6)
    )
    np.testing.assert_allclose(
        prediction.pose_local_to_rig_t0.vec3, [EGO_ADVANCE_M, 0.0, 0.0]
    )


def test_feeds_speed_in_kmh_and_route_reanchored_at_t0(
    model: OnnxTrajModel, feeds: list[dict]
) -> None:
    model.predict(_input({}, speed=10.0))

    (feed,) = feeds
    np.testing.assert_allclose(feed["inputs_speed"], [[[36.0]]])
    # The route starts 5 m behind the rig at t0 and ends 75 m ahead; samples
    # sit 1 m ahead of the ego, then every 10 m, clamped to the route's end.
    forward_m = np.minimum(1.0 + 10.0 * np.arange(10), 80.0 - EGO_ADVANCE_M)
    np.testing.assert_allclose(
        feed["inputs_waypoints"][0, 0],
        np.stack([np.zeros(10), forward_m / 100.0], axis=-1),
        atol=1e-6,
    )
    assert feed["inputs_image_cam_front_left"].shape == (1, 1, 3, 8, 8)


def test_kv_cache_persists_per_session_and_is_isolated(
    model: OnnxTrajModel, feeds: list[dict]
) -> None:
    first: dict = {}
    second: dict = {}

    model.predict(_input(first, speed=1.0))
    model.predict(_input(second, speed=5.0))
    model.predict(_input(first, speed=2.0))

    # The dummy graph's new_k echoes the speed fed in (km/h).
    ring = first[KV_RING_KEY]
    assert ring.frame_index == 2
    np.testing.assert_allclose(ring.past_k[..., 0:3, :], 3.6)
    np.testing.assert_allclose(ring.past_k[..., 3:6, :], 7.2)
    other = second[KV_RING_KEY]
    assert other.frame_index == 1
    np.testing.assert_allclose(other.past_k[..., 0:3, :], 18.0)
    np.testing.assert_array_equal(other.cache_bias[..., 3:6], -1e4)
    # The first session's second call saw its own first frame and RoPE index 1.
    np.testing.assert_allclose(feeds[2]["inputs_past_k"][..., 0:3, :], 3.6)
    np.testing.assert_array_equal(
        feeds[2]["inputs_rope_cos"], frame_rope_cos_sin(1, 4)[0]
    )


def test_rejects_wrong_camera_set() -> None:
    with pytest.raises(ValueError, match="cam_front_left"):
        OnnxTrajModel(
            checkpoint_path=str(FIXTURES / "dummy_decoder.onnx"),
            device=torch.device("cpu"),
            camera_ids=["cam_front_center"],
        )


def test_rejects_graph_without_trajectory_head() -> None:
    with pytest.raises(ValueError, match="policy.trajectory"):
        OnnxTrajModel(
            checkpoint_path=str(FIXTURES / "dummy_no_trajectory.onnx"),
            device=torch.device("cpu"),
            camera_ids=["cam_front_left"],
        )


def test_rejects_non_native_frames(model: OnnxTrajModel) -> None:
    with pytest.raises(ValueError, match="native fisheye resolution"):
        model.predict(_input({}, image_hw=(540, 960)))
