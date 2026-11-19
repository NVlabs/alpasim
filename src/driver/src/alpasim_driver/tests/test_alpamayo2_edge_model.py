# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Edge adapter contracts exercised without gated weights or a GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from alpasim_driver.models.alpamayo2_edge_model import Alpamayo2EdgeModel
from alpasim_driver.models.base import (
    DriveCommand,
    ModelInputValidationError,
    PredictionInput,
)
from alpasim_driver.models.trajectory_selection import TrajectorySelectionStrategy
from alpasim_driver.schema import ModelConfig, TrajectorySelectionConfig
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Vec3


def _input():
    times = range(1_000_000, 3_000_001, 100_000)
    return PredictionInput(
        camera_images={
            camera: [
                (time, np.full((8, 12, 3), i, dtype=np.uint8))
                for i, time in enumerate(times)
            ]
            for camera in Alpamayo2EdgeModel.CAMERA_IDS
        },
        ego_pose_history=[
            PoseAtTime(
                timestamp_us=time,
                pose=Pose(vec=Vec3(x=time / 1e6), quat=Quat(w=1.0)),
            )
            for time in times
        ],
        navigation=SimpleNamespace(instruction_text="Turn right in 30m"),
        command=DriveCommand.RIGHT,
        speed=1.0,
        acceleration=0.0,
        inference_seed=43,
        previous_plan=None,
        route=None,
    )


def test_edge_samples_native_offsets_and_retains_actual_capture_times():
    model = object.__new__(Alpamayo2EdgeModel)
    prediction = _input()
    # Sensor captures can trail their intended trigger by a few microseconds.
    prediction.camera_images = {
        camera: [(time - 7, image) for time, image in frames]
        for camera, frames in prediction.camera_images.items()
    }
    prepared = model._prepare_sample(prediction, 3_000_000)
    sample = prepared["sample"]
    assert sample["image_frames"].shape == (22, 1, 3, 8, 12)
    assert sample["image_frames"].dtype == torch.uint8
    assert sample["image_frames"].is_contiguous()
    for camera_index in (0, 1, 2, 3, 5, 6):
        selected = sample["camera_indices"] == camera_index
        offsets = (
            model.FRONT_OFFSETS_US if camera_index in (1, 6) else model.OTHER_OFFSETS_US
        )
        assert sample["absolute_timestamps"][selected].tolist() == [
            3_000_000 + offset - 7 for offset in offsets
        ]
    assert sample["ego_history_xyz"].shape == (1, 1, 20, 3)
    torch.testing.assert_close(
        sample["ego_history_xyz"][0, 0, :, 0], torch.linspace(-1.9, 0, 20)
    )
    assert sample["nav_text"] == ["Turn right in 30m"]


@pytest.mark.parametrize(
    "problem", ["missing_camera", "short_history", "no_navigation", "future"]
)
def test_edge_rejects_incomplete_or_future_inputs(problem):
    prediction = _input()
    camera = Alpamayo2EdgeModel.CAMERA_IDS[0]
    if problem == "missing_camera":
        prediction.camera_images.pop(camera)
    elif problem == "short_history":
        prediction.ego_pose_history = prediction.ego_pose_history[-5:]
    elif problem == "no_navigation":
        prediction.navigation = None
    else:
        prediction.camera_images = {
            camera: [(time + 3_000_000, image) for time, image in frames]
            for camera, frames in prediction.camera_images.items()
        }
    with pytest.raises((ModelInputValidationError, ValueError)):
        object.__new__(Alpamayo2EdgeModel)._prepare_sample(prediction, 3_000_000)


def test_edge_never_samples_future_camera_frames():
    prediction = _input()
    for frames in prediction.camera_images.values():
        frames.append((3_100_000, np.full((8, 12, 3), 255, dtype=np.uint8)))
    sample = object.__new__(Alpamayo2EdgeModel)._prepare_sample(prediction, 3_000_000)[
        "sample"
    ]
    assert sample["absolute_timestamps"].max().item() == 3_000_000
    assert sample["image_frames"].max().item() == 20


def test_edge_refuses_to_pad_missing_old_camera_frames():
    prediction = _input()
    camera = "camera_front_wide_120fov"
    prediction.camera_images[camera] = prediction.camera_images[camera][1:]
    with pytest.raises(ModelInputValidationError, match="no frame"):
        object.__new__(Alpamayo2EdgeModel)._prepare_sample(prediction, 3_000_000)


def test_edge_accepts_different_rendered_camera_widths():
    prediction = _input()
    camera = "camera_rear_left_70fov"
    prediction.camera_images[camera] = [
        (time, image[:, :10]) for time, image in prediction.camera_images[camera]
    ]
    sample = object.__new__(Alpamayo2EdgeModel)._prepare_sample(prediction, 3_000_000)[
        "sample"
    ]
    assert sample["image_frames"].shape == (22, 1, 3, 8, 12)
    assert sample["image_frames"].dtype == torch.uint8
    assert sample["image_frames"][
        sample["camera_indices"] == 3, 0, 0, 0, 0
    ].tolist() == [10, 19, 20]


def test_edge_prediction_preserves_planning_origin_and_body_rotations():
    model = object.__new__(Alpamayo2EdgeModel)
    positions = torch.zeros(1, 1, 1, 64, 3)
    positions[0, 0, 0, :, 0] = torch.arange(1, 65) / 10
    angle = 0.1
    body_rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    rotations = body_rotation.expand(1, 1, 1, 64, 3, 3)
    calls = []

    def sample(inputs, **kwargs):
        calls.append((inputs, kwargs))
        return positions, rotations, None, {"cot": np.array([[["Drive safely"]]])}

    model._helper = SimpleNamespace(
        prepare_model_inputs=lambda data, config, tokenizer: data,
        to_device=lambda data, device: data,
    )
    model._model = SimpleNamespace(
        config=None, tokenizer=None, sample_trajectories_from_data=sample
    )
    model._device = torch.device("cpu")
    model._deterministic = True
    model._num_samples = 1
    model._strategy = TrajectorySelectionStrategy.ALWAYS_FIRST
    model._selection = TrajectorySelectionConfig()
    prediction = model.predict(_input())
    assert calls[0][0]["image_frames"].shape[0] == 22
    assert calls[0][1]["num_traj_samples"] == 1
    assert prediction.model_t0_us == 3_000_000
    assert prediction.waypoint_timestamps_us[0] == 3_100_000
    assert prediction.waypoint_timestamps_us[-1] == 9_400_000
    np.testing.assert_allclose(prediction.candidate_rotations[0, 0], body_rotation)
    np.testing.assert_allclose(prediction.selected_plan.positions[0], [3.1, 0, 0])
    assert prediction.reasoning_text == "Drive safely"


def test_edge_config_validation_does_not_load_a_checkpoint():
    config = ModelConfig(
        model_type="alpamayo2_edge", checkpoint_path="unused", device="cpu"
    )
    with pytest.raises(ValueError, match="context_length"):
        Alpamayo2EdgeModel.from_config(
            config, torch.device("cpu"), list(Alpamayo2EdgeModel.CAMERA_IDS), 4, 10
        )


@pytest.mark.parametrize("offline", [False, True])
def test_edge_loader_respects_hugging_face_offline_mode(monkeypatch, offline):
    from alpamayo2_edge import checkpoint
    from huggingface_hub import constants as hf_constants

    load_kwargs = {}

    def load_model(model_id, **kwargs):
        load_kwargs.update(kwargs)
        return (
            SimpleNamespace(
                expert=SimpleNamespace(action_space=SimpleNamespace(dt=0.1))
            ),
            {},
        )

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", offline)
    monkeypatch.setattr(checkpoint, "load_model", load_model)
    config = ModelConfig(
        model_type="alpamayo2_edge", checkpoint_path="unused", device="cpu"
    )
    Alpamayo2EdgeModel.from_config(
        config, torch.device("cpu"), list(Alpamayo2EdgeModel.CAMERA_IDS), 21, 10
    )
    assert load_kwargs["local_files_only"] is offline


@pytest.mark.parametrize("dt, accepted", [(float(np.float32(0.1)), True), (0.2, False)])
def test_edge_accepts_a_float32_ten_hertz_checkpoint(monkeypatch, dt, accepted):
    from alpamayo2_edge import checkpoint

    def load_model(model_id, **kwargs):
        return (
            SimpleNamespace(
                expert=SimpleNamespace(action_space=SimpleNamespace(dt=dt))
            ),
            {},
        )

    monkeypatch.setattr(checkpoint, "load_model", load_model)
    config = ModelConfig(
        model_type="alpamayo2_edge", checkpoint_path="unused", device="cpu"
    )

    def build():
        return Alpamayo2EdgeModel.from_config(
            config, torch.device("cpu"), list(Alpamayo2EdgeModel.CAMERA_IDS), 21, 10
        )

    if accepted:
        build()
    else:
        with pytest.raises(ValueError, match="10 Hz"):
            build()


def test_edge_sample_matches_installed_sdk_contract():
    pytest.importorskip("alpamayo2_edge")
    from alpamayo2_edge.config import Alpamayo2EdgeConfig
    from alpamayo2_edge.helper import create_messages

    sample = object.__new__(Alpamayo2EdgeModel)._prepare_sample(_input(), 3_000_000)[
        "sample"
    ]
    content = create_messages(sample, Alpamayo2EdgeConfig())[1]["content"]
    assert sum(entry["type"] == "image" for entry in content) == 22
    text = "".join(entry["text"] for entry in content if entry["type"] == "text")
    assert text.count("<|traj_history|>") == 57
    assert "Turn right in 30m" in text
