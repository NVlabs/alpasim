# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo 2 driver-contract tests without loading a checkpoint."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

try:
    import alpamayo2_super
except ModuleNotFoundError:
    alpamayo2_super = ModuleType("alpamayo2_super")
    helper = ModuleType("alpamayo2_super.helper")
    helper.prepare_model_inputs = lambda data, _config, _tokenizer: data
    helper.to_device = lambda data, _device: data
    models = ModuleType("alpamayo2_super.models")
    model_module = ModuleType("alpamayo2_super.models.alpamayo2_super")

    class Alpamayo2Super:
        """Import-time placeholder for the external package."""

    model_module.Alpamayo2Super = Alpamayo2Super
    alpamayo2_super.helper = helper
    alpamayo2_super.models = models
    models.alpamayo2_super = model_module
    sys.modules.update(
        {
            "alpamayo2_super": alpamayo2_super,
            "alpamayo2_super.helper": helper,
            "alpamayo2_super.models": models,
            "alpamayo2_super.models.alpamayo2_super": model_module,
        }
    )

from alpasim_driver.models.alpamayo2_model import Alpamayo2Model
from alpasim_driver.models.base import (
    DriveCommand,
    ModelInputValidationError,
    PredictionInput,
)
from alpasim_driver.models.trajectory_selection import (
    TrajectorySelectionStrategy,
    plan_in_local_frame,
)
from alpasim_driver.schema import ModelConfig, TrajectorySelectionConfig
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Vec3
from alpasim_utils.geometry import Pose as GeometryPose

CAMERA_ID = "camera_front_wide_120fov"
LATEST_TIMESTAMP_US = 2_000_000
NUM_WAYPOINTS = 20


class _StubInferenceModel:
    """Records arguments and returns deterministic sampled candidates."""

    config = SimpleNamespace()
    tokenizer = SimpleNamespace()

    def __init__(self, candidates: np.ndarray) -> None:
        self._candidates = candidates
        self.sample_kwargs: dict | None = None
        self.sample_kwargs_list: list[dict] = []
        self._next_candidate_index = 0

    def sample_trajectories_from_data(self, **kwargs: object) -> tuple:
        self.sample_kwargs = kwargs
        self.sample_kwargs_list.append(kwargs)
        sample_size = int(kwargs["num_traj_samples"])
        start = self._next_candidate_index
        end = start + sample_size
        self._next_candidate_index = end
        positions = torch.from_numpy(self._candidates[start:end]).float()[None, None]
        rotations = torch.eye(3).expand(*positions.shape[:-1], 3, 3)
        cot = np.array(
            [
                [
                    [
                        f"reasoning {candidate_index}"
                        for candidate_index in range(start, end)
                    ]
                ]
            ]
        )
        return positions, rotations, torch.zeros_like(positions[..., 0]), {"cot": cot}


def _poses() -> list[PoseAtTime]:
    return [
        PoseAtTime(
            timestamp_us=timestamp_us,
            pose=Pose(vec=Vec3(), quat=Quat(w=1.0)),
        )
        for timestamp_us in range(0, LATEST_TIMESTAMP_US + 1, 100_000)
    ]


def _moving_poses(latest_timestamp_us: int) -> list[PoseAtTime]:
    """Straight-line poses with a fixed 90-degree yaw through planning t0."""
    half_sqrt_two = float(np.sqrt(0.5))
    return [
        PoseAtTime(
            timestamp_us=timestamp_us,
            pose=Pose(
                vec=Vec3(x=timestamp_us / 1e6),
                quat=Quat(w=half_sqrt_two, z=half_sqrt_two),
            ),
        )
        for timestamp_us in range(0, latest_timestamp_us + 1, 100_000)
    ]


def _prediction_input(previous_plan: object | None) -> PredictionInput:
    return PredictionInput(
        camera_images={
            CAMERA_ID: [(LATEST_TIMESTAMP_US, np.zeros((8, 8, 3), dtype=np.uint8))]
        },
        command=DriveCommand.STRAIGHT,
        speed=0.0,
        acceleration=0.0,
        ego_pose_history=_poses(),
        inference_seed=0,
        previous_plan=previous_plan,
        route=None,
    )


def _candidates() -> np.ndarray:
    forward = 0.1 * np.arange(1, NUM_WAYPOINTS + 1)
    return np.stack(
        [
            np.column_stack(
                (forward, np.full(NUM_WAYPOINTS, 2.0), np.zeros(NUM_WAYPOINTS))
            ),
            np.column_stack(
                (forward, np.zeros(NUM_WAYPOINTS), np.zeros(NUM_WAYPOINTS))
            ),
        ]
    )


def _model(candidates: np.ndarray) -> Alpamayo2Model:
    model = object.__new__(Alpamayo2Model)
    model._model = _StubInferenceModel(candidates)
    model._device = torch.device("cpu")
    model._camera_ids = [CAMERA_ID]
    model._context_length = 1
    model._num_traj_samples = len(candidates)
    model._trajectory_candidate_microbatch_size = None
    model._top_p = 0.98
    model._temperature = 0.6
    model._diffusion_steps = 10
    model._force_determinism = False
    model._selection_strategy = TrajectorySelectionStrategy.CLOSEST_LATERAL
    model._max_num_distance_points = 64
    model._skip_first_n_distance_points = 0
    return model


def test_prediction_selects_the_candidate_matching_the_previous_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _candidates()
    model = _model(candidates)
    monkeypatch.setattr(
        "alpasim_driver.models.alpamayo2_model.helper.prepare_model_inputs",
        lambda data, _config, _tokenizer: data,
    )
    monkeypatch.setattr(
        "alpasim_driver.models.alpamayo2_model.helper.to_device",
        lambda data, _device: data,
    )
    rotations = np.tile(np.eye(3), (NUM_WAYPOINTS, 1, 1))
    previous_plan = plan_in_local_frame(
        candidates[1],
        rotations,
        1_500_000 + np.arange(1, NUM_WAYPOINTS + 1, dtype=np.uint64) * 100_000,
        GeometryPose(np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0])),
    )

    prediction = model.predict(_prediction_input(previous_plan))

    np.testing.assert_allclose(prediction.selected_positions, candidates[1])
    np.testing.assert_allclose(prediction.candidate_positions, candidates)
    assert prediction.reasoning_text == "reasoning 1"
    assert prediction.selected_plan is not None
    np.testing.assert_array_equal(
        prediction.selected_plan.timestamps_us,
        LATEST_TIMESTAMP_US
        + np.arange(1, NUM_WAYPOINTS + 1, dtype=np.uint64) * 100_000,
    )
    assert model._model.sample_kwargs is not None
    assert model._model.sample_kwargs["top_p"] == 0.98
    assert model._model.sample_kwargs["temperature"] == 0.6
    assert model._model.sample_kwargs["num_traj_samples"] == 2
    assert model._model.sample_kwargs["diffusion_kwargs"] == {"inference_step": 10}
    assert model._model.sample_kwargs["return_extra"] is True


def test_stale_camera_keeps_images_but_uses_latest_ego_as_planning_t0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_timestamp_us = 2_000_000
    ego_timestamp_us = 2_200_000
    camera_image = np.full((8, 8, 3), 17, dtype=np.uint8)
    prediction_input = PredictionInput(
        camera_images={CAMERA_ID: [(camera_timestamp_us, camera_image)]},
        command=DriveCommand.STRAIGHT,
        speed=0.0,
        acceleration=0.0,
        ego_pose_history=_moving_poses(ego_timestamp_us),
        inference_seed=0,
        previous_plan=None,
        route=None,
    )
    model = _model(_candidates())
    captured_camera_images: dict[str, object] = {}

    def capture_images(camera_images: object) -> torch.Tensor:
        captured_camera_images["value"] = camera_images
        return torch.zeros((1, 1, 3, 8, 8), dtype=torch.uint8)

    monkeypatch.setattr(model, "_preprocess_images", capture_images)
    monkeypatch.setattr(
        "alpasim_driver.models.alpamayo2_model.helper.prepare_model_inputs",
        lambda data, _config, _tokenizer: data,
    )
    monkeypatch.setattr(
        "alpasim_driver.models.alpamayo2_model.helper.to_device",
        lambda data, _device: data,
    )

    prediction = model.predict(prediction_input)

    assert captured_camera_images["value"] is prediction_input.camera_images
    assert prediction.model_t0_us == ego_timestamp_us
    expected_pose = np.array(
        [
            [0.0, -1.0, 0.0, 2.2],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert prediction.pose_local_to_rig_t0 is not None
    np.testing.assert_allclose(
        prediction.pose_local_to_rig_t0.as_se3(), expected_pose, atol=1e-6
    )
    np.testing.assert_array_equal(
        prediction.waypoint_timestamps_us,
        ego_timestamp_us + np.arange(1, NUM_WAYPOINTS + 1, dtype=np.uint64) * 100_000,
    )


def test_prediction_rejects_camera_history_entirely_after_latest_ego() -> None:
    prediction_input = _prediction_input(previous_plan=None)
    prediction_input.camera_images = {
        CAMERA_ID: [
            (LATEST_TIMESTAMP_US + 100_000, np.zeros((8, 8, 3), dtype=np.uint8))
        ]
    }

    with pytest.raises(
        ModelInputValidationError,
        match=(
            r"needs at least one camera frame at or before " r"planning_t0_us=2000000"
        ),
    ):
        _model(_candidates()).predict(prediction_input)


def test_preprocess_images_keeps_frames_uint8() -> None:
    model = _model(_candidates())
    values = np.array([0, 1, 2, 127, 128, 254, 255, 17], dtype=np.uint8)
    image = values.reshape(2, 4, 1).repeat(3, axis=2)
    camera_images = {CAMERA_ID: [(LATEST_TIMESTAMP_US, image)]}

    frames = model._preprocess_images(camera_images)

    assert frames.dtype is torch.uint8
    torch.testing.assert_close(
        frames[0, 0], torch.from_numpy(image).permute(2, 0, 1), rtol=0, atol=0
    )


def test_prediction_microbatches_six_candidates_in_groups_of_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forward = 0.1 * np.arange(1, NUM_WAYPOINTS + 1)
    candidates = np.stack(
        [
            np.column_stack(
                (
                    forward,
                    np.full(NUM_WAYPOINTS, lateral_offset),
                    np.zeros(NUM_WAYPOINTS),
                )
            )
            for lateral_offset in range(6)
        ]
    )
    model = _model(candidates)
    model._trajectory_candidate_microbatch_size = 3
    monkeypatch.setattr(
        "alpasim_driver.models.alpamayo2_model.helper.prepare_model_inputs",
        lambda data, _config, _tokenizer: data,
    )
    monkeypatch.setattr(
        "alpasim_driver.models.alpamayo2_model.helper.to_device",
        lambda data, _device: data,
    )
    rotations = np.tile(np.eye(3), (NUM_WAYPOINTS, 1, 1))
    previous_plan = plan_in_local_frame(
        candidates[4],
        rotations,
        LATEST_TIMESTAMP_US
        + np.arange(1, NUM_WAYPOINTS + 1, dtype=np.uint64) * 100_000,
        GeometryPose(np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0])),
    )

    prediction = model.predict(_prediction_input(previous_plan))

    np.testing.assert_allclose(prediction.selected_positions, candidates[4])
    assert prediction.reasoning_text == "reasoning 4"
    assert [
        kwargs["num_traj_samples"] for kwargs in model._model.sample_kwargs_list
    ] == [3, 3]


def test_from_config_passes_trajectory_selection_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_kwargs: dict[str, object] = {}

    def fake_init(self: Alpamayo2Model, **kwargs: object) -> None:
        init_kwargs.update(kwargs)

    monkeypatch.setattr(Alpamayo2Model, "__init__", fake_init)
    config = ModelConfig(
        model_type="alpamayo2",
        checkpoint_path="checkpoint",
        device="cuda",
        num_trajectory_samples=3,
        trajectory_candidate_microbatch_size=2,
        trajectory_selection=TrajectorySelectionConfig(
            strategy="CLOSEST_3D",
            max_num_distance_points=12,
            skip_first_n_distance_points=2,
        ),
    )

    Alpamayo2Model.from_config(
        config,
        torch.device("cuda"),
        [CAMERA_ID],
        context_length=None,
        output_frequency_hz=10,
    )

    assert init_kwargs["num_traj_samples"] == 3
    assert init_kwargs["trajectory_candidate_microbatch_size"] == 2
    assert init_kwargs["selection_strategy"] is TrajectorySelectionStrategy.CLOSEST_3D
    assert init_kwargs["max_num_distance_points"] == 12
    assert init_kwargs["skip_first_n_distance_points"] == 2


def test_from_config_rejects_unsupported_navigation_cfg() -> None:
    config = ModelConfig(
        model_type="alpamayo2",
        checkpoint_path="checkpoint",
        device="cuda",
        cfg_guidance_weight=1.5,
    )

    with pytest.raises(ValueError, match="two-GPU"):
        Alpamayo2Model.from_config(
            config,
            torch.device("cuda"),
            [CAMERA_ID],
            context_length=None,
            output_frequency_hz=10,
        )


def test_checkpoint_waypoint_frequency_must_match_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoadedModel:
        expert = SimpleNamespace(
            action_space=SimpleNamespace(
                dt=0.2,
                get_action_space_dims=lambda: (NUM_WAYPOINTS, 2),
            )
        )

        def eval(self) -> None:
            pass

    monkeypatch.setattr(
        "alpasim_driver.models.alpamayo2_model.Alpamayo2Super",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: LoadedModel()),
    )

    with pytest.raises(ValueError, match="checkpoint predicts a waypoint every"):
        Alpamayo2Model(
            checkpoint_path="checkpoint",
            device=torch.device("cpu"),
            camera_ids=[CAMERA_ID],
        )
