# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for batched predict_batch in TransfuserModel.

Mirrors the VAM batched tests — verifies:
  - predict() delegates to predict_batch()
  - predict_batch runs a single forward pass for the whole batch
  - correct shapes and per-sample unpacking
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest import mock

import numpy as np
import pytest
import torch
from alpasim_driver.models.base import (
    CameraFrame,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.models.transfuser_model import TransfuserModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_IDS = ["cam_l0", "cam_f0", "cam_r0", "cam_r1"]
NUM_CAMERAS = 4
PER_CAM_H, PER_CAM_W = 270, 480
N_WAYPOINTS = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_image(h: int = PER_CAM_H, w: int = PER_CAM_W, seed: int = 0) -> np.ndarray:
    """Generate a deterministic random image."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _make_prediction_input(
    seed: int = 0,
    command: DriveCommand = DriveCommand.STRAIGHT,
    speed: float = 10.0,
    acceleration: float = 0.0,
) -> PredictionInput:
    """Build a PredictionInput with a single frame per camera."""
    camera_images = {
        cam_id: [CameraFrame(timestamp_us=1_000_000, image=_make_image(seed=seed + i))]
        for i, cam_id in enumerate(CAMERA_IDS)
    }
    return PredictionInput(
        camera_images=camera_images,
        command=command,
        speed=speed,
        acceleration=acceleration,
        ego_pose_history=[],
    )


# ---------------------------------------------------------------------------
# Stubs for the Transfuser model internals
# ---------------------------------------------------------------------------
@dataclass
class _FakeConfig:
    """Minimal config stub providing torch_float_type."""

    torch_float_type: torch.dtype = torch.float32


@dataclass
class _FakePrediction:
    """Mimics transfuser_impl.Prediction returned by Model.forward()."""

    pred_future_waypoints: torch.Tensor  # (B, N, 2)
    pred_headings: torch.Tensor | None  # (B, N) or None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def transfuser_model():
    """Construct a TransfuserModel with a fully mocked underlying model.

    Yields ``(model, mock_forward)`` so tests can inspect call counts and
    arguments on the mocked forward pass.
    """

    def _forward_fn(data: dict[str, torch.Tensor]) -> _FakePrediction:
        """(B, ...) inputs -> Prediction with random waypoints and headings."""
        b = data["rgb"].shape[0]
        return _FakePrediction(
            pred_future_waypoints=torch.randn(b, N_WAYPOINTS, 2),
            pred_headings=torch.randn(b, N_WAYPOINTS),
        )

    mock_model = mock.MagicMock(name="transfuser_model")
    mock_model.side_effect = _forward_fn
    mock_model.config = _FakeConfig()

    with mock.patch(
        "alpasim_driver.models.transfuser_model.load_tf",
        return_value=mock_model,
    ):
        model = TransfuserModel(
            checkpoint_path="/fake/checkpoint.pth",
            device=torch.device("cpu"),
            camera_ids=list(CAMERA_IDS),
        )

    yield model, mock_model


# ===================================================================
# TestPredictBatch
# ===================================================================
class TestPredictBatch:
    """Tests targeting the batched predict_batch implementation."""

    def test_single_input_returns_single_result(self, transfuser_model):
        """predict_batch with 1 input returns 1 ModelPrediction."""
        model, _ = transfuser_model
        inp = _make_prediction_input()

        results = model.predict_batch([inp])

        assert len(results) == 1
        pred = results[0]
        assert isinstance(pred, ModelPrediction)
        assert pred.trajectory_xy.shape == (N_WAYPOINTS, 2)
        assert pred.headings.shape == (N_WAYPOINTS,)

    def test_batch_returns_correct_count(self, transfuser_model):
        """predict_batch with 4 inputs returns 4 ModelPredictions."""
        model, _ = transfuser_model
        inputs = [_make_prediction_input(seed=i * NUM_CAMERAS) for i in range(4)]

        results = model.predict_batch(inputs)

        assert len(results) == 4
        for pred in results:
            assert isinstance(pred, ModelPrediction)
            assert pred.trajectory_xy.shape == (N_WAYPOINTS, 2)
            assert pred.headings.shape == (N_WAYPOINTS,)

    def test_empty_input_returns_empty(self, transfuser_model):
        """predict_batch with empty list returns empty list."""
        model, mock_fwd = transfuser_model

        results = model.predict_batch([])

        assert results == []
        mock_fwd.assert_not_called()

    def test_model_called_once_with_batched_tensors(self, transfuser_model):
        """Model forward must be called exactly once with batch dim B=3."""
        model, mock_fwd = transfuser_model
        inputs = [_make_prediction_input(seed=i * NUM_CAMERAS) for i in range(3)]

        model.predict_batch(inputs)

        assert mock_fwd.call_count == 1, (
            f"Expected model forward to be called once, "
            f"but was called {mock_fwd.call_count} times"
        )
        data_arg = mock_fwd.call_args[0][0]
        assert data_arg["rgb"].shape[0] == 3
        assert data_arg["command"].shape == (3, 4)
        assert data_arg["speed"].shape == (3,)
        assert data_arg["acceleration"].shape == (3,)

    def test_rgb_has_correct_concatenated_width(self, transfuser_model):
        """Batched RGB tensor should have width = NUM_CAMERAS * per-cam width."""
        model, mock_fwd = transfuser_model
        inp = _make_prediction_input()

        model.predict_batch([inp])

        data_arg = mock_fwd.call_args[0][0]
        # (B, 3, H, W) where W = NUM_CAMERAS * PER_CAM_W
        assert data_arg["rgb"].shape == (
            1,
            3,
            PER_CAM_H,
            NUM_CAMERAS * PER_CAM_W,
        )

    def test_predict_delegates_to_predict_batch(self, transfuser_model):
        """predict() should delegate to predict_batch([input])[0]."""
        model, _ = transfuser_model
        inp = _make_prediction_input()

        sentinel = ModelPrediction(
            trajectory_xy=np.zeros((N_WAYPOINTS, 2)),
            headings=np.zeros(N_WAYPOINTS),
        )
        with mock.patch.object(
            model, "predict_batch", return_value=[sentinel]
        ) as mock_pb:
            result = model.predict(inp)

        mock_pb.assert_called_once()
        call_args = mock_pb.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert call_args[0] is inp
        assert result is sentinel

    def test_y_axis_inverted_for_rig_frame(self, transfuser_model):
        """Waypoint Y values should be negated (CARLA Y+ right -> rig Y+ left)."""
        model, mock_fwd = transfuser_model

        # Return a known deterministic prediction
        known_waypoints = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        known_headings = torch.tensor([[0.5, 1.0]])  # (1, 2)

        mock_fwd.side_effect = None
        mock_fwd.return_value = _FakePrediction(
            pred_future_waypoints=known_waypoints,
            pred_headings=known_headings,
        )

        inp = _make_prediction_input()
        results = model.predict_batch([inp])

        np.testing.assert_array_almost_equal(
            results[0].trajectory_xy, np.array([[1.0, -2.0], [3.0, -4.0]])
        )
        np.testing.assert_array_almost_equal(
            results[0].headings, np.array([-0.5, -1.0])
        )

    def test_headings_computed_when_none(self, transfuser_model):
        """When pred_headings is None, headings are computed from trajectory."""
        model, mock_fwd = transfuser_model

        known_waypoints = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])  # (1, 2, 2)
        mock_fwd.side_effect = None
        mock_fwd.return_value = _FakePrediction(
            pred_future_waypoints=known_waypoints,
            pred_headings=None,
        )

        inp = _make_prediction_input()
        results = model.predict_batch([inp])

        # Y is inverted: waypoints become [[1, 0], [2, 0]]
        # Heading from (0,0)->(1,0) = atan2(0, 1) = 0
        # Heading from (1,0)->(2,0) = atan2(0, 1) = 0
        assert results[0].headings.shape == (2,)
        np.testing.assert_array_almost_equal(results[0].headings, [0.0, 0.0])

    def test_wrong_frame_count_raises(self, transfuser_model):
        """ValueError when a camera has != 1 frame."""
        model, _ = transfuser_model

        # Build input with 2 frames for first camera
        camera_images = {
            cam_id: [
                CameraFrame(timestamp_us=1_000_000, image=_make_image(seed=i)),
            ]
            for i, cam_id in enumerate(CAMERA_IDS)
        }
        # Add an extra frame to the first camera
        camera_images[CAMERA_IDS[0]].append(
            CameraFrame(timestamp_us=2_000_000, image=_make_image(seed=99))
        )

        inp = PredictionInput(
            camera_images=camera_images,
            command=DriveCommand.STRAIGHT,
            speed=10.0,
            acceleration=0.0,
            ego_pose_history=[],
        )

        with pytest.raises(ValueError, match=r"expects 1 frame"):
            model.predict_batch([inp])

    def test_different_commands_encoded_correctly(self, transfuser_model):
        """Each input's command should be independently one-hot encoded."""
        model, mock_fwd = transfuser_model

        inputs = [
            _make_prediction_input(seed=0, command=DriveCommand.LEFT),
            _make_prediction_input(seed=4, command=DriveCommand.STRAIGHT),
            _make_prediction_input(seed=8, command=DriveCommand.RIGHT),
        ]

        model.predict_batch(inputs)

        data_arg = mock_fwd.call_args[0][0]
        commands = data_arg["command"]  # (3, 4) one-hot
        # LEFT=0, STRAIGHT=1, RIGHT=2
        assert commands[0].argmax().item() == 0  # LEFT
        assert commands[1].argmax().item() == 1  # FORWARD/STRAIGHT
        assert commands[2].argmax().item() == 2  # RIGHT
