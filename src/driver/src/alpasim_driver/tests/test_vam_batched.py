# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""TDD tests for batched tokenization and predict_batch in VAMModel.

These tests define the *desired* behaviour after the VAM batching rework:
  - LRU token cache removed
  - Batch-tokenize all frames in a single GPU call
  - predict_batch overridden for true batched inference
  - predict() delegates to predict_batch()

Many of these tests are expected to **fail** against the current
implementation — that is intentional (red phase of TDD).
"""

from __future__ import annotations

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
from alpasim_driver.models.vam_model import VAMModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTEXT_LENGTH = 8
CAMERA_ID = "front_center"
IMG_H, IMG_W = 900, 1600  # NeuroNCAP expected dimensions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_image(seed: int = 0) -> np.ndarray:
    """Generate a deterministic random image of the expected size."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)


def _make_prediction_input(
    n_frames: int = CONTEXT_LENGTH,
    base_ts: int = 1_000_000,
    dt: int = 100_000,
    seed: int = 0,
    command: DriveCommand = DriveCommand.STRAIGHT,
) -> PredictionInput:
    """Build a PredictionInput with deterministic dummy camera frames."""
    frames = [
        CameraFrame(timestamp_us=base_ts + i * dt, image=_make_image(seed + i))
        for i in range(n_frames)
    ]
    return PredictionInput(
        camera_images={CAMERA_ID: frames},
        command=command,
        speed=10.0,
        acceleration=0.0,
        ego_pose_history=[],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def vam_model():
    """Construct a VAMModel with a fully mocked tokenizer and VAM network.

    Yields ``(model, mock_tokenizer, mock_vam)`` so individual tests can
    inspect call counts and arguments on the mocked components.
    """

    def _tokenizer_fn(input_tensor: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, 8, 16) long token indices."""
        n = input_tensor.shape[0]
        return torch.zeros(n, 8, 16, dtype=torch.long)

    def _vam_fn(
        visual_tokens: torch.Tensor,
        command_tensor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """(B, T, ...) tokens -> (B, 1, 6, 2) trajectory."""
        b = visual_tokens.shape[0]
        return torch.randn(b, 1, 6, 2)

    with (
        mock.patch(
            "alpasim_driver.models.vam_model.load_inference_VAM",
        ) as mock_load_vam,
        mock.patch("torch.jit.load") as mock_jit_load,
        mock.patch(
            "alpasim_driver.models.vam_model.NeuroNCAPTransform",
        ) as mock_transform,
    ):
        # -- tokenizer mock --
        mock_tokenizer = mock.MagicMock(name="tokenizer")
        mock_tokenizer.side_effect = _tokenizer_fn
        mock_jit_load.return_value = mock_tokenizer

        # -- VAM forward mock --
        mock_vam = mock.MagicMock(name="vam")
        mock_vam.side_effect = _vam_fn
        mock_load_vam.return_value = mock_vam

        # -- preprocessing mock (returns CHW float tensor) --
        mock_preproc = mock_transform.return_value
        mock_preproc.side_effect = lambda img: torch.rand(3, 224, 224)

        model = VAMModel(
            checkpoint_path="/fake/checkpoint.pt",
            tokenizer_path="/fake/tokenizer.jit",
            device=torch.device("cpu"),
            camera_ids=[CAMERA_ID],
            context_length=CONTEXT_LENGTH,
        )

        yield model, mock_tokenizer, mock_vam


# ===================================================================
# TestPredictBatch
# ===================================================================
class TestPredictBatch:
    """Tests targeting the future batched predict_batch implementation."""

    def test_single_input_returns_single_result(self, vam_model):
        """predict_batch with 1 input returns 1 ModelPrediction with correct shapes."""
        model, _, _ = vam_model
        inp = _make_prediction_input()

        results = model.predict_batch([inp])

        assert len(results) == 1
        pred = results[0]
        assert isinstance(pred, ModelPrediction)
        assert pred.trajectory_xy.shape == (6, 2)
        assert pred.headings.shape == (6,)

    def test_batch_returns_correct_count(self, vam_model):
        """predict_batch with 4 inputs returns 4 ModelPredictions with correct shapes."""
        model, _, _ = vam_model
        inputs = [
            _make_prediction_input(base_ts=1_000_000 * (i + 1), seed=i * CONTEXT_LENGTH)
            for i in range(4)
        ]

        results = model.predict_batch(inputs)

        assert len(results) == 4
        for pred in results:
            assert isinstance(pred, ModelPrediction)
            assert pred.trajectory_xy.shape == (6, 2)
            assert pred.headings.shape == (6,)

    def test_tokenizer_called_once_with_full_batch(self, vam_model):
        """For 3 inputs * 8 frames = 24 frames, the tokenizer must be called
        exactly once with a tensor whose batch dimension equals 24.
        """
        model, mock_tokenizer, _ = vam_model
        inputs = [
            _make_prediction_input(base_ts=1_000_000 * (i + 1), seed=i * CONTEXT_LENGTH)
            for i in range(3)
        ]

        model.predict_batch(inputs)

        assert mock_tokenizer.call_count == 1, (
            f"Expected tokenizer to be called once (batched), "
            f"but was called {mock_tokenizer.call_count} times"
        )
        first_arg = mock_tokenizer.call_args[0][0]
        assert (
            first_arg.shape[0] == 24
        ), f"Expected tokenizer input batch dim == 24, got {first_arg.shape[0]}"

    def test_vam_called_once_with_batched_tokens(self, vam_model):
        """VAM forward must be called exactly once with batch dim B=3."""
        model, _, mock_vam = vam_model
        inputs = [
            _make_prediction_input(base_ts=1_000_000 * (i + 1), seed=i * CONTEXT_LENGTH)
            for i in range(3)
        ]

        model.predict_batch(inputs)

        assert mock_vam.call_count == 1, (
            f"Expected VAM forward to be called once, "
            f"but was called {mock_vam.call_count} times"
        )
        visual_tokens_arg = mock_vam.call_args[0][0]
        assert visual_tokens_arg.shape[0] == 3, (
            f"Expected visual_tokens batch dim == 3, "
            f"got {visual_tokens_arg.shape[0]}"
        )
        # T dimension should match context_length
        assert visual_tokens_arg.shape[1] == CONTEXT_LENGTH

    def test_predict_delegates_to_predict_batch(self, vam_model):
        """predict() should delegate to predict_batch([input])[0]."""
        model, _, _ = vam_model
        inp = _make_prediction_input()

        sentinel = ModelPrediction(
            trajectory_xy=np.zeros((6, 2)),
            headings=np.zeros(6),
        )
        with mock.patch.object(
            model, "predict_batch", return_value=[sentinel]
        ) as mock_pb:
            result = model.predict(inp)

        mock_pb.assert_called_once()
        # The single argument should be a list containing the input
        call_args = mock_pb.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert call_args[0] is inp
        assert result is sentinel

    def test_wrong_context_length_raises(self, vam_model):
        """ValueError when frame count != context_length."""
        model, _, _ = vam_model
        inp = _make_prediction_input(n_frames=4)  # wrong: 4 instead of 8

        with pytest.raises(ValueError, match=r"expects.*frames|context_length"):
            model.predict_batch([inp])


# ===================================================================
# TestNoCacheCrossContamination
# ===================================================================
class TestNoCacheCrossContamination:
    """Tests ensuring the LRU token cache is removed."""

    def test_same_timestamps_different_images_produce_different_tokenizer_input(
        self, vam_model
    ):
        """Two inputs sharing timestamps but with distinct pixel data must
        both be tokenized — the tokenizer should see all 16 frames in a
        single batched call (no timestamp-keyed caching).
        """
        model, mock_tokenizer, _ = vam_model

        inp_a = _make_prediction_input(base_ts=1_000_000, seed=0)
        inp_b = _make_prediction_input(base_ts=1_000_000, seed=100)

        model.predict_batch([inp_a, inp_b])

        assert mock_tokenizer.call_count == 1, (
            f"Expected tokenizer to be called once (batched), "
            f"but was called {mock_tokenizer.call_count} times"
        )
        first_arg = mock_tokenizer.call_args[0][0]
        assert first_arg.shape[0] == 16, (
            f"Expected tokenizer input batch dim == 16 (2 x 8 frames), "
            f"got {first_arg.shape[0]}"
        )

    def test_no_token_cache_attribute(self, vam_model):
        """After the rework, VAMModel should not carry a _token_cache."""
        model, _, _ = vam_model
        assert not hasattr(
            model, "_token_cache"
        ), "VAMModel still has _token_cache; it should be removed"
