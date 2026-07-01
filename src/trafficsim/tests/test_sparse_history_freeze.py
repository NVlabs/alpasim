# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for sparse-history freezing in CATKTrafficPredictor."""

from __future__ import annotations

from typing import Any

import torch
from alpasim_trafficsim.grpc.catk_predictor import CATKTrafficPredictor
from alpasim_trafficsim.grpc.config import CatkConfig


def _predictor(min_valid_history_steps: int | None) -> CATKTrafficPredictor:
    cfg = CatkConfig(predict_static=True)
    if min_valid_history_steps is not None:
        cfg.min_valid_history_steps = min_valid_history_steps
    predictor = CATKTrafficPredictor.__new__(CATKTrafficPredictor)
    predictor.cfg = cfg
    predictor.predict_static = cfg.predict_static
    predictor.history_window_steps = cfg.loader.num_history_steps
    predictor.min_valid_history_steps = cfg.min_valid_history_steps
    predictor.model = None
    predictor._token_stride = 5
    return predictor


def _env_with_history(valid_history: list[bool]) -> dict:
    """Single agent whose trailing history validity is ``valid_history``.

    The current step (prev_step_idx) is the last entry. Position/heading at the
    current step are distinctive so we can detect a freeze (copy-forward).
    """
    steps = len(valid_history)
    xyz = torch.zeros((1, steps, 3), dtype=torch.float32)
    heading = torch.zeros((1, steps), dtype=torch.float32)
    valid = torch.tensor([valid_history], dtype=torch.bool)
    xyz[0, -1] = torch.tensor([7.0, 8.0, 9.0])
    heading[0, -1] = 0.5
    return {
        "env": {"agent_is_static": [False]},
        "agents": {
            "xyz": xyz,
            "heading": heading,
            "valid_mask": valid,
            "num_obstacles": 1,
        },
    }


def _model_pred(num_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_xyz = torch.zeros((1, num_steps, 3), dtype=torch.float32)
    pred_xyz[0, :, 0] = torch.arange(1, num_steps + 1, dtype=torch.float32) * 100.0
    pred_heading = torch.full((1, num_steps), 1.234, dtype=torch.float32)
    pred_valid = torch.ones((1, num_steps), dtype=torch.bool)
    return pred_xyz, pred_heading, pred_valid


def test_default_threshold_matches_packaged_config() -> None:
    predictor = _predictor(min_valid_history_steps=None)
    assert predictor.min_valid_history_steps == 5


def test_single_valid_step_agent_frozen_at_default_threshold() -> None:
    predictor = _predictor(min_valid_history_steps=None)
    env = _env_with_history([False] * 15 + [True])
    pred = _model_pred(5)
    out_xyz, out_heading, out_valid = predictor._postprocess_predictions(
        env,
        future_step_indices=[16, 17, 18, 19, 20],
        pred_xyz=pred[0],
        pred_heading=pred[1],
        pred_valid=pred[2],
    )
    for step in range(5):
        torch.testing.assert_close(out_xyz[0, step], torch.tensor([7.0, 8.0, 9.0]))
        assert out_heading[0, step].item() == 0.5
        assert bool(out_valid[0, step].item())


def test_recently_spawned_agent_frozen_with_default() -> None:
    predictor = _predictor(min_valid_history_steps=None)
    env = _env_with_history([False] * 14 + [True, True])
    pred = _model_pred(5)
    out_xyz, out_heading, out_valid = predictor._postprocess_predictions(
        env,
        future_step_indices=[16, 17, 18, 19, 20],
        pred_xyz=pred[0],
        pred_heading=pred[1],
        pred_valid=pred[2],
    )
    for step in range(5):
        torch.testing.assert_close(out_xyz[0, step], torch.tensor([7.0, 8.0, 9.0]))
        assert out_heading[0, step].item() == 0.5
        assert bool(out_valid[0, step].item())


def test_explicit_threshold_one_keeps_single_valid_step_agent_predictable() -> None:
    predictor = _predictor(min_valid_history_steps=1)
    env = _env_with_history([False] * 15 + [True])
    pred = _model_pred(5)
    out_xyz, out_heading, out_valid = predictor._postprocess_predictions(
        env,
        future_step_indices=[16, 17, 18, 19, 20],
        pred_xyz=pred[0],
        pred_heading=pred[1],
        pred_valid=pred[2],
    )
    torch.testing.assert_close(out_xyz[0, :, 0], pred[0][0, :, 0])
    torch.testing.assert_close(out_heading[0], pred[1][0])
    assert out_valid[0].all()


def test_threshold_zero_never_freezes_for_sparse_history() -> None:
    predictor = _predictor(min_valid_history_steps=0)
    env = _env_with_history([False] * 15 + [True])
    pred = _model_pred(5)
    out_xyz, _, _ = predictor._postprocess_predictions(
        env,
        future_step_indices=[16, 17, 18, 19, 20],
        pred_xyz=pred[0],
        pred_heading=pred[1],
        pred_valid=pred[2],
    )
    torch.testing.assert_close(out_xyz[0, :, 0], pred[0][0, :, 0])


class _EmptyMapModel:
    model_predict_step_num = 0

    def create_model_input(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        del args, kwargs
        return None

    def inference(self, input_data: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("inference should not run when map input is empty")


def _env_for_inference() -> dict[str, Any]:
    return {
        "env": {"curr_t": 15, "agent_is_static": [False]},
        "ego": {
            "xyz": torch.zeros((16, 3), dtype=torch.float32),
            "heading": torch.zeros((16,), dtype=torch.float32),
        },
        "agents": {
            "xyz": torch.zeros((1, 16, 3), dtype=torch.float32),
            "heading": torch.zeros((1, 16), dtype=torch.float32),
            "valid_mask": torch.ones((1, 16), dtype=torch.bool),
            "num_obstacles": 1,
        },
        "map": {},
    }


def test_empty_filtered_map_returns_none_when_prediction_unavailable() -> None:
    predictor = _predictor(min_valid_history_steps=None)
    predictor.model = _EmptyMapModel()

    assert predictor.run_inference(_env_for_inference(), predict_steps=1) is None


def test_model_input_value_error_still_propagates() -> None:
    class BrokenModel(_EmptyMapModel):
        def create_model_input(
            self, *args: Any, **kwargs: Any
        ) -> dict[str, Any] | None:
            del args, kwargs
            raise ValueError("different CATK input failure")

    predictor = _predictor(min_valid_history_steps=None)
    predictor.model = BrokenModel()

    try:
        predictor.run_inference(_env_for_inference(), predict_steps=1)
    except ValueError as exc:
        assert str(exc) == "different CATK input failure"
    else:
        raise AssertionError("expected model-input ValueError to propagate")
