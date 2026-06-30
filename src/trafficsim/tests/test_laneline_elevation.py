# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import pytest
import torch
from alpasim_trafficsim.grpc.catk_predictor import CATKTrafficPredictor
from alpasim_trafficsim.grpc.config import CatkConfig
from alpasim_trafficsim.grpc.pipeline.laneline_elevation import (
    agent_center_z_from_nearest_lanelines,
)
from alpasim_trafficsim.grpc.service_structures import SessionState


def _predictor(
    *,
    predict_static: bool = True,
) -> CATKTrafficPredictor:
    cfg = CatkConfig(
        predict_static=predict_static,
        min_valid_history_steps=0,
    )
    predictor = CATKTrafficPredictor.__new__(CATKTrafficPredictor)
    predictor.cfg = cfg
    predictor.predict_static = cfg.predict_static
    predictor.history_window_steps = cfg.loader.num_history_steps
    predictor.min_valid_history_steps = cfg.min_valid_history_steps
    predictor.model = None
    predictor._token_stride = 5
    return predictor


def _sloped_map() -> dict:
    return {
        "lanelines": {
            "polylines": torch.tensor(
                [
                    [[0.0, 0.0, 10.0], [10.0, 0.0, 20.0]],
                    [[0.0, 10.0, 100.0], [10.0, 10.0, 100.0]],
                ],
                dtype=torch.float32,
            ),
            "label": torch.tensor([0, 0], dtype=torch.long),
        }
    }


def test_agent_center_z_from_nearest_lanelines_interpolates_map_height() -> None:
    agent_xy = torch.tensor([[5.0, 1.0], [2.0, 9.0], [999.0, 999.0]])
    agent_lwh = torch.tensor(
        [
            [4.5, 2.0, 2.0],
            [4.5, 2.0, 4.0],
            [4.5, 2.0, 6.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor([True, True, False])
    fallback_z = torch.tensor([-1.0, -2.0, -3.0])

    z = agent_center_z_from_nearest_lanelines(
        _sloped_map(),
        agent_xy,
        agent_lwh=agent_lwh,
        valid_mask=valid_mask,
        fallback_z=fallback_z,
    )

    torch.testing.assert_close(z, torch.tensor([16.0, 102.0, -3.0]))


def test_catk_apply_predictions_sets_agent_z_from_nearest_laneline() -> None:
    env_data = {
        "env": {"agent_object_ids": ["agent-1"]},
        "map": _sloped_map(),
        "ego": {
            "xyz": torch.zeros((1, 3), dtype=torch.float32),
            "heading": torch.zeros((1,), dtype=torch.float32),
        },
        "agents": {
            "xyz": torch.zeros((1, 2, 3), dtype=torch.float32),
            "heading": torch.zeros((1, 2), dtype=torch.float32),
            "valid_mask": torch.tensor([[True, False]]),
            "lwh": torch.tensor([[4.5, 2.0, 2.0]], dtype=torch.float32),
            "track_ids": torch.tensor([1], dtype=torch.long),
            "class_ids": torch.tensor([1], dtype=torch.long),
            "num_obstacles": 1,
        },
    }
    session_state = SessionState(
        session_uuid="session-1",
        scene_id="clipgt-test-scene",
        current_ts_us=0,
        closed_loop_trajectories={},
        env_data=env_data,
        handover_time_us=1_000_000,
    )
    actions = {
        "agent_future_xyz": torch.tensor([[[4.0, 1.0, 0.0]]], dtype=torch.float32),
        "agent_future_heading": torch.tensor([[0.0]], dtype=torch.float32),
        "agent_future_valid_mask": torch.tensor([[True]]),
    }

    _predictor().apply_predictions_to_env(
        session_state,
        future_step_indices=[1],
        actions=actions,
    )

    torch.testing.assert_close(
        env_data["agents"]["xyz"][0, 1],
        torch.tensor([4.0, 1.0, 15.0]),
    )


def _static_session_state() -> tuple[SessionState, dict]:
    env_data = {
        "env": {
            "agent_object_ids": ["static-1"],
            "agent_is_static": [True],
        },
        "map": _sloped_map(),
        "ego": {
            "xyz": torch.zeros((1, 3), dtype=torch.float32),
            "heading": torch.zeros((1,), dtype=torch.float32),
        },
        "agents": {
            "xyz": torch.zeros((1, 2, 3), dtype=torch.float32),
            "heading": torch.zeros((1, 2), dtype=torch.float32),
            "valid_mask": torch.tensor([[True, False]]),
            "lwh": torch.tensor([[4.5, 2.0, 2.0]], dtype=torch.float32),
            "track_ids": torch.tensor([1], dtype=torch.long),
            "class_ids": torch.tensor([1], dtype=torch.long),
            "num_obstacles": 1,
        },
    }
    env_data["agents"]["xyz"][0, 0] = torch.tensor([4.0, 1.0, 15.0])
    env_data["agents"]["heading"][0, 0] = 0.25
    session_state = SessionState(
        session_uuid="session-1",
        scene_id="clipgt-test-scene",
        current_ts_us=0,
        closed_loop_trajectories={},
        env_data=env_data,
        handover_time_us=1_000_000,
    )
    return session_state, env_data


def test_catk_apply_predictions_freezes_static_agent_when_predict_static_false() -> (
    None
):
    session_state, env_data = _static_session_state()
    actions = {
        "agent_future_xyz": torch.tensor([[[100.0, 0.0, 0.0]]], dtype=torch.float32),
        "agent_future_heading": torch.tensor([[0.0]], dtype=torch.float32),
        "agent_future_valid_mask": torch.tensor([[True]]),
    }

    _predictor(predict_static=False).apply_predictions_to_env(
        session_state,
        future_step_indices=[1],
        actions=actions,
    )

    torch.testing.assert_close(
        env_data["agents"]["xyz"][0, 1],
        torch.tensor([4.0, 1.0, 15.0]),
    )
    assert env_data["agents"]["heading"][0, 1].item() == pytest.approx(0.25)
    assert bool(env_data["agents"]["valid_mask"][0, 1].item())


def test_catk_apply_predictions_moves_static_agent_when_predict_static_true() -> None:
    session_state, env_data = _static_session_state()
    actions = {
        "agent_future_xyz": torch.tensor([[[4.0, 1.0, 0.0]]], dtype=torch.float32),
        "agent_future_heading": torch.tensor([[0.0]], dtype=torch.float32),
        "agent_future_valid_mask": torch.tensor([[True]]),
    }

    _predictor(predict_static=True).apply_predictions_to_env(
        session_state,
        future_step_indices=[1],
        actions=actions,
    )

    torch.testing.assert_close(
        env_data["agents"]["xyz"][0, 1],
        torch.tensor([4.0, 1.0, 15.0]),
    )
    assert env_data["agents"]["heading"][0, 1].item() == pytest.approx(0.0)
