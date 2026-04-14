# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import numpy as np
import pytest
import torch

from alpasim_driver.models.base import (
    CameraFrame,
    DriveCommand,
    PredictionInput,
)
from alpasim_driver.models.pdm_model import PDMModel
from alpasim_driver.schema import ModelConfig


CAMERA_ID = "camera_front_wide_120fov"


def _make_model() -> PDMModel:
    return PDMModel.from_config(
        ModelConfig(model_type="pdm", checkpoint_path="unused", device="cpu"),
        device=torch.device("cpu"),
        camera_ids=[CAMERA_ID],
        context_length=1,
        output_frequency_hz=10,
    )


def _make_prediction_input(
    *,
    planner_context: dict | None,
    command: DriveCommand = DriveCommand.STRAIGHT,
) -> PredictionInput:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    return PredictionInput(
        camera_images={CAMERA_ID: [CameraFrame(timestamp_us=1_000_000, image=image)]},
        command=command,
        speed=6.0,
        acceleration=0.0,
        ego_pose_history=[],
        planner_context=planner_context,
    )


def _full_planner_context() -> dict:
    return {
        "ego": {
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.0,
        },
        "route_waypoints_in_rig": [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 1.0, 0.0],
        ],
        "nearby_lanes": [
            {
                "id": "lane_main",
                "centerline_in_rig": [[0.0, 0.0], [10.0, 0.0], [25.0, 1.0]],
            },
            {
                "id": "lane_left",
                "centerline_in_rig": [[0.0, 2.0], [10.0, 2.0], [25.0, 3.0]],
            },
        ],
        "actors": [
            {
                "id": "lead",
                "position_in_rig": [14.0, 0.2],
                "yaw": 0.0,
            }
        ],
        "traffic_rules": {
            "wait_lines_in_rig": [
                {
                    "type": "Stop",
                    "points": [[6.0, -2.0], [6.0, 2.0]],
                }
            ],
            "crosswalks_in_rig": [
                [[18.0, -2.0], [20.0, -2.0], [20.0, 2.0], [18.0, 2.0]]
            ],
        },
    }


def test_pdm_predict_with_full_context_returns_non_empty_trajectory() -> None:
    model = _make_model()
    prediction = model.predict(_make_prediction_input(planner_context=_full_planner_context()))

    assert prediction.trajectory_xy.shape == (40, 2)
    assert prediction.headings.shape == (40,)
    assert prediction.debug_metadata is not None
    assert prediction.debug_metadata["planner_backend"] == "pdm_closed"
    assert prediction.debug_metadata["proposal_count"] >= 1
    assert prediction.debug_metadata["route_available"] is True
    assert prediction.debug_metadata["actor_count"] == 1
    assert prediction.debug_metadata["wait_line_count"] == 1
    assert prediction.debug_metadata["crosswalk_count"] == 1


@pytest.mark.parametrize(
    ("planner_context", "expected_fallback"),
    [
        ({}, "heuristic_centerline"),
        (
            {
                "ego": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                "actors": [],
                "traffic_rules": {},
            },
            "heuristic_centerline",
        ),
        (
            {
                "ego": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                "route_waypoints_in_rig": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "actors": [],
                "traffic_rules": {},
            },
            "missing_nearby_lanes",
        ),
    ],
)
def test_pdm_predict_degrades_gracefully_when_context_missing(
    planner_context: dict,
    expected_fallback: str,
) -> None:
    model = _make_model()
    prediction = model.predict(_make_prediction_input(planner_context=planner_context))

    assert prediction.trajectory_xy.shape == (40, 2)
    assert prediction.headings.shape == (40,)
    assert prediction.debug_metadata is not None
    assert prediction.debug_metadata["fallback_reason"] == expected_fallback


def test_pdm_predict_respects_rig_frame_heading_for_turns() -> None:
    model = _make_model()
    prediction = model.predict(
        _make_prediction_input(planner_context=None, command=DriveCommand.LEFT)
    )

    assert prediction.trajectory_xy[-1, 1] > 0.0
    assert np.max(prediction.headings) > 0.1
