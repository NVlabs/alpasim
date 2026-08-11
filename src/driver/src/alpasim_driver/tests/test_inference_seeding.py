# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ..main import DriveJob, EgoDriverService
from ..models.alpamayo1_5_model import Alpamayo15Model
from ..models.alpamayo1_model import Alpamayo1Model
from ..models.alpamayo_base import AlpamayoBaseModel
from ..models.base import DriveCommand, ModelPrediction, PredictionInput


class _CapturingModel:
    def __init__(self) -> None:
        self.inputs: list[PredictionInput] = []

    def predict_batch(self, inputs: list[PredictionInput]) -> list[ModelPrediction]:
        self.inputs.extend(inputs)
        return [
            ModelPrediction.from_planar(np.zeros((0, 2)), np.zeros(0)) for _ in inputs
        ]


def _drive_job(session: SimpleNamespace) -> DriveJob:
    return DriveJob(
        session_id="session",
        session=session,
        command=DriveCommand.STRAIGHT,
        pose=None,
        timestamp_us=0,
        result=None,  # type: ignore[arg-type]
    )


def test_run_batch_assigns_consecutive_per_session_inference_seeds() -> None:
    service = EgoDriverService.__new__(EgoDriverService)
    service._model = _CapturingModel()
    service._get_speed_and_acceleration = lambda session: (0.0, 0.0)
    service._prepare_camera_images = lambda session: {}
    session = SimpleNamespace(
        seed=123,
        inference_count=0,
        poses=[],
        last_selected_plan=None,
        route=None,
        frames_trail_request_warned=False,
    )

    service._run_batch([_drive_job(session)])
    service._run_batch([_drive_job(session)])

    assert [model_input.inference_seed for model_input in service._model.inputs] == [
        123,
        124,
    ]
    assert session.inference_count == 2


@pytest.mark.parametrize("model_class", [Alpamayo1Model, Alpamayo15Model])
def test_alpamayo_force_determinism_reseeds_each_prediction(
    model_class: type[AlpamayoBaseModel],
) -> None:
    model = model_class.__new__(model_class)
    model._force_determinism = True
    prediction_input = PredictionInput(
        camera_images={},
        command=DriveCommand.STRAIGHT,
        speed=0.0,
        acceleration=0.0,
        ego_pose_history=[],
        inference_seed=123,
        previous_plan=None,
        route=None,
    )

    with patch.object(model, "_validate_cameras", side_effect=RuntimeError):
        with pytest.raises(RuntimeError):
            model.predict(prediction_input)
        first = torch.rand((2, 2)).numpy()
        torch.manual_seed(999)
        with pytest.raises(RuntimeError):
            model.predict(prediction_input)
        second = torch.rand((2, 2)).numpy()

    np.testing.assert_array_equal(first, second)
