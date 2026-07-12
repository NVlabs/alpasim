# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ..main import DriveJob, EgoDriverService
from ..models.alpamayo1_model import Alpamayo1Model
from ..models.alpamayo_base import AlpamayoBaseModel
from ..models.base import DriveCommand, ModelPrediction, PredictionInput


class _CapturingModel:
    def __init__(self) -> None:
        self.inputs: list[PredictionInput] = []

    def predict_batch(self, inputs: list[PredictionInput]) -> list[ModelPrediction]:
        self.inputs.extend(inputs)
        return []


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
    session = SimpleNamespace(seed=123, inference_count=0, poses=[])

    service._run_batch([_drive_job(session)])
    service._run_batch([_drive_job(session)])

    assert [model_input.inference_seed for model_input in service._model.inputs] == [
        123,
        124,
    ]
    assert session.inference_count == 2


def test_alpamayo1_force_determinism_reseeds_each_prediction() -> None:
    model = Alpamayo1Model.__new__(Alpamayo1Model)
    model._force_determinism = True
    prediction_input = PredictionInput(
        camera_images={},
        command=DriveCommand.STRAIGHT,
        speed=0.0,
        acceleration=0.0,
        ego_pose_history=[],
        inference_seed=123,
    )

    def random_prediction(
        _model: AlpamayoBaseModel,
        _prediction_input: PredictionInput,
    ) -> ModelPrediction:
        return ModelPrediction(
            trajectory_xy=torch.rand((2, 2)).numpy(),
            headings=np.zeros(2),
        )

    with patch.object(AlpamayoBaseModel, "predict", random_prediction):
        first = model.predict(prediction_input)
        torch.manual_seed(999)
        second = model.predict(prediction_input)

    np.testing.assert_array_equal(first.trajectory_xy, second.trajectory_xy)
