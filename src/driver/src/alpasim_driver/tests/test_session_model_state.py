# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from types import SimpleNamespace

import numpy as np

from ..main import DriveJob, EgoDriverService, Session
from ..models.base import DriveCommand, ModelPrediction, PredictionInput


class _CountingModel:
    """Counts its own inferences in ``session_state``, like a stateful model."""

    def predict_batch(self, inputs: list[PredictionInput]) -> list[ModelPrediction]:
        for inp in inputs:
            inp.session_state["calls"] = inp.session_state.get("calls", 0) + 1
        return [
            ModelPrediction.from_planar(np.zeros((0, 2)), np.zeros(0)) for _ in inputs
        ]


def _session(model_state: dict) -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        inference_count=0,
        poses=[],
        last_selected_plan=None,
        route=None,
        model_state=model_state,
        frames_trail_request_warned=False,
    )


def _drive_job(session: SimpleNamespace) -> DriveJob:
    return DriveJob(
        session_id="session",
        session=session,
        command=DriveCommand.STRAIGHT,
        pose=None,
        timestamp_us=0,
        result=None,  # type: ignore[arg-type]
    )


def test_model_state_persists_within_and_is_isolated_across_sessions() -> None:
    service = EgoDriverService.__new__(EgoDriverService)
    service._model = _CountingModel()
    service._get_speed_and_acceleration = lambda session: (0.0, 0.0)
    service._prepare_camera_images = lambda session: {}
    first = _session({})
    second = _session({})

    service._run_batch([_drive_job(first), _drive_job(second)])
    service._run_batch([_drive_job(first)])

    assert first.model_state == {"calls": 2}
    assert second.model_state == {"calls": 1}


def test_new_sessions_get_distinct_model_state() -> None:
    def create() -> Session:
        return Session(uuid="u", seed=0, debug_scene_id="s", frame_caches={})

    assert create().model_state is not create().model_state
