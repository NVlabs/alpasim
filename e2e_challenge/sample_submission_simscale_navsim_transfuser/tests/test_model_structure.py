# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import timm
import torch
from navsim_transfuser_challenge.simscale_ltf import TransfuserConfig, TransfuserModel


@pytest.fixture(scope="module")
def default_model() -> tuple[TransfuserModel, list[dict[str, Any]]]:
    create_model = timm.create_model
    create_model_calls: list[dict[str, Any]] = []

    def offline_create_model(
        *args: Any,
        **kwargs: Any,
    ) -> torch.nn.Module:
        create_model_calls.append(kwargs.copy())
        assert kwargs.get("pretrained") is False
        assert "pretrained_cfg_overlay" not in kwargs
        factory: Callable[..., torch.nn.Module] = create_model
        return factory(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(timm, "create_model", offline_create_model)
    try:
        model = TransfuserModel(TransfuserConfig())
    finally:
        monkeypatch.undo()

    return model, create_model_calls


def test_default_model_matches_released_checkpoint_structure(
    default_model: tuple[TransfuserModel, list[dict[str, Any]]],
) -> None:
    model, create_model_calls = default_model
    state_dict = model.state_dict()

    assert len(create_model_calls) == 2
    assert model._trajectory_head._num_poses == 8
    assert model._trajectory_head._mlp[-1].out_features == 24
    assert len(state_dict) == 669
    assert state_dict["_backbone.lidar_latent"].shape == (1, 1, 256, 256)
    assert state_dict["_keyval_embedding.weight"].shape == (65, 256)
    assert state_dict["_trajectory_head._mlp.2.weight"].shape == (24, 1024)
    assert "_backbone.image_encoder.conv1.weight" in state_dict
    assert "_backbone.lidar_encoder.conv1.weight" in state_dict


def test_default_trajectory_head_output_shape(
    default_model: tuple[TransfuserModel, list[dict[str, Any]]],
) -> None:
    model, _ = default_model

    output = model._trajectory_head(torch.zeros(1, 1, 256))

    assert output["trajectory"].shape == (1, 8, 3)
