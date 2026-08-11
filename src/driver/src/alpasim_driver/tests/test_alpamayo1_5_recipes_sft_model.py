# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the native Alpamayo Recipes 1.5 SFT adapter."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import torch
from alpasim_driver.models import alpamayo1_5_recipes_sft_model as recipes_model
from alpasim_driver.models.alpamayo1_5_model import Alpamayo15Model


def test_load_recipes_model_uses_public_sft_api(monkeypatch) -> None:
    package = ModuleType("alpamayo1_5_sft")
    models_package = ModuleType("alpamayo1_5_sft.models")
    model_module = ModuleType("alpamayo1_5_sft.models.sft_alpamayo_r1")
    model_class = Mock()
    loaded_model = Mock()
    device_model = Mock()
    model_class.from_pretrained.return_value = loaded_model
    loaded_model.to.return_value = device_model
    device_model.eval.return_value = device_model
    model_module.TrainableAlpamayoR1 = model_class

    monkeypatch.setitem(sys.modules, "alpamayo1_5_sft", package)
    monkeypatch.setitem(sys.modules, "alpamayo1_5_sft.models", models_package)
    monkeypatch.setitem(
        sys.modules, "alpamayo1_5_sft.models.sft_alpamayo_r1", model_module
    )

    device = torch.device("cpu")
    result = recipes_model._load_recipes_model("/checkpoint", device)

    assert result is device_model
    model_class.from_pretrained.assert_called_once_with(
        "/checkpoint",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    loaded_model.to.assert_called_once_with(device)
    device_model.eval.assert_called_once_with()


def test_model_reuses_alpamayo15_driver_initialization(monkeypatch) -> None:
    model = Mock()
    model.tokenizer = object()
    model.action_space.get_action_space_dims.return_value = (64, 2)
    model.action_space.dt = 0.1
    processor = object()
    chat_template = object()
    configure_determinism = Mock()

    monkeypatch.setattr(recipes_model, "_load_recipes_model", Mock(return_value=model))
    monkeypatch.setattr(
        recipes_model,
        "_load_recipes_chat_template",
        Mock(return_value=chat_template),
    )
    monkeypatch.setattr(
        recipes_model.helper, "get_processor", Mock(return_value=processor)
    )
    monkeypatch.setattr(
        recipes_model, "configure_deterministic_runtime", configure_determinism
    )

    adapter = recipes_model.Alpamayo15RecipesSFTModel(
        checkpoint_path="/checkpoint",
        device=torch.device("cpu"),
        camera_ids=["camera_front_wide_120fov"],
        force_determinism=True,
    )

    configure_determinism.assert_called_once_with()
    assert isinstance(adapter, Alpamayo15Model)
    assert adapter._model is model
    assert adapter._processor is processor
    assert adapter._chat_template is chat_template
    assert adapter._force_determinism is True
    assert adapter._cfg_guidance_weight is None


def test_model_uses_recipes_future_only_prompt_with_alpamayo15_camera_order() -> None:
    adapter = object.__new__(recipes_model.Alpamayo15RecipesSFTModel)
    adapter._camera_ids = [
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
    ]
    adapter._model = SimpleNamespace(
        config=SimpleNamespace(
            tokens_per_history_traj=48,
            tokens_per_future_traj=128,
            include_camera_ids=True,
            include_frame_nums=True,
        )
    )
    adapter._chat_template = Mock()
    adapter._chat_template.build_conversation.return_value = ["message"]
    frames = torch.zeros(2, 4, 3, 2, 2)

    result = adapter._create_chat_message(frames, nav_text=None)

    assert result == ["message"]
    adapter._chat_template.build_conversation.assert_called_once()
    kwargs = adapter._chat_template.build_conversation.call_args.kwargs
    assert kwargs["data"] == {"image_frames": frames}
    assert kwargs["num_tokens_per_history_traj"] == 48
    assert kwargs["num_tokens_per_future_traj"] == 128
    assert kwargs["components_order"] == [
        "image",
        "traj_history",
        "prompt",
        "traj_future",
    ]
    assert kwargs["components_prompt"] == ["traj_future"]
    assert kwargs["generation_mode"] is True
    assert kwargs["include_camera_ids"] is True
    assert torch.equal(kwargs["camera_ids"], torch.tensor([0, 1]))
    assert kwargs["include_frame_nums"] is True
