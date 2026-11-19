# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for driver schema and plugin-based model discovery."""

from pathlib import Path

import pytest
from alpasim_driver.models.base import BaseTrajectoryModel
from alpasim_driver.schema import DriverConfig
from alpasim_plugins.plugins import models as model_registry
from omegaconf import OmegaConf

# The entry-point names that the driver package registers.
# transfuser is provided by the optional alpasim_transfuser plugin, not the core driver.
EXPECTED_MODELS = [
    "alpamayo1",
    "alpamayo1_5",
    "alpamayo1_5_recipes_sft",
    "alpamayo2",
    "manual",
    "vam",
]

RELEASE_NATIVE_GRAPH_PRESETS = [
    ("alpamayo1", 4),
    ("alpamayo1_5", 4),
    ("alpamayo2", 1),
]


@pytest.mark.parametrize(("preset", "max_graphs"), RELEASE_NATIVE_GRAPH_PRESETS)
def test_release_native_presets_enable_cuda_graphs_by_default(
    preset: str, max_graphs: int
) -> None:
    """Release-native Alpamayo presets default to bounded graph capture."""
    repo_root = Path(__file__).resolve().parents[5]
    config_path = repo_root / "src" / "wizard" / "configs" / "driver" / f"{preset}.yaml"
    raw_cfg = OmegaConf.load(config_path)
    del raw_cfg["defaults"]
    cfg = OmegaConf.merge(OmegaConf.structured(DriverConfig), raw_cfg)

    assert cfg.model.diffusion_expert_cuda_graph is True
    assert cfg.model.diffusion_expert_cuda_graph_max_graphs == max_graphs

    opted_out = OmegaConf.merge(
        cfg,
        {"model": {"diffusion_expert_cuda_graph": False}},
    )
    assert opted_out.model.diffusion_expert_cuda_graph is False


def test_all_expected_models_registered() -> None:
    """Every core model must be discoverable via the plugin registry."""
    available = model_registry.get_names()
    for name in EXPECTED_MODELS:
        assert name in available, f"Model '{name}' not found. Available: {available}"


@pytest.mark.parametrize("model_name", EXPECTED_MODELS)
def test_registered_models_have_from_config(model_name: str) -> None:
    """Every registered model class must provide a from_config() classmethod."""
    model_cls = model_registry.get(model_name)
    assert hasattr(
        model_cls, "from_config"
    ), f"Model class {model_cls.__name__} is missing from_config() classmethod"
    assert callable(model_cls.from_config)


@pytest.mark.parametrize("model_name", EXPECTED_MODELS)
def test_registered_models_are_base_trajectory_subclasses(model_name: str) -> None:
    """Every registered model class must be a subclass of BaseTrajectoryModel."""
    model_cls = model_registry.get(model_name)
    assert issubclass(
        model_cls, BaseTrajectoryModel
    ), f"{model_cls.__name__} is not a subclass of BaseTrajectoryModel"
