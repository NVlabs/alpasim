# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for driver schema and plugin-based model discovery."""

import pytest
from alpasim_driver.models.base import BaseTrajectoryModel
from alpasim_plugins.plugins import models as model_registry

# The entry-point names that the driver package registers.
# transfuser is provided by the optional alpasim_transfuser plugin, not the core driver.
EXPECTED_MODELS = ["ar1", "manual", "pdm", "vam"]


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
