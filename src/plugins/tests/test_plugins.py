# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the plugin registry and discovery (Phase 1 extensible framework)."""

import pytest
from alpasim_plugins.plugins import (
    PluginNotFoundError,
    PluginRegistry,
    get_plugin_info,
    models,
    mpc_controllers,
)


def test_plugin_registry_nonexistent_group_returns_empty() -> None:
    """A group with no entry points returns empty list from get_names()."""
    registry = PluginRegistry("alpasim.nonexistent.group.xyz")
    assert registry.get_names() == []
    assert registry.get_available() == {}
    assert registry.is_available("anything") is False
    with pytest.raises(PluginNotFoundError, match="not found"):
        registry.get("anything")


def test_plugin_registry_get_names_sorted() -> None:
    """get_names() returns sorted list."""
    registry = PluginRegistry("alpasim.scorers")  # typically empty
    names = registry.get_names()
    assert isinstance(names, list)
    assert names == sorted(names)


def test_get_plugin_info_returns_all_groups() -> None:
    """get_plugin_info() returns dict with expected group keys."""
    info = get_plugin_info()
    expected_groups = [
        "alpasim.models",
        "alpasim.mpc",
        "alpasim.scorers",
        "alpasim.tools",
        "alpasim.configs",
    ]
    for group in expected_groups:
        assert group in info
        assert isinstance(info[group], list)


def test_models_registry_has_core_models_when_driver_installed() -> None:
    """When alpasim_driver is installed, alpasim.models lists ar1, manual, vam."""
    names = models.get_names()
    expected = {"ar1", "manual", "vam"}
    for name in expected:
        assert name in names, f"Expected model {name} in alpasim.models (got {names})"


def test_mpc_registry_has_linear_and_nonlinear_when_controller_installed() -> None:
    """When alpasim_controller is installed, alpasim.mpc lists linear and nonlinear."""
    names = mpc_controllers.get_names()
    expected = {"linear", "nonlinear"}
    for name in expected:
        assert name in names, f"Expected MPC {name} in alpasim.mpc (got {names})"


def test_mpc_registry_create_linear_returns_controller() -> None:
    """mpc_controllers.create('linear', vehicle_params=...) returns an MPC instance."""
    from alpasim_controller.vehicle_model import VehicleModel

    params = VehicleModel.Parameters()
    controller = mpc_controllers.create("linear", vehicle_params=params)
    assert controller is not None
    assert hasattr(controller, "compute_control")
