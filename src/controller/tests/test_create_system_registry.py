# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests that create_system uses the plugin registry."""

import os
import tempfile

import pytest
from alpasim_controller.mpc_controller import ControllerConfig, MPCImplementation
from alpasim_controller.system import create_system
from alpasim_grpc.v0 import common_pb2


def _make_initial_state() -> common_pb2.StateAtTime:
    state = common_pb2.StateAtTime()
    state.timestamp_us = 0
    state.pose.vec.x = 0.0
    state.pose.vec.y = 0.0
    state.pose.vec.z = 0.0
    state.pose.quat.x = 0.0
    state.pose.quat.y = 0.0
    state.pose.quat.z = 0.0
    state.pose.quat.w = 1.0
    state.state.linear_velocity.x = 5.0
    state.state.linear_velocity.y = 0.0
    state.state.angular_velocity.z = 0.0
    return state


@pytest.mark.parametrize(
    "mpc_impl", [MPCImplementation.LINEAR, MPCImplementation.NONLINEAR]
)
def test_create_system_via_registry(mpc_impl: MPCImplementation) -> None:
    """create_system() returns a System with a controller from the plugin registry."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        log_file = f.name
    try:
        initial_state = _make_initial_state()
        system = create_system(
            log_file=log_file,
            initial_state=initial_state,
            controller_config=ControllerConfig(mpc_implementation=str(mpc_impl)),
        )
        assert system is not None
        assert hasattr(system, "_controller")
        assert hasattr(system._controller, "compute_control")
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)
