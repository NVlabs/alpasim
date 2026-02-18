# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Alpasim Controller - Vehicle dynamics and MPC control.

This package provides:
- System: Main simulation system with pluggable MPC controllers
- VehicleModel: Planar dynamic bicycle model for vehicle simulation
- MPCController: Abstract interface for MPC implementations
- LinearMPC: Fast linear MPC using OSQP
- NonlinearMPC: Accurate nonlinear MPC using do_mpc/CasADi
- MPCGains: Cost function weights for MPC

Example usage:
    from alpasim_controller import create_system, System, LinearMPC, MPCGains, MPCImplementation

    # Using factory function (recommended for most cases)
    system = create_system(log_file, initial_state, mpc_implementation=MPCImplementation.LINEAR)

    # Or with explicit controller injection and custom gains
    gains = MPCGains(heading_weight=2.0)
    controller = LinearMPC(gains=gains)
    system = System(log_file, initial_state, controller)
"""

from alpasim_controller.mpc_controller import (
    ControllerInput,
    ControllerOutput,
    MPCController,
    MPCGains,
    MPCImplementation,
)
from alpasim_controller.mpc_impl import LinearMPC, NonlinearMPC
from alpasim_controller.system import System, create_system
from alpasim_controller.vehicle_model import VehicleModel

__all__ = [
    "System",
    "VehicleModel",
    "create_system",
    "MPCController",
    "MPCGains",
    "MPCImplementation",
    "ControllerInput",
    "ControllerOutput",
    "LinearMPC",
    "NonlinearMPC",
]
