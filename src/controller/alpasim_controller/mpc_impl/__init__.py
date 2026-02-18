# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""MPC implementations for the controller.

This subpackage contains the concrete MPC controller implementations:
- LinearMPC: Fast linear MPC using OSQP
- NonlinearMPC: Accurate nonlinear MPC using do_mpc/CasADi
"""

from alpasim_controller.mpc_impl.linear_mpc import LinearMPC
from alpasim_controller.mpc_impl.nonlinear_mpc import NonlinearMPC

__all__ = ["LinearMPC", "NonlinearMPC"]
