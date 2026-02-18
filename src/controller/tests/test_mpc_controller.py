# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for MPC controller interface and dataclasses."""

import pytest
from alpasim_controller.mpc_controller import MPCController, MPCGains


class TestMPCGains:
    """Tests for MPCGains dataclass."""

    def test_all_weights_positive(self):
        """Default weights should all be non-negative."""
        gains = MPCGains()

        assert gains.long_position_weight >= 0
        assert gains.lat_position_weight >= 0
        assert gains.heading_weight >= 0
        assert gains.acceleration_weight >= 0
        assert gains.rel_front_steering_angle_weight >= 0
        assert gains.rel_acceleration_weight >= 0
        assert gains.idx_start_penalty >= 0


class TestMPCControllerInterface:
    """Tests for MPCController abstract interface."""

    def test_abstract_methods(self):
        """MPCController should be abstract and not instantiable."""
        with pytest.raises(TypeError):
            MPCController()
