# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for VehicleModel."""

import math

import numpy as np
import pytest
from alpasim_controller.vehicle_model import VehicleModel


class TestVehicleModelParameters:
    """Tests for VehicleModel.Parameters dataclass."""

    def test_default_parameters(self):
        """Default parameters should have reasonable values."""
        params = VehicleModel.Parameters()

        assert params.mass > 0
        assert params.inertia > 0
        assert params.wheelbase > 0
        assert params.l_rig_to_cg > 0
        assert params.l_rig_to_cg < params.wheelbase
        assert params.front_cornering_stiffness > 0
        assert params.rear_cornering_stiffness > 0
        assert params.steering_time_constant > 0
        assert params.acceleration_time_constant > 0
        assert params.kinematic_threshold_speed > 0

    def test_custom_parameters(self):
        """Custom parameters should be settable."""
        params = VehicleModel.Parameters(mass=1500.0, wheelbase=3.0)

        assert params.mass == 1500.0
        assert params.wheelbase == 3.0


class TestVehicleModelInit:
    """Tests for VehicleModel initialization."""

    def test_init_stationary(self):
        """Initialize with zero velocity."""
        model = VehicleModel(
            initial_velocity=np.array([0.0, 0.0]),
            initial_yaw_rate=0.0,
        )

        state = model.state
        assert state[0] == 0.0  # x
        assert state[1] == 0.0  # y
        assert state[2] == 0.0  # yaw
        assert state[3] == 0.0  # vx
        assert state[4] == 0.0  # vy
        assert state[5] == 0.0  # yaw_rate
        assert state[6] == 0.0  # steering
        assert state[7] == 0.0  # accel

    def test_init_with_velocity(self):
        """Initialize with forward velocity."""
        vx = 10.0
        model = VehicleModel(
            initial_velocity=np.array([vx, 0.0]),
            initial_yaw_rate=0.0,
        )

        assert model.state[3] == vx
        assert model.state[4] == 0.0

    def test_init_with_yaw_rate(self):
        """Initialize with yaw rate computes initial steering angle."""
        vx = 10.0
        yaw_rate = 0.1
        model = VehicleModel(
            initial_velocity=np.array([vx, 0.0]),
            initial_yaw_rate=yaw_rate,
        )

        # Steering angle should be computed from kinematic relationship
        expected_steering = math.atan(yaw_rate / vx * model.parameters.wheelbase)
        assert model.state[6] == pytest.approx(expected_steering, abs=1e-6)


class TestVehicleModelProperties:
    """Tests for VehicleModel properties."""

    def test_parameters_property(self):
        """parameters property returns Parameters instance."""
        model = VehicleModel(np.array([0.0, 0.0]), 0.0)
        params = model.parameters

        assert isinstance(params, VehicleModel.Parameters)
        assert params.mass > 0

    def test_state_property(self):
        """state property returns 8-element array."""
        model = VehicleModel(np.array([5.0, 0.0]), 0.0)
        state = model.state

        assert isinstance(state, np.ndarray)
        assert state.shape == (8,)

    def test_front_steering_angle_property(self):
        """front_steering_angle returns state[6]."""
        model = VehicleModel(np.array([10.0, 0.0]), 0.1)

        assert model.front_steering_angle == model.state[6]

    def test_accelerations_property(self):
        """accelerations property returns 3-element array."""
        model = VehicleModel(np.array([0.0, 0.0]), 0.0)

        accels = model.accelerations
        assert isinstance(accels, np.ndarray)
        assert accels.shape == (3,)


class TestVehicleModelMethods:
    """Tests for VehicleModel methods."""

    def test_reset_origin(self):
        """reset_origin sets x, y, yaw to zero."""
        model = VehicleModel(np.array([10.0, 0.0]), 0.0)

        # Advance to get non-zero position
        model.advance(np.array([0.0, 0.0]), 0.1)
        assert model.state[0] != 0.0  # x should have changed

        model.reset_origin()

        assert model.state[0] == 0.0
        assert model.state[1] == 0.0
        assert model.state[2] == 0.0
        # Velocity should be preserved
        assert model.state[3] != 0.0

    def test_set_velocity(self):
        """set_velocity updates vx and vy."""
        model = VehicleModel(np.array([0.0, 0.0]), 0.0)

        model.set_velocity(15.0, 0.5)

        assert model.state[3] == 15.0
        assert model.state[4] == 0.5


class TestVehicleModelAdvance:
    """Tests for VehicleModel.advance()."""

    def test_advance_stationary(self):
        """Stationary vehicle should stay stationary."""
        model = VehicleModel(np.array([0.0, 0.0]), 0.0)

        model.advance(np.array([0.0, 0.0]), 0.1)

        assert model.state[0] == pytest.approx(0.0, abs=1e-6)
        assert model.state[1] == pytest.approx(0.0, abs=1e-6)

    def test_advance_straight(self):
        """Vehicle moving straight should increase x position."""
        vx = 10.0  # m/s
        dt = 0.1  # s
        model = VehicleModel(np.array([vx, 0.0]), 0.0)

        model.advance(np.array([0.0, 0.0]), dt)

        # Should move approximately vx * dt in x direction
        assert model.state[0] == pytest.approx(vx * dt, abs=0.1)
        assert model.state[1] == pytest.approx(0.0, abs=0.01)

    def test_advance_returns_qvec(self):
        """advance() returns a QVec pose."""
        model = VehicleModel(np.array([10.0, 0.0]), 0.0)

        pose = model.advance(np.array([0.0, 0.0]), 0.1)

        assert hasattr(pose, "vec3")
        assert hasattr(pose, "quat")
        assert pose.vec3.shape == (3,)
        assert pose.quat.shape == (4,)

    def test_advance_with_acceleration(self):
        """Acceleration command should increase velocity."""
        model = VehicleModel(np.array([5.0, 0.0]), 0.0)
        initial_vx = model.state[3]

        # Apply positive acceleration for several steps
        for _ in range(10):
            model.advance(np.array([0.0, 2.0]), 0.1)

        # Velocity should have increased
        assert model.state[3] > initial_vx

    def test_advance_velocity_non_negative(self):
        """Velocity should not go negative (no reverse)."""
        model = VehicleModel(np.array([1.0, 0.0]), 0.0)

        # Apply strong braking
        for _ in range(50):
            model.advance(np.array([0.0, -5.0]), 0.1)

        # Velocity should be clamped to zero
        assert model.state[3] >= 0.0
