# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Reference trajectory generators for controller benchmarking.

Generates trajectories covering various driving scenarios:
- Straight driving at different speeds (1-25 m/s)
- Left and right turns of varying radii
- Acceleration and deceleration maneuvers
- Combined maneuvers (turn + speed change)
- Stop-then-go scenarios (start from stop, accelerate with turn)
- Deceleration-to-stop scenarios (turn while decelerating to full stop)
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryPoint:
    """A single point in the reference trajectory."""

    timestamp_us: int
    x: float  # position in local frame [m]
    y: float  # position in local frame [m]
    yaw: float  # heading [rad]
    vx: float  # longitudinal velocity [m/s]


@dataclass
class ReferenceTrajectory:
    """A complete reference trajectory for benchmarking."""

    name: str
    description: str
    points: list[TrajectoryPoint]
    simulation_duration_us: int  # How long to run the simulation
    trajectory_duration_us: int  # Total trajectory length (includes lookahead)

    @property
    def simulation_duration_s(self) -> float:
        return self.simulation_duration_us / 1e6

    # Keep duration_us as alias for simulation_duration_us for compatibility
    @property
    def duration_us(self) -> int:
        return self.simulation_duration_us

    @property
    def duration_s(self) -> float:
        return self.simulation_duration_us / 1e6


class TrajectoryGenerator:
    """Generates reference trajectories for controller benchmarking."""

    DT_US = 100_000  # 0.1s timestep (matches MPC)
    SIMULATION_DURATION_S = 10.0  # 10 second simulations
    HORIZON_POINTS = 51  # Number of points in planning horizon (5s lookahead)
    # Trajectory must be long enough for simulation + full horizon lookahead
    TRAJECTORY_DURATION_S = SIMULATION_DURATION_S + (HORIZON_POINTS - 1) * DT_US / 1e6

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_all(self) -> list[ReferenceTrajectory]:
        """Generate all 120 reference trajectories."""
        trajectories = []

        # Straight driving at various speeds (24 trajectories)
        for speed in [1.0, 5.0, 10.0, 15.0, 20.0, 25.0]:
            for i in range(4):
                # Add small random variations
                actual_speed = speed + self.rng.uniform(-1.0, 1.0)
                trajectories.append(
                    self._generate_straight(
                        f"straight_v{speed:.0f}_{i}",
                        f"Straight at {actual_speed:.1f} m/s",
                        actual_speed,
                    )
                )

        # Left turns at various radii (20 trajectories)
        for radius in [20.0, 50.0, 100.0, 200.0, 500.0]:
            for i in range(4):
                speed = min(
                    15.0, math.sqrt(radius * 3.0)
                )  # Reasonable speed for radius
                trajectories.append(
                    self._generate_turn(
                        f"left_r{radius:.0f}_{i}",
                        f"Left turn, radius {radius:.0f}m at {speed:.1f} m/s",
                        radius,
                        speed,
                        direction=1,  # left
                    )
                )

        # Right turns at various radii (20 trajectories)
        for radius in [20.0, 50.0, 100.0, 200.0, 500.0]:
            for i in range(4):
                speed = min(15.0, math.sqrt(radius * 3.0))
                trajectories.append(
                    self._generate_turn(
                        f"right_r{radius:.0f}_{i}",
                        f"Right turn, radius {radius:.0f}m at {speed:.1f} m/s",
                        radius,
                        speed,
                        direction=-1,  # right
                    )
                )

        # Acceleration maneuvers (15 trajectories)
        for v_start, v_end in [
            (5.0, 15.0),
            (10.0, 20.0),
            (5.0, 25.0),
            (15.0, 25.0),
            (8.0, 18.0),
        ]:
            for i in range(3):
                trajectories.append(
                    self._generate_speed_change(
                        f"accel_{v_start:.0f}to{v_end:.0f}_{i}",
                        f"Accelerate from {v_start:.0f} to {v_end:.0f} m/s",
                        v_start,
                        v_end,
                    )
                )

        # Deceleration maneuvers (15 trajectories)
        for v_start, v_end in [
            (20.0, 10.0),
            (25.0, 15.0),
            (15.0, 5.0),
            (20.0, 5.0),
            (18.0, 8.0),
        ]:
            for i in range(3):
                trajectories.append(
                    self._generate_speed_change(
                        f"decel_{v_start:.0f}to{v_end:.0f}_{i}",
                        f"Decelerate from {v_start:.0f} to {v_end:.0f} m/s",
                        v_start,
                        v_end,
                    )
                )

        # Combined maneuvers: turn + speed change (10 trajectories)
        combined_params = [
            (50.0, 1, 10.0, 15.0),  # left turn, accelerate
            (50.0, -1, 10.0, 15.0),  # right turn, accelerate
            (100.0, 1, 15.0, 10.0),  # left turn, decelerate
            (100.0, -1, 15.0, 10.0),  # right turn, decelerate
            (75.0, 1, 8.0, 12.0),  # gentle left, mild accel
        ]
        for i, (radius, direction, v_start, v_end) in enumerate(combined_params):
            dir_name = "left" if direction == 1 else "right"
            change_name = "accel" if v_end > v_start else "decel"
            trajectories.append(
                self._generate_turn_with_speed_change(
                    f"combined_{dir_name}_{change_name}_{i}",
                    f"{dir_name.title()} turn r={radius:.0f}m with {change_name}",
                    radius,
                    direction,
                    v_start,
                    v_end,
                )
            )
            # Add a second variation
            trajectories.append(
                self._generate_turn_with_speed_change(
                    f"combined_{dir_name}_{change_name}_{i}_v2",
                    f"{dir_name.title()} turn r={radius:.0f}m with {change_name} (v2)",
                    radius * 1.2,
                    direction,
                    v_start * 0.9,
                    v_end * 1.1,
                )
            )

        # Stop-then-go with turn scenarios (8 trajectories)
        for radius, direction in [(50.0, 1), (50.0, -1), (100.0, 1), (100.0, -1)]:
            dir_name = "left" if direction == 1 else "right"
            for v_end in [10.0, 15.0]:
                trajectories.append(
                    self._generate_stop_then_go_with_turn(
                        f"stop_go_{dir_name}_r{radius:.0f}_v{v_end:.0f}",
                        f"Stop then {dir_name} turn r={radius:.0f}m to {v_end:.0f} m/s",
                        radius,
                        direction,
                        v_end,
                    )
                )

        # Deceleration to stop with turn scenarios (8 trajectories)
        for radius, direction in [(50.0, 1), (50.0, -1), (100.0, 1), (100.0, -1)]:
            dir_name = "left" if direction == 1 else "right"
            for v_start in [10.0, 15.0]:
                trajectories.append(
                    self._generate_turn_to_stop(
                        f"turn_stop_{dir_name}_r{radius:.0f}_v{v_start:.0f}",
                        f"{dir_name.title()} turn r={radius:.0f}m from {v_start:.0f} m/s to stop",
                        radius,
                        direction,
                        v_start,
                    )
                )

        return trajectories

    def generate_quick_set(self) -> list[ReferenceTrajectory]:
        """Generate a smaller set of trajectories for quick testing."""
        return [
            self._generate_straight("straight_slow", "Straight at 10 m/s", 10.0),
            self._generate_straight("straight_fast", "Straight at 20 m/s", 20.0),
            self._generate_turn(
                "left_tight", "Left turn, radius 30m", 30.0, 10.0, direction=1
            ),
            self._generate_turn(
                "left_wide", "Left turn, radius 100m", 100.0, 15.0, direction=1
            ),
            self._generate_turn(
                "right_tight", "Right turn, radius 30m", 30.0, 10.0, direction=-1
            ),
            self._generate_turn(
                "right_wide", "Right turn, radius 100m", 100.0, 15.0, direction=-1
            ),
            self._generate_speed_change(
                "accel_5to20", "Accelerate 5 to 20 m/s", 5.0, 20.0
            ),
            self._generate_speed_change(
                "decel_20to5", "Decelerate 20 to 5 m/s", 20.0, 5.0
            ),
            self._generate_turn_with_speed_change(
                "combined_left_accel", "Left turn with accel", 80.0, 1, 12.0, 15.0
            ),
            self._generate_turn_with_speed_change(
                "combined_right_decel", "Right turn with decel", 80.0, -1, 15.0, 12.0
            ),
            # Stop-then-go scenarios
            self._generate_stop_then_go_with_turn(
                "stop_go_left", "Stop then left turn to 10 m/s", 80.0, 1, 10.0
            ),
            self._generate_stop_then_go_with_turn(
                "stop_go_right", "Stop then right turn to 10 m/s", 80.0, -1, 10.0
            ),
            # Deceleration to stop scenarios
            self._generate_turn_to_stop(
                "turn_stop_left", "Left turn from 12 m/s to stop", 80.0, 1, 12.0
            ),
            self._generate_turn_to_stop(
                "turn_stop_right", "Right turn from 12 m/s to stop", 80.0, -1, 12.0
            ),
        ]

    def _generate_straight(
        self, name: str, description: str, speed: float
    ) -> ReferenceTrajectory:
        """Generate a straight trajectory at constant speed."""
        points = []
        n_steps = int(self.TRAJECTORY_DURATION_S * 1e6 / self.DT_US)

        for i in range(n_steps + 1):
            t_us = i * self.DT_US
            t_s = t_us / 1e6
            points.append(
                TrajectoryPoint(
                    timestamp_us=t_us,
                    x=speed * t_s,
                    y=0.0,
                    yaw=0.0,
                    vx=speed,
                )
            )

        sim_steps = int(self.SIMULATION_DURATION_S * 1e6 / self.DT_US)
        return ReferenceTrajectory(
            name=name,
            description=description,
            points=points,
            simulation_duration_us=sim_steps * self.DT_US,
            trajectory_duration_us=n_steps * self.DT_US,
        )

    def _generate_turn(
        self,
        name: str,
        description: str,
        radius: float,
        speed: float,
        direction: int,
    ) -> ReferenceTrajectory:
        """Generate a constant-radius turn trajectory.

        Args:
            radius: Turn radius in meters
            speed: Constant speed in m/s
            direction: 1 for left, -1 for right
        """
        points = []
        n_steps = int(self.TRAJECTORY_DURATION_S * 1e6 / self.DT_US)
        angular_rate = direction * speed / radius

        for i in range(n_steps + 1):
            t_us = i * self.DT_US
            t_s = t_us / 1e6

            # Integrate position along arc
            yaw = angular_rate * t_s
            # Position is integral of velocity
            if abs(angular_rate) > 1e-6:
                x = (speed / angular_rate) * math.sin(yaw)
                y = (speed / angular_rate) * (1 - math.cos(yaw))
            else:
                x = speed * t_s
                y = 0.0

            points.append(
                TrajectoryPoint(
                    timestamp_us=t_us,
                    x=x,
                    y=y,
                    yaw=yaw,
                    vx=speed,
                )
            )

        sim_steps = int(self.SIMULATION_DURATION_S * 1e6 / self.DT_US)
        return ReferenceTrajectory(
            name=name,
            description=description,
            points=points,
            simulation_duration_us=sim_steps * self.DT_US,
            trajectory_duration_us=n_steps * self.DT_US,
        )

    def _generate_speed_change(
        self, name: str, description: str, v_start: float, v_end: float
    ) -> ReferenceTrajectory:
        """Generate a straight trajectory with speed change.

        Speed changes linearly over the first half, then holds constant.
        """
        points = []
        n_steps = int(self.TRAJECTORY_DURATION_S * 1e6 / self.DT_US)
        ramp_steps = n_steps // 2

        x = 0.0
        for i in range(n_steps + 1):
            t_us = i * self.DT_US

            # Linear speed ramp over first half
            if i <= ramp_steps:
                alpha = i / ramp_steps
                vx = v_start + alpha * (v_end - v_start)
            else:
                vx = v_end

            points.append(
                TrajectoryPoint(
                    timestamp_us=t_us,
                    x=x,
                    y=0.0,
                    yaw=0.0,
                    vx=vx,
                )
            )

            # Integrate position
            if i < n_steps:
                dt = self.DT_US / 1e6
                x += vx * dt

        sim_steps = int(self.SIMULATION_DURATION_S * 1e6 / self.DT_US)
        return ReferenceTrajectory(
            name=name,
            description=description,
            points=points,
            simulation_duration_us=sim_steps * self.DT_US,
            trajectory_duration_us=n_steps * self.DT_US,
        )

    def _generate_turn_with_speed_change(
        self,
        name: str,
        description: str,
        radius: float,
        direction: int,
        v_start: float,
        v_end: float,
    ) -> ReferenceTrajectory:
        """Generate a turn trajectory with speed change."""
        points = []
        n_steps = int(self.TRAJECTORY_DURATION_S * 1e6 / self.DT_US)
        ramp_steps = n_steps // 2

        x, y, yaw = 0.0, 0.0, 0.0

        for i in range(n_steps + 1):
            t_us = i * self.DT_US

            # Linear speed ramp over first half
            if i <= ramp_steps:
                alpha = i / ramp_steps
                vx = v_start + alpha * (v_end - v_start)
            else:
                vx = v_end

            points.append(
                TrajectoryPoint(
                    timestamp_us=t_us,
                    x=x,
                    y=y,
                    yaw=yaw,
                    vx=vx,
                )
            )

            # Integrate for next step
            if i < n_steps:
                dt = self.DT_US / 1e6
                angular_rate = direction * vx / radius
                yaw += angular_rate * dt
                x += vx * math.cos(yaw) * dt
                y += vx * math.sin(yaw) * dt

        sim_steps = int(self.SIMULATION_DURATION_S * 1e6 / self.DT_US)
        return ReferenceTrajectory(
            name=name,
            description=description,
            points=points,
            simulation_duration_us=sim_steps * self.DT_US,
            trajectory_duration_us=n_steps * self.DT_US,
        )

    def _generate_stop_then_go_with_turn(
        self,
        name: str,
        description: str,
        radius: float,
        direction: int,
        v_end: float,
    ) -> ReferenceTrajectory:
        """Generate a trajectory that starts stopped, then accelerates into a turn.

        First half: vehicle remains at stop (vx=0)
        Second half: gentle acceleration while turning

        Args:
            radius: Turn radius in meters
            direction: 1 for left, -1 for right
            v_end: Final velocity in m/s
        """
        points = []
        n_steps = int(self.TRAJECTORY_DURATION_S * 1e6 / self.DT_US)
        stop_steps = n_steps // 2  # First half is stopped
        accel_steps = n_steps - stop_steps  # Second half accelerates

        x, y, yaw = 0.0, 0.0, 0.0

        for i in range(n_steps + 1):
            t_us = i * self.DT_US

            if i <= stop_steps:
                # First half: stopped
                vx = 0.0
            else:
                # Second half: linear acceleration to v_end
                alpha = (i - stop_steps) / accel_steps
                vx = alpha * v_end

            points.append(
                TrajectoryPoint(
                    timestamp_us=t_us,
                    x=x,
                    y=y,
                    yaw=yaw,
                    vx=vx,
                )
            )

            # Integrate for next step
            if i < n_steps and vx > 0:
                dt = self.DT_US / 1e6
                angular_rate = direction * vx / radius
                yaw += angular_rate * dt
                x += vx * math.cos(yaw) * dt
                y += vx * math.sin(yaw) * dt

        sim_steps = int(self.SIMULATION_DURATION_S * 1e6 / self.DT_US)
        return ReferenceTrajectory(
            name=name,
            description=description,
            points=points,
            simulation_duration_us=sim_steps * self.DT_US,
            trajectory_duration_us=n_steps * self.DT_US,
        )

    def _generate_turn_to_stop(
        self,
        name: str,
        description: str,
        radius: float,
        direction: int,
        v_start: float,
    ) -> ReferenceTrajectory:
        """Generate a trajectory that turns while decelerating to a full stop.

        First 25%: constant speed with turn
        25% to 75%: decelerate to stop while turning
        75% to 100%: hold at stop

        Args:
            radius: Turn radius in meters
            direction: 1 for left, -1 for right
            v_start: Initial velocity in m/s
        """
        points = []
        n_steps = int(self.TRAJECTORY_DURATION_S * 1e6 / self.DT_US)
        constant_steps = n_steps // 4  # First 25% at constant speed
        decel_steps = n_steps // 2  # 25% to 75% decelerating
        # Remaining 25% at stop

        x, y, yaw = 0.0, 0.0, 0.0

        for i in range(n_steps + 1):
            t_us = i * self.DT_US

            if i <= constant_steps:
                # First 25%: constant speed
                vx = v_start
            elif i <= constant_steps + decel_steps:
                # 25% to 75%: linear deceleration to zero
                alpha = (i - constant_steps) / decel_steps
                vx = v_start * (1 - alpha)
            else:
                # 75% to 100%: stopped
                vx = 0.0

            points.append(
                TrajectoryPoint(
                    timestamp_us=t_us,
                    x=x,
                    y=y,
                    yaw=yaw,
                    vx=vx,
                )
            )

            # Integrate for next step
            if i < n_steps and vx > 0:
                dt = self.DT_US / 1e6
                angular_rate = direction * vx / radius
                yaw += angular_rate * dt
                x += vx * math.cos(yaw) * dt
                y += vx * math.sin(yaw) * dt

        sim_steps = int(self.SIMULATION_DURATION_S * 1e6 / self.DT_US)
        return ReferenceTrajectory(
            name=name,
            description=description,
            points=points,
            simulation_duration_us=sim_steps * self.DT_US,
            trajectory_duration_us=n_steps * self.DT_US,
        )

    def get_horizon_at_time(
        self, trajectory: ReferenceTrajectory, current_time_us: int
    ) -> list[TrajectoryPoint]:
        """Extract the planning horizon starting at current_time_us.

        Returns HORIZON_POINTS points starting from current_time_us.
        If the trajectory ends before the horizon, the last point is repeated.
        """
        horizon = []
        for i in range(self.HORIZON_POINTS):
            target_time = current_time_us + i * self.DT_US

            # Find the point at or after target_time
            point = None
            for p in trajectory.points:
                if p.timestamp_us >= target_time:
                    point = p
                    break

            if point is None:
                # Past end of trajectory, use last point
                point = trajectory.points[-1]

            horizon.append(point)

        return horizon
