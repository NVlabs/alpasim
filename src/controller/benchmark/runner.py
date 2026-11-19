# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Benchmark runner for controller performance testing.

Runs closed-loop simulations and collects timing and trajectory data.
"""

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from alpasim_controller.mpc_controller import ControllerConfig
from alpasim_controller.system_manager import SystemManager
from alpasim_controller.vehicle_model import VehicleModel
from alpasim_grpc.v0 import common_pb2, controller_pb2

from .trajectories import ReferenceTrajectory, TrajectoryGenerator, TrajectoryPoint


@dataclass(frozen=True)
class BenchmarkControlLimits:
    """Control and smoothness bounds reported by the benchmark."""

    steering_cmd_abs_rad: float = 2.0
    acceleration_cmd_min_mps2: float = -9.0
    acceleration_cmd_max_mps2: float = 6.0
    jerk_cmd_abs_mps3: float = 20.0
    curvature_abs_1pm: float = 0.35
    curvature_rate_abs_1pmps: float = 2.0

    def to_dict(self) -> dict[str, float]:
        """Convert limits to JSON fields."""
        return asdict(self)


@dataclass
class IterationResult:
    """Results from a single MPC iteration."""

    timestamp_us: int
    solve_time_ms: float
    x: float
    y: float
    yaw: float
    vx: float  # Vehicle speed (m/s)
    ref_x: float
    ref_y: float
    ref_yaw: float
    steering_cmd: float = 0.0
    acceleration_cmd: float = 0.0
    jerk_cmd: float = 0.0
    curvature: float = 0.0
    curvature_rate: float = 0.0
    controller_status: str = ""


@dataclass
class SimulationResult:
    """Results from a complete closed-loop simulation."""

    trajectory_name: str
    trajectory_description: str
    total_time_ms: float
    iterations: list[IterationResult]

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    @property
    def min_solve_time_ms(self) -> float:
        return min(it.solve_time_ms for it in self.iterations)

    @property
    def max_solve_time_ms(self) -> float:
        return max(it.solve_time_ms for it in self.iterations)

    @property
    def mean_solve_time_ms(self) -> float:
        return sum(it.solve_time_ms for it in self.iterations) / len(self.iterations)

    @property
    def timestamps_s(self) -> list[float]:
        return [it.timestamp_us / 1e6 for it in self.iterations]

    @property
    def x_positions(self) -> list[float]:
        return [it.x for it in self.iterations]

    @property
    def y_positions(self) -> list[float]:
        return [it.y for it in self.iterations]

    @property
    def ref_x_positions(self) -> list[float]:
        return [it.ref_x for it in self.iterations]

    @property
    def ref_y_positions(self) -> list[float]:
        return [it.ref_y for it in self.iterations]

    @property
    def velocities(self) -> list[float]:
        return [it.vx for it in self.iterations]

    @property
    def xy_errors_m(self) -> list[float]:
        return [math.hypot(it.x - it.ref_x, it.y - it.ref_y) for it in self.iterations]

    @property
    def tracking_rmse_m(self) -> float:
        return _rms(self.xy_errors_m)

    @property
    def tracking_max_error_m(self) -> float:
        return max(self.xy_errors_m)

    def control_summary(
        self,
        limits: BenchmarkControlLimits | None = None,
    ) -> dict[str, float | int | dict[str, float]]:
        """Summarize control effort, smoothness, and bound violations."""
        limits = limits or BenchmarkControlLimits()
        steering_cmds = [it.steering_cmd for it in self.iterations]
        acceleration_cmds = [it.acceleration_cmd for it in self.iterations]
        jerk_cmds = [it.jerk_cmd for it in self.iterations]
        curvatures = [it.curvature for it in self.iterations]
        curvature_rates = [it.curvature_rate for it in self.iterations]

        violation_counts = _control_violation_counts(self.iterations, limits)
        return {
            "limits": limits.to_dict(),
            "tracking_rmse_m": self.tracking_rmse_m,
            "tracking_max_error_m": self.tracking_max_error_m,
            "steering_cmd_abs_max_rad": _max_abs(steering_cmds),
            "steering_cmd_abs_mean_rad": _mean_abs(steering_cmds),
            "steering_cmd_rms_rad": _rms(steering_cmds),
            "acceleration_cmd_abs_max_mps2": _max_abs(acceleration_cmds),
            "acceleration_cmd_abs_mean_mps2": _mean_abs(acceleration_cmds),
            "acceleration_cmd_min_mps2": min(acceleration_cmds),
            "acceleration_cmd_max_mps2": max(acceleration_cmds),
            "acceleration_cmd_rms_mps2": _rms(acceleration_cmds),
            "jerk_cmd_abs_max_mps3": _max_abs(jerk_cmds),
            "jerk_cmd_abs_mean_mps3": _mean_abs(jerk_cmds),
            "jerk_cmd_rms_mps3": _rms(jerk_cmds),
            "curvature_abs_max_1pm": _max_abs(curvatures),
            "curvature_abs_mean_1pm": _mean_abs(curvatures),
            "curvature_rms_1pm": _rms(curvatures),
            "curvature_rate_abs_max_1pmps": _max_abs(curvature_rates),
            "curvature_rate_abs_mean_1pmps": _mean_abs(curvature_rates),
            "curvature_rate_rms_1pmps": _rms(curvature_rates),
            "control_aggressiveness_score": _control_aggressiveness_score(
                steering_cmds,
                acceleration_cmds,
                jerk_cmds,
                curvatures,
                curvature_rates,
                limits,
            ),
            "violation_counts": violation_counts,
            "total_violation_count": sum(violation_counts.values()),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    timestamp: str
    description: str
    simulations: list[SimulationResult]

    @property
    def total_time_s(self) -> float:
        return sum(sim.total_time_ms for sim in self.simulations) / 1000.0

    @property
    def n_simulations(self) -> int:
        return len(self.simulations)

    def summary_stats(self) -> dict:
        """Compute aggregate statistics across all simulations."""
        all_solve_times = [
            it.solve_time_ms for sim in self.simulations for it in sim.iterations
        ]
        all_iterations = [it for sim in self.simulations for it in sim.iterations]
        control_summary = _control_summary_for_iterations(all_iterations)
        return {
            "n_simulations": self.n_simulations,
            "total_iterations": len(all_solve_times),
            "total_time_s": self.total_time_s,
            "solve_time_min_ms": min(all_solve_times),
            "solve_time_max_ms": max(all_solve_times),
            "solve_time_mean_ms": sum(all_solve_times) / len(all_solve_times),
            "solve_time_median_ms": sorted(all_solve_times)[len(all_solve_times) // 2],
            **control_summary,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "description": self.description,
            "summary": self.summary_stats(),
            "simulations": [
                {
                    "trajectory_name": sim.trajectory_name,
                    "trajectory_description": sim.trajectory_description,
                    "total_time_ms": sim.total_time_ms,
                    "min_solve_time_ms": sim.min_solve_time_ms,
                    "max_solve_time_ms": sim.max_solve_time_ms,
                    "mean_solve_time_ms": sim.mean_solve_time_ms,
                    "control_summary": sim.control_summary(),
                    "iterations": [asdict(it) for it in sim.iterations],
                }
                for sim in self.simulations
            ],
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkResult":
        """Load results from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        simulations = []
        for sim_data in data["simulations"]:
            iterations = []
            for it_data in sim_data["iterations"]:
                # Handle backward compatibility for old files without vx
                if "vx" not in it_data:
                    it_data["vx"] = 0.0
                if "controller_status" not in it_data:
                    it_data["controller_status"] = ""
                for field_name in (
                    "steering_cmd",
                    "acceleration_cmd",
                    "jerk_cmd",
                    "curvature",
                    "curvature_rate",
                ):
                    if field_name not in it_data:
                        it_data[field_name] = 0.0
                iterations.append(IterationResult(**it_data))
            simulations.append(
                SimulationResult(
                    trajectory_name=sim_data["trajectory_name"],
                    trajectory_description=sim_data["trajectory_description"],
                    total_time_ms=sim_data["total_time_ms"],
                    iterations=iterations,
                )
            )

        return cls(
            timestamp=data["timestamp"],
            description=data.get("description", ""),
            simulations=simulations,
        )


class BenchmarkRunner:
    """Runs controller benchmarks."""

    DT_US = 100_000  # 0.1s timestep
    SESSION_UUID = "benchmark_session"

    def __init__(
        self,
        controller_config: ControllerConfig,
        log_dir: Path | None = None,
    ):
        """Initialize the benchmark runner.

        Args:
            controller_config: Full controller configuration.
            log_dir: Directory for controller logs (default: temp directory)
        """
        self.log_dir = log_dir or Path(tempfile.mkdtemp())
        self.trajectory_generator = TrajectoryGenerator()
        self.controller_config = controller_config

    def run(
        self,
        quick: bool = False,
        description: str = "",
        trajectories: list[ReferenceTrajectory] | None = None,
    ) -> BenchmarkResult:
        """Run the full benchmark suite.

        Args:
            quick: If True, run only 10 trajectories instead of ~104
            description: Description of this benchmark run
            trajectories: Explicit reference trajectories. If set, overrides quick/full generation.

        Returns:
            BenchmarkResult with all simulation results
        """
        if trajectories is None and quick:
            trajectories = self.trajectory_generator.generate_quick_set()
        elif trajectories is None:
            trajectories = self.trajectory_generator.generate_all()

        if not trajectories:
            raise ValueError(
                "No trajectories to benchmark. Provide a non-empty trajectory set."
            )
        invalid = [
            traj.name
            for traj in trajectories
            if traj.simulation_duration_us % self.DT_US != 0
        ]
        if invalid:
            raise ValueError(
                f"Trajectory simulation durations must be multiples of {self.DT_US} us: "
                + ", ".join(invalid[:5])
            )

        print(f"Running benchmark with {len(trajectories)} trajectories...")

        simulations = []
        for i, traj in enumerate(trajectories):
            print(f"  [{i + 1}/{len(trajectories)}] {traj.name}: {traj.description}")
            result = self._run_simulation(traj)
            simulations.append(result)
            print(
                f"    -> {result.n_iterations} iterations, "
                f"mean={result.mean_solve_time_ms:.1f}ms, "
                f"max={result.max_solve_time_ms:.1f}ms"
            )

        return BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            description=description,
            simulations=simulations,
        )

    def _run_simulation(self, trajectory: ReferenceTrajectory) -> SimulationResult:
        """Run a single closed-loop simulation."""
        system_manager = SystemManager(
            str(self.log_dir), controller_config=self.controller_config
        )
        system_manager.start_session(self.SESSION_UUID)

        iterations = []
        sim_start_time = time.perf_counter()
        previous_acceleration_cmd: float | None = None
        previous_curvature: float | None = None
        wheelbase = VehicleModel.Parameters().wheelbase

        # Initialize state at trajectory start
        state = common_pb2.StateAtTime()
        state.timestamp_us = 0
        state.pose.vec.x = trajectory.points[0].x
        state.pose.vec.y = trajectory.points[0].y
        state.pose.quat.z = math.sin(trajectory.points[0].yaw / 2)
        state.pose.quat.w = math.cos(trajectory.points[0].yaw / 2)
        state.state.linear_velocity.x = trajectory.points[0].vx

        n_steps = int(trajectory.duration_us / self.DT_US)

        for step in range(n_steps):
            current_time_us = step * self.DT_US
            future_time_us = current_time_us + self.DT_US

            # Build request
            request = self._build_request(
                state, trajectory, current_time_us, future_time_us
            )

            # Time the MPC solve
            iter_start = time.perf_counter()
            response = system_manager.run_controller_and_vehicle_model(request)
            iter_time_ms = (time.perf_counter() - iter_start) * 1000.0
            control_input = system_manager.get_control_input(self.SESSION_UUID)
            controller_status = system_manager.get_controller_status(self.SESSION_UUID)

            # Extract result — use the final propagated state
            final = response.states[-1]
            ref_point = self._get_reference_at_time(trajectory, final.timestamp_us)
            result_pose = final.pose_local_to_rig
            result_yaw = 2.0 * math.atan2(result_pose.quat.z, result_pose.quat.w)
            result_vx = final.dynamic_state.linear_velocity.x
            result_yaw_rate = final.dynamic_state.angular_velocity.z
            steering_cmd = float(control_input[0])
            acceleration_cmd = float(control_input[1])
            jerk_cmd = 0.0
            if previous_acceleration_cmd is not None:
                jerk_cmd = (acceleration_cmd - previous_acceleration_cmd) / (
                    self.DT_US / 1e6
                )
            curvature = _curvature_from_state(
                vx=result_vx,
                yaw_rate=result_yaw_rate,
                steering_cmd=steering_cmd,
                wheelbase=wheelbase,
            )
            curvature_rate = 0.0
            if previous_curvature is not None:
                curvature_rate = (curvature - previous_curvature) / (self.DT_US / 1e6)
            previous_acceleration_cmd = acceleration_cmd
            previous_curvature = curvature

            iterations.append(
                IterationResult(
                    timestamp_us=final.timestamp_us,
                    solve_time_ms=iter_time_ms,
                    x=result_pose.vec.x,
                    y=result_pose.vec.y,
                    yaw=result_yaw,
                    vx=result_vx,
                    ref_x=ref_point.x,
                    ref_y=ref_point.y,
                    ref_yaw=ref_point.yaw,
                    steering_cmd=steering_cmd,
                    acceleration_cmd=acceleration_cmd,
                    jerk_cmd=jerk_cmd,
                    curvature=curvature,
                    curvature_rate=curvature_rate,
                    controller_status=controller_status,
                )
            )

            # Update state for next iteration
            state = common_pb2.StateAtTime()
            state.timestamp_us = final.timestamp_us
            state.pose.CopyFrom(final.pose_local_to_rig)
            state.state.CopyFrom(final.dynamic_state)

        # Close session
        close_request = controller_pb2.VDCSessionCloseRequest(
            session_uuid=self.SESSION_UUID
        )
        system_manager.close_session(close_request)

        total_time_ms = (time.perf_counter() - sim_start_time) * 1000.0

        return SimulationResult(
            trajectory_name=trajectory.name,
            trajectory_description=trajectory.description,
            total_time_ms=total_time_ms,
            iterations=iterations,
        )

    def _build_request(
        self,
        state: common_pb2.StateAtTime,
        trajectory: ReferenceTrajectory,
        current_time_us: int,
        future_time_us: int,
    ) -> controller_pb2.RunControllerAndVehicleModelRequest:
        """Build a controller request with the planning horizon."""
        request = controller_pb2.RunControllerAndVehicleModelRequest()
        request.session_uuid = self.SESSION_UUID
        request.state.CopyFrom(state)
        request.state.timestamp_us = current_time_us
        request.future_time_us = future_time_us

        # Get current pose for transforming trajectory to rig frame
        current_x = state.pose.vec.x
        current_y = state.pose.vec.y
        current_yaw = 2.0 * math.atan2(state.pose.quat.z, state.pose.quat.w)

        # Build planning horizon in rig frame
        horizon = self.trajectory_generator.get_horizon_at_time(
            trajectory, current_time_us
        )

        for i, point in enumerate(horizon):
            pose_at_time = request.planned_trajectory_in_rig.poses.add()
            # Use computed horizon timestamp, not the reference point's timestamp
            pose_at_time.timestamp_us = current_time_us + i * self.DT_US

            # Transform from local frame to rig frame
            dx = point.x - current_x
            dy = point.y - current_y
            cos_yaw = math.cos(-current_yaw)
            sin_yaw = math.sin(-current_yaw)

            pose_at_time.pose.vec.x = cos_yaw * dx - sin_yaw * dy
            pose_at_time.pose.vec.y = sin_yaw * dx + cos_yaw * dy

            # Relative yaw
            rel_yaw = point.yaw - current_yaw
            pose_at_time.pose.quat.z = math.sin(rel_yaw / 2)
            pose_at_time.pose.quat.w = math.cos(rel_yaw / 2)

        return request

    def _get_reference_at_time(
        self, trajectory: ReferenceTrajectory, time_us: int
    ) -> TrajectoryPoint:
        """Get the reference trajectory point at a given time."""
        for point in trajectory.points:
            if point.timestamp_us >= time_us:
                return point
        return trajectory.points[-1]


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _curvature_from_state(
    vx: float,
    yaw_rate: float,
    steering_cmd: float,
    wheelbase: float,
) -> float:
    """Estimate finite curvature from vehicle motion or low-speed steering."""
    if abs(vx) > 0.5:
        return yaw_rate / vx

    steer_for_curvature = max(
        min(steering_cmd, (math.pi / 2) - 1e-3),
        -(math.pi / 2) + 1e-3,
    )
    return math.tan(steer_for_curvature) / wheelbase


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(abs(value) for value in values)


def _mean_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _control_summary_for_iterations(
    iterations: list[IterationResult],
    limits: BenchmarkControlLimits | None = None,
) -> dict[str, float | int | dict[str, float] | dict[str, int]]:
    limits = limits or BenchmarkControlLimits()
    xy_errors = [math.hypot(it.x - it.ref_x, it.y - it.ref_y) for it in iterations]
    steering_cmds = [it.steering_cmd for it in iterations]
    acceleration_cmds = [it.acceleration_cmd for it in iterations]
    jerk_cmds = [it.jerk_cmd for it in iterations]
    curvatures = [it.curvature for it in iterations]
    curvature_rates = [it.curvature_rate for it in iterations]
    violation_counts = _control_violation_counts(iterations, limits)
    return {
        "control_limits": limits.to_dict(),
        "tracking_rmse_m": _rms(xy_errors),
        "tracking_max_error_m": max(xy_errors) if xy_errors else 0.0,
        "steering_cmd_abs_max_rad": _max_abs(steering_cmds),
        "steering_cmd_abs_mean_rad": _mean_abs(steering_cmds),
        "steering_cmd_rms_rad": _rms(steering_cmds),
        "acceleration_cmd_abs_max_mps2": _max_abs(acceleration_cmds),
        "acceleration_cmd_abs_mean_mps2": _mean_abs(acceleration_cmds),
        "acceleration_cmd_min_mps2": (
            min(acceleration_cmds) if acceleration_cmds else 0.0
        ),
        "acceleration_cmd_max_mps2": (
            max(acceleration_cmds) if acceleration_cmds else 0.0
        ),
        "acceleration_cmd_rms_mps2": _rms(acceleration_cmds),
        "jerk_cmd_abs_max_mps3": _max_abs(jerk_cmds),
        "jerk_cmd_abs_mean_mps3": _mean_abs(jerk_cmds),
        "jerk_cmd_rms_mps3": _rms(jerk_cmds),
        "curvature_abs_max_1pm": _max_abs(curvatures),
        "curvature_abs_mean_1pm": _mean_abs(curvatures),
        "curvature_rms_1pm": _rms(curvatures),
        "curvature_rate_abs_max_1pmps": _max_abs(curvature_rates),
        "curvature_rate_abs_mean_1pmps": _mean_abs(curvature_rates),
        "curvature_rate_rms_1pmps": _rms(curvature_rates),
        "control_aggressiveness_score": _control_aggressiveness_score(
            steering_cmds,
            acceleration_cmds,
            jerk_cmds,
            curvatures,
            curvature_rates,
            limits,
        ),
        "control_violation_counts": violation_counts,
        "total_control_violation_count": sum(violation_counts.values()),
    }


def _control_violation_counts(
    iterations: list[IterationResult],
    limits: BenchmarkControlLimits,
) -> dict[str, int]:
    return {
        "steering_cmd_abs": sum(
            abs(it.steering_cmd) > limits.steering_cmd_abs_rad for it in iterations
        ),
        "acceleration_cmd_min": sum(
            it.acceleration_cmd < limits.acceleration_cmd_min_mps2 for it in iterations
        ),
        "acceleration_cmd_max": sum(
            it.acceleration_cmd > limits.acceleration_cmd_max_mps2 for it in iterations
        ),
        "jerk_cmd_abs": sum(
            abs(it.jerk_cmd) > limits.jerk_cmd_abs_mps3 for it in iterations
        ),
        "curvature_abs": sum(
            abs(it.curvature) > limits.curvature_abs_1pm for it in iterations
        ),
        "curvature_rate_abs": sum(
            abs(it.curvature_rate) > limits.curvature_rate_abs_1pmps
            for it in iterations
        ),
    }


def _control_aggressiveness_score(
    steering_cmds: list[float],
    acceleration_cmds: list[float],
    jerk_cmds: list[float],
    curvatures: list[float],
    curvature_rates: list[float],
    limits: BenchmarkControlLimits,
) -> float:
    accel_limit = max(
        abs(limits.acceleration_cmd_min_mps2),
        abs(limits.acceleration_cmd_max_mps2),
    )
    normalized_rms_values = [
        _rms(steering_cmds) / limits.steering_cmd_abs_rad,
        _rms(acceleration_cmds) / accel_limit,
        _rms(jerk_cmds) / limits.jerk_cmd_abs_mps3,
        _rms(curvatures) / limits.curvature_abs_1pm,
        _rms(curvature_rates) / limits.curvature_rate_abs_1pmps,
    ]
    return _rms(normalized_rms_values)
