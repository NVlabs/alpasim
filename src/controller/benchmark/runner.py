# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Benchmark runner for controller performance testing.

Runs closed-loop simulations and collects timing and trajectory data.
"""

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from alpasim_controller.mpc_controller import MPCImplementation
from alpasim_controller.system_manager import SystemManager
from alpasim_grpc.v0 import common_pb2, controller_pb2

from .trajectories import ReferenceTrajectory, TrajectoryGenerator, TrajectoryPoint


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
        return {
            "n_simulations": self.n_simulations,
            "total_iterations": len(all_solve_times),
            "total_time_s": self.total_time_s,
            "solve_time_min_ms": min(all_solve_times),
            "solve_time_max_ms": max(all_solve_times),
            "solve_time_mean_ms": sum(all_solve_times) / len(all_solve_times),
            "solve_time_median_ms": sorted(all_solve_times)[len(all_solve_times) // 2],
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
        log_dir: Optional[Path] = None,
        mpc_implementation: MPCImplementation | None = None,
    ):
        """
        Initialize the benchmark runner.

        Args:
            log_dir: Directory for controller logs (default: temp directory)
            mpc_implementation: MPC implementation to use (default: LINEAR)
        """
        self.log_dir = log_dir or Path(tempfile.mkdtemp())
        self.trajectory_generator = TrajectoryGenerator()
        self.mpc_implementation = mpc_implementation or MPCImplementation.LINEAR

    def run(
        self,
        quick: bool = False,
        description: str = "",
    ) -> BenchmarkResult:
        """Run the full benchmark suite.

        Args:
            quick: If True, run only 10 trajectories instead of ~104
            description: Description of this benchmark run

        Returns:
            BenchmarkResult with all simulation results
        """
        if quick:
            trajectories = self.trajectory_generator.generate_quick_set()
        else:
            trajectories = self.trajectory_generator.generate_all()

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
            str(self.log_dir), mpc_implementation=self.mpc_implementation
        )
        system_manager.start_session(self.SESSION_UUID)

        iterations = []
        sim_start_time = time.perf_counter()

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

            # Get reference point for this timestep
            ref_point = self._get_reference_at_time(trajectory, current_time_us)

            # Time the MPC solve
            iter_start = time.perf_counter()
            response = system_manager.run_controller_and_vehicle_model(request)
            iter_time_ms = (time.perf_counter() - iter_start) * 1000.0

            # Extract result
            result_pose = response.pose_local_to_rig.pose
            result_yaw = 2.0 * math.atan2(result_pose.quat.z, result_pose.quat.w)
            result_vx = response.dynamic_state.linear_velocity.x

            iterations.append(
                IterationResult(
                    timestamp_us=current_time_us,
                    solve_time_ms=iter_time_ms,
                    x=result_pose.vec.x,
                    y=result_pose.vec.y,
                    yaw=result_yaw,
                    vx=result_vx,
                    ref_x=ref_point.x,
                    ref_y=ref_point.y,
                    ref_yaw=ref_point.yaw,
                )
            )

            # Update state for next iteration
            state = common_pb2.StateAtTime()
            state.timestamp_us = response.pose_local_to_rig.timestamp_us
            state.pose.CopyFrom(response.pose_local_to_rig.pose)
            state.state.CopyFrom(response.dynamic_state)

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
