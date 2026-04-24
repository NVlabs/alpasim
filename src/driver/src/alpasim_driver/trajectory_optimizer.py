# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation
# Copyright (c) 2025 Brayden Zhang

"""
Trajectory Optimization for VAM Driver.

Post-processes predicted trajectories using classical optimization for
smoothness and comfort constraint enforcement.

Code adapted from: https://drive.google.com/file/d/1u-Hmpc304HySIXZwrptJnPQTT3fbC1Jz/view

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of trajectory optimization."""

    trajectory: np.ndarray  # Optimized trajectory [N, 3] (x, y, heading)
    success: bool
    iterations: int | None = None
    final_cost: float | None = None
    message: str = ""


@dataclass
class VehicleConstraints:
    """Vehicle kinematic and comfort constraints."""

    max_deviation: float = 2.0  # Maximum deviation from original trajectory (meters)
    max_heading_change: float = 0.5236  # ~30 degrees in radians
    max_speed: float = 15.0  # m/s
    max_accel: float = 5.0  # m/s²

    # Comfort limits (from PDMS spec)
    max_abs_yaw_rate: float = 0.95  # rad/s
    max_abs_yaw_acc: float = 1.93  # rad/s²
    max_lon_acc_pos: float = 4.89  # m/s²
    max_lon_acc_neg: float = -4.05  # m/s²
    max_abs_lon_jerk: float = 8.37  # m/s³


class TrajectoryOptimizer:
    """
    Trajectory optimizer for smoothness and comfort.

    Refines VAM-predicted trajectories using classical optimization with
    cost functions for trajectory smoothness and comfort constraint enforcement.
    """

    def __init__(
        self,
        smoothness_weight: float = 1.0,
        deviation_weight: float = 0.1,
        comfort_weight: float = 2.0,
        max_iterations: int = 100,
        # Frenet/retiming options
        enable_frenet_retiming: bool = True,
        retime_alpha: float = 0.25,
    ):
        """
        Initialize the trajectory optimizer.

        Args:
            smoothness_weight: Weight for trajectory smoothness term (curvature penalty)
            deviation_weight: Weight for deviation from original trajectory
            comfort_weight: Weight for comfort constraint soft penalty
            max_iterations: Maximum optimization iterations
            enable_frenet_retiming: Whether to redistribute waypoints along path
            retime_alpha: Retiming strength in [0,1]; higher = more front-loaded distance
        """
        self.smoothness_weight = smoothness_weight
        self.deviation_weight = deviation_weight
        self.comfort_weight = comfort_weight
        self.max_iterations = max_iterations
        self.enable_frenet_retiming = bool(enable_frenet_retiming)
        self.retime_alpha = float(np.clip(retime_alpha, 0.0, 1.0))

    def optimize(
        self,
        trajectory: np.ndarray,
        time_step: float = 0.5,
        vehicle_constraints: VehicleConstraints | None = None,
        retime_in_frenet: bool | None = None,
        retime_alpha: float | None = None,
    ) -> OptimizationResult:
        """
        Optimize trajectory for smoothness and comfort.

        Args:
            trajectory: Original trajectory [N, 3] (x, y, heading) in rig/ego frame
            time_step: Time between waypoints in seconds
            vehicle_constraints: Vehicle kinematic and comfort limits
            retime_in_frenet: Override for enabling Frenet retiming
            retime_alpha: Override for retiming strength

        Returns:
            OptimizationResult with optimized trajectory and metadata
        """
        if trajectory is None or len(trajectory) < 2:
            return OptimizationResult(
                trajectory=trajectory if trajectory is not None else np.array([]),
                success=False,
                message="Trajectory too short to optimize",
            )

        # Ensure numpy array
        if hasattr(trajectory, "detach"):  # torch.Tensor
            trajectory = trajectory.detach().cpu().numpy()
        trajectory = np.asarray(trajectory, dtype=np.float64)

        if trajectory.shape[1] < 3:
            return OptimizationResult(
                trajectory=trajectory,
                success=False,
                message="Trajectory must have shape [N, 3] (x, y, heading)",
            )

        original_trajectory = trajectory.copy()
        constraints = vehicle_constraints or VehicleConstraints()

        # Determine retiming settings
        use_retiming = (
            self.enable_frenet_retiming
            if retime_in_frenet is None
            else bool(retime_in_frenet)
        )
        alpha = (
            self.retime_alpha
            if retime_alpha is None
            else float(np.clip(retime_alpha, 0.0, 1.0))
        )

        # Apply Frenet-style retiming as initial guess
        if use_retiming:
            try:
                initial_guess = self._retime_along_path(
                    original_trajectory, alpha=alpha
                )
            except Exception as e:
                logger.debug("Retiming failed, using original: %s", e)
                initial_guess = original_trajectory
        else:
            initial_guess = original_trajectory

        # Create cost function
        cost_fn = self._create_cost_function(
            original_trajectory, time_step, constraints
        )

        # Create bounds
        bounds = self._create_bounds(original_trajectory, constraints)

        # Create constraints (fixed endpoints + hard comfort limits)
        opt_constraints = self._create_constraints(
            original_trajectory, time_step, constraints, enforce_comfort=True
        )

        # Run optimization
        x0 = initial_guess.flatten()

        try:
            # Try L-BFGS-B first (stable, no constraint support but respects bounds)
            result = minimize(
                cost_fn,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iterations, "ftol": 1e-6},
            )

            if not result.success:
                # Fallback to SLSQP with constraints
                result = minimize(
                    cost_fn,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=opt_constraints,
                    options={"maxiter": self.max_iterations, "ftol": 1e-6},
                )

            if result.success:
                optimized = result.x.reshape(-1, 3)
                return OptimizationResult(
                    trajectory=optimized,
                    success=True,
                    iterations=getattr(result, "nit", None),
                    final_cost=float(result.fun),
                    message=str(result.message),
                )
            else:
                logger.warning("Optimization did not converge: %s", result.message)
                return OptimizationResult(
                    trajectory=original_trajectory,
                    success=False,
                    message=f"Optimization failed: {result.message}",
                )

        except Exception as e:
            logger.warning("Optimization error: %s", e)
            return OptimizationResult(
                trajectory=original_trajectory,
                success=False,
                message=f"Optimization error: {e}",
            )

    def _create_cost_function(
        self,
        original_trajectory: np.ndarray,
        time_step: float,
        constraints: VehicleConstraints,
    ):
        """Create the cost function for optimization."""

        def cost_function(x: np.ndarray) -> float:
            traj = x.reshape(-1, 3)

            # Smoothness cost (second-order differences)
            smoothness_cost = self._compute_smoothness_cost(traj)

            # Deviation from original
            deviation_cost = float(np.sum((traj - original_trajectory) ** 2))

            # Comfort soft penalty
            comfort_penalty = self._compute_comfort_penalty(
                traj, time_step, constraints
            )

            total = (
                self.smoothness_weight * smoothness_cost
                + self.deviation_weight * deviation_cost
                + self.comfort_weight * comfort_penalty
            )
            return total

        return cost_function

    def _compute_smoothness_cost(self, trajectory: np.ndarray) -> float:
        """Compute trajectory smoothness using second-order differences."""
        if trajectory.shape[0] < 3:
            return 0.0

        # Position second differences (curvature proxy)
        p = trajectory[:, :2]
        ddp = p[2:] - 2 * p[1:-1] + p[:-2]
        pos_smooth = float(np.sum(np.linalg.norm(ddp, axis=1) ** 2))

        # Heading second difference with angle wrap
        yaw = trajectory[:, 2]
        dyaw1 = np.arctan2(np.sin(yaw[1:] - yaw[:-1]), np.cos(yaw[1:] - yaw[:-1]))
        if len(dyaw1) < 2:
            return pos_smooth
        dyaw2 = np.arctan2(
            np.sin(dyaw1[1:] - dyaw1[:-1]), np.cos(dyaw1[1:] - dyaw1[:-1])
        )
        yaw_smooth = float(np.sum(dyaw2**2))

        return pos_smooth + yaw_smooth

    def _compute_comfort_penalty(
        self,
        trajectory: np.ndarray,
        dt: float,
        constraints: VehicleConstraints,
    ) -> float:
        """Compute soft penalty for exceeding comfort limits."""
        if trajectory.shape[0] < 4:
            return 0.0

        yaw_rate, yaw_acc, lon_a, lon_j = self._compute_kinematics(trajectory, dt)

        def over_abs(arr: np.ndarray, limit: float) -> np.ndarray:
            return np.maximum(0.0, np.abs(arr) - limit)

        def over_upper(arr: np.ndarray, limit: float) -> np.ndarray:
            return np.maximum(0.0, arr - limit)

        def over_lower(arr: np.ndarray, limit: float) -> np.ndarray:
            return np.maximum(0.0, limit - arr)

        p_yaw_rate = over_abs(yaw_rate, constraints.max_abs_yaw_rate)
        p_yaw_acc = over_abs(yaw_acc, constraints.max_abs_yaw_acc)
        p_lon_a = np.sqrt(
            over_upper(lon_a, constraints.max_lon_acc_pos) ** 2
            + over_lower(lon_a, constraints.max_lon_acc_neg) ** 2
        )
        p_lon_j = over_abs(lon_j, constraints.max_abs_lon_jerk)

        penalty = (
            np.mean(p_yaw_rate**2)
            + np.mean(p_yaw_acc**2)
            + 1.5 * np.mean(p_lon_a**2)
            + 1.5 * np.mean(p_lon_j**2)
        )
        return float(penalty)

    def _compute_kinematics(
        self, trajectory: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute kinematic quantities from trajectory."""
        dt = max(float(dt), 1e-3)
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        yaw = trajectory[:, 2]

        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)

        # Longitudinal velocity along heading
        lon_v = vx * np.cos(yaw) + vy * np.sin(yaw)
        lon_a = np.gradient(lon_v, dt)
        lon_j = np.gradient(lon_a, dt)

        yaw_rate = np.gradient(yaw, dt)
        yaw_acc = np.gradient(yaw_rate, dt)

        return yaw_rate, yaw_acc, lon_a, lon_j

    def _create_bounds(
        self,
        original_trajectory: np.ndarray,
        constraints: VehicleConstraints,
    ) -> List[Tuple[float, float]]:
        """Create optimization bounds."""
        bounds = []
        for i in range(len(original_trajectory)):
            bounds.extend(
                [
                    (
                        original_trajectory[i, 0] - constraints.max_deviation,
                        original_trajectory[i, 0] + constraints.max_deviation,
                    ),
                    (
                        original_trajectory[i, 1] - constraints.max_deviation,
                        original_trajectory[i, 1] + constraints.max_deviation,
                    ),
                    (
                        original_trajectory[i, 2] - constraints.max_heading_change,
                        original_trajectory[i, 2] + constraints.max_heading_change,
                    ),
                ]
            )
        return bounds

    def _create_constraints(
        self,
        original_trajectory: np.ndarray,
        time_step: float,
        constraints: VehicleConstraints,
        enforce_comfort: bool = True,
    ) -> List[Dict[str, Any]]:
        """Create optimization constraints (fixed endpoints and optional comfort limits)."""
        opt_constraints = []

        # Keep start point fixed
        opt_constraints.append(
            {"type": "eq", "fun": lambda x: x[:3] - original_trajectory[0]}
        )

        # Keep end point fixed
        opt_constraints.append(
            {"type": "eq", "fun": lambda x: x[-3:] - original_trajectory[-1]}
        )

        # Hard comfort constraints: ensure 'c' metric stays 1.0
        if enforce_comfort:

            def comfort_margins(z: np.ndarray) -> np.ndarray:
                traj = z.reshape(-1, 3)
                yaw_rate, yaw_acc, lon_a, lon_j = self._compute_kinematics(
                    traj, time_step
                )
                margins = [
                    # yaw rate: |.| <= max_abs_yaw_rate
                    constraints.max_abs_yaw_rate - np.max(np.abs(yaw_rate)),
                    # yaw accel: |.| <= max_abs_yaw_acc
                    constraints.max_abs_yaw_acc - np.max(np.abs(yaw_acc)),
                    # lon accel upper: <= max_lon_acc_pos
                    constraints.max_lon_acc_pos - np.max(lon_a),
                    # lon accel lower: >= max_lon_acc_neg
                    np.min(lon_a) - constraints.max_lon_acc_neg,
                    # lon jerk: |.| <= max_abs_lon_jerk
                    constraints.max_abs_lon_jerk - np.max(np.abs(lon_j)),
                ]
                return np.array(margins, dtype=np.float64)

            opt_constraints.append({"type": "ineq", "fun": comfort_margins})

        return opt_constraints

    # -------------------------
    # Frenet-style retiming
    # -------------------------

    def _polyline_arclen(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cumulative arc-length along a 2D polyline."""
        if pts.shape[0] < 2:
            return np.zeros((pts.shape[0],), dtype=np.float64), np.zeros(
                (0,), dtype=np.float64
            )
        diffs = pts[1:] - pts[:-1]
        seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
        s = np.zeros((pts.shape[0],), dtype=np.float64)
        s[1:] = np.cumsum(seg_len)
        return s, seg_len

    def _sample_polyline_by_s(
        self, pts: np.ndarray, s: np.ndarray, s_query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a 2D polyline at query arc-lengths."""
        N = pts.shape[0]
        if N == 0:
            return np.zeros((0, 2)), np.zeros((0,))

        total_s = s[-1] if len(s) > 0 else 0.0
        if total_s <= 1e-6:
            xy = np.repeat(pts[:1], repeats=len(s_query), axis=0)
            yaw = np.zeros((len(s_query),), dtype=np.float64)
            if N >= 2:
                d = pts[1] - pts[0]
                if np.hypot(d[0], d[1]) > 1e-9:
                    yaw[:] = np.arctan2(d[1], d[0])
            return xy, yaw

        xy = np.zeros((len(s_query), 2), dtype=np.float64)
        yaw = np.zeros((len(s_query),), dtype=np.float64)
        i = 0

        for k, sq in enumerate(s_query):
            sq = float(np.clip(sq, 0.0, total_s))
            while i < N - 2 and s[i + 1] < sq:
                i += 1

            ds = s[i + 1] - s[i]
            if ds <= 1e-12:
                j = i
                while j < N - 2 and (s[j + 1] - s[j]) <= 1e-12:
                    j += 1
                dvec = pts[j + 1] - pts[j] if j < N - 1 else pts[i + 1] - pts[i]
                xy[k] = pts[i]
                yaw[k] = (
                    np.arctan2(dvec[1], dvec[0])
                    if np.hypot(dvec[0], dvec[1]) > 1e-12
                    else (yaw[k - 1] if k > 0 else 0.0)
                )
            else:
                u = (sq - s[i]) / ds
                p0, p1 = pts[i], pts[i + 1]
                dvec = p1 - p0
                xy[k] = p0 + u * dvec
                yaw[k] = np.arctan2(dvec[1], dvec[0])

        return xy, yaw

    def _retime_along_path(
        self, trajectory: np.ndarray, alpha: float = 0.25
    ) -> np.ndarray:
        """
        Redistribute waypoints along path (Frenet-style) keeping endpoints fixed.

        Args:
            trajectory: [N, 3] array of (x, y, yaw)
            alpha: Retiming strength in [0, 1]; higher = more front-loaded distance

        Returns:
            Retimed trajectory [N, 3] with same endpoints, adjusted spacing.
        """
        if trajectory is None or trajectory.shape[0] < 2:
            return trajectory

        pts = np.ascontiguousarray(trajectory[:, :2], dtype=np.float64)
        s, _ = self._polyline_arclen(pts)
        total = s[-1] if len(s) > 0 else 0.0

        if total <= 1e-6:
            return trajectory

        N = trajectory.shape[0]
        t = np.linspace(0.0, 1.0, N, dtype=np.float64)

        # Ease-out warp: w(t) = 1 - (1-t)^(1 + 4*alpha)
        beta = 1.0 + 4.0 * float(np.clip(alpha, 0.0, 1.0))
        w = 1.0 - np.power(1.0 - t, beta)
        s_new = total * w

        xy_new, yaw_new = self._sample_polyline_by_s(pts, s, s_new)

        out = trajectory.copy()
        out[:, 0:2] = xy_new
        out[:, 2] = yaw_new

        # Ensure exact start/end equality
        out[0] = trajectory[0]
        out[-1] = trajectory[-1]

        return out


def add_heading_to_trajectory(xy_trajectory: np.ndarray) -> np.ndarray:
    """
    Add heading column to an [N, 2] trajectory.

    Computes heading from consecutive position deltas.

    Args:
        xy_trajectory: [N, 2] array of (x, y) positions

    Returns:
        [N, 3] array of (x, y, heading)
    """
    if xy_trajectory is None or len(xy_trajectory) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    xy = np.asarray(xy_trajectory, dtype=np.float64)
    if xy.ndim == 1:
        xy = xy.reshape(1, -1)

    N = xy.shape[0]
    headings = np.zeros(N, dtype=np.float64)

    prev_pos = np.array([0.0, 0.0])  # Start at ego origin
    for i in range(N):
        delta = xy[i] - prev_pos
        dist = np.hypot(delta[0], delta[1])
        if dist > 1e-4:
            headings[i] = np.arctan2(delta[1], delta[0])
        else:
            headings[i] = headings[i - 1] if i > 0 else 0.0
        prev_pos = xy[i]

    return np.column_stack([xy, headings])
