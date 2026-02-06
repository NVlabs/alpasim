# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Vehicle simulation system with pluggable MPC controller.

This module provides the System class which handles:
- Trajectory management and coordinate transforms
- Vehicle model simulation
- gRPC request/response handling
- Logging

The MPC algorithm is delegated to an MPCController instance, allowing
different implementations (LinearMPC, NonlinearMPC) to be chosen at runtime.
"""

import logging

import numpy as np
from alpasim_controller.mpc_controller import (
    ControllerInput,
    MPCController,
    MPCImplementation,
)
from alpasim_controller.vehicle_model import VehicleModel
from alpasim_grpc.v0 import common_pb2, controller_pb2
from alpasim_utils.geometry import (
    Pose,
    Trajectory,
    pose_from_grpc,
    pose_to_grpc,
    trajectory_from_grpc,
)

__all__ = ["System", "VehicleModel", "create_system"]


class System:
    """Vehicle simulation system vehicle model and controller."""

    def __init__(
        self,
        log_file: str,
        initial_state_grpc: common_pb2.StateAtTime,
        controller: MPCController,
    ):
        """Initialize the System.

        Args:
            log_file: Path to CSV log file
            initial_state_grpc: Initial vehicle state from gRPC
            controller: MPCController instance to use for control computation
        """
        self._timestamp_us = initial_state_grpc.timestamp_us
        self._reference_trajectory = None
        initial_pose = pose_from_grpc(initial_state_grpc.pose)
        self._trajectory = Trajectory.from_poses(
            timestamps=np.array([initial_state_grpc.timestamp_us], dtype=np.uint64),
            poses=[initial_pose],
        )

        self._vehicle_model = VehicleModel(
            initial_velocity=np.array(
                [
                    initial_state_grpc.state.linear_velocity.x,
                    initial_state_grpc.state.linear_velocity.y,
                ]
            ),
            initial_yaw_rate=initial_state_grpc.state.angular_velocity.z,
        )
        self._controller = controller

        self._log_file_handle = open(log_file, "w", encoding="utf-8")
        self._log_header()

        # Initialize attributes that will be set during stepping
        self._first_reference_pose_rig: Pose = Pose.identity()
        self.control_input: np.ndarray = np.array([0.0, 0.0])
        self._solve_time_ms: float = 0.0

    def _dynamic_state_to_cg_velocity(
        self, dynamic_state: common_pb2.DynamicState
    ) -> np.ndarray:
        """Convert rig frame velocity to CG frame velocity."""
        return np.array(
            [
                dynamic_state.linear_velocity.x,
                dynamic_state.linear_velocity.y
                + self._vehicle_model.parameters.l_rig_to_cg
                * dynamic_state.angular_velocity.z,
            ]
        )

    def run_controller_and_vehicle_model(
        self, request: controller_pb2.RunControllerAndVehicleModelRequest
    ) -> controller_pb2.RunControllerAndVehicleModelResponse:
        """Run the controller and vehicle model for the given request.

        Args:
            request: The request containing current state, planned trajectory,
                and future time.

        Returns:
            A response containing the current pose and dynamic state in the local frame.
        """
        logging.debug(
            "run_controller_and_vehicle_model: %s: %s -> %s",
            request.session_uuid,
            request.state.timestamp_us,
            request.future_time_us,
        )

        # Input sanity checks
        if request.state.timestamp_us != self._timestamp_us:
            raise ValueError(
                f"Timestamp mismatch: expected {self._timestamp_us}, "
                f"got {request.state.timestamp_us}"
            )
        if len(request.planned_trajectory_in_rig.poses) == 0:
            raise ValueError("Planned trajectory is empty")
        if request.future_time_us <= request.state.timestamp_us:
            raise ValueError(
                f"future_time_us ({request.future_time_us}) must be greater than "
                f"current timestamp ({request.state.timestamp_us})"
            )

        # Update the pose (corrected for ground constraints) for the current timestamp
        if request.state.timestamp_us != self._trajectory.timestamps_us[-1]:
            raise ValueError(
                f"Timestamp mismatch: expected {self._trajectory.timestamps_us[-1]}, "
                f"got {request.state.timestamp_us}"
            )
        logging.debug(
            "overriding pose at timestamp %s with %s",
            request.state.timestamp_us,
            request.state.pose,
        )
        corrected_pose = pose_from_grpc(request.state.pose)
        self._trajectory.set_pose(-1, corrected_pose)

        # Store the reference trajectory
        self._reference_trajectory = trajectory_from_grpc(
            request.planned_trajectory_in_rig
        )

        # Setup and execute the MPC
        # Reinitialize the state
        if request.coerce_dynamic_state:
            velocity_cg = self._dynamic_state_to_cg_velocity(request.state.state)
            self._vehicle_model.set_velocity(velocity_cg[0], velocity_cg[1])

        # Choose the number of steps such that:
        # - we do at least one step
        # - all the steps are DT_MPC seconds long, except possibly the last one
        # - the last step can be shorter or slightly longer than DT_MPC seconds
        dt_request_us = request.future_time_us - self._timestamp_us
        dt_mpc_us = int(1e6 * self._controller.DT_MPC)
        n_steps = dt_request_us // dt_mpc_us
        if (dt_request_us % dt_mpc_us) / dt_mpc_us > 0.1:
            n_steps += 1
        n_steps = max(1, n_steps)

        for i in range(n_steps):
            if i == n_steps - 1:
                dt_us = request.future_time_us - self._timestamp_us
            else:
                dt_us = int(1e6 * self._controller.DT_MPC)
            self._step(dt_us)

        current_pose_local_to_rig = common_pb2.PoseAtTime(
            timestamp_us=self._timestamp_us,
            pose=pose_to_grpc(self._trajectory.last_pose),
        )

        # Build dynamic state with velocities and accelerations in rig frame
        dynamic_state = self._build_dynamic_state_in_rig_frame()

        return controller_pb2.RunControllerAndVehicleModelResponse(
            pose_local_to_rig=current_pose_local_to_rig,
            pose_local_to_rig_estimated=current_pose_local_to_rig,
            dynamic_state=dynamic_state,
            dynamic_state_estimated=dynamic_state,
        )

    def _build_dynamic_state_in_rig_frame(self) -> common_pb2.DynamicState:
        """
        Builds the DynamicState message with velocities and accelerations
        converted from CG frame to rig frame.

        The rig frame is offset from the CG by l_rig_to_cg along the x-axis.
        The transformation accounts for:
        - Velocity: v_rig = v_cg + omega × r_cg_to_rig
        - Acceleration: a_rig = a_cg + alpha × r_cg_to_rig + omega × (omega × r_cg_to_rig)

        Where r_cg_to_rig = [-l_rig_to_cg, 0, 0] (CG is ahead of rig origin).
        """
        l_rig_to_cg = self._vehicle_model.parameters.l_rig_to_cg
        state = self._vehicle_model.state
        accels = self._vehicle_model.accelerations

        # Extract CG-frame quantities
        v_cg_x = state[3]  # Longitudinal velocity at CG
        v_cg_y = state[4]  # Lateral velocity at CG
        yaw_rate = state[5]  # Angular velocity (yaw rate)

        a_cg_x = accels[0]  # Longitudinal acceleration at CG (= state[7])
        a_cg_y = accels[1]  # Lateral acceleration at CG
        d_yaw_rate = accels[2]  # Angular acceleration

        # Convert velocity from CG to rig frame:
        # v_rig = v_cg + omega × r_cg_to_rig
        # omega × [-l, 0, 0] = [0, 0, yaw_rate] × [-l, 0, 0] = [0, -yaw_rate * l, 0]
        v_rig_x = v_cg_x
        v_rig_y = v_cg_y - yaw_rate * l_rig_to_cg

        # Convert acceleration from CG to rig frame:
        # a_rig = a_cg + alpha × r_cg_to_rig + omega × (omega × r_cg_to_rig)
        # alpha × [-l, 0, 0] = [0, 0, d_yaw] × [-l, 0, 0] = [0, -d_yaw * l, 0]
        # omega × (omega × r) = omega × [0, -yaw_rate * l, 0]
        #                     = [0, 0, yaw_rate] × [0, -yaw_rate * l, 0]
        #                     = [yaw_rate² * l, 0, 0]
        a_rig_x = a_cg_x + yaw_rate * yaw_rate * l_rig_to_cg
        a_rig_y = a_cg_y - d_yaw_rate * l_rig_to_cg

        return common_pb2.DynamicState(
            linear_velocity=common_pb2.Vec3(x=v_rig_x, y=v_rig_y, z=0.0),
            angular_velocity=common_pb2.Vec3(x=0.0, y=0.0, z=yaw_rate),
            linear_acceleration=common_pb2.Vec3(x=a_rig_x, y=a_rig_y, z=0.0),
            angular_acceleration=common_pb2.Vec3(x=0.0, y=0.0, z=d_yaw_rate),
        )

    def _step(self, dt_us: int) -> None:
        """Execute one MPC step."""
        # Reset the integrated positional states (x, y, psi)
        # This is equivalent to resetting the vehicle model to the origin so we can
        # compute the relative pose over the propagation time
        self._vehicle_model.reset_origin()

        # Transform reference trajectory to rig frame
        ref_in_rig = self._get_reference_in_rig_frame()
        if ref_in_rig is None:
            raise ValueError("Cannot step controller: no reference trajectory set. ")

        # Build controller input
        ctrl_input = ControllerInput(
            state=self._vehicle_model.state.copy(),
            reference_trajectory=ref_in_rig,
            timestamp_us=self._timestamp_us,
        )

        # Compute control
        ctrl_output = self._controller.compute_control(ctrl_input)
        self.control_input = ctrl_output.control

        # Cache info for logging
        self._solve_time_ms = ctrl_output.solve_time_ms
        if ref_in_rig is not None and len(ref_in_rig) > 0:
            self._first_reference_pose_rig = ref_in_rig.get_pose(0)

        # Advance the vehicle model and update the trajectory history
        pose_rig_t0_to_rig_t1 = self._vehicle_model.advance(
            self.control_input, dt_us * 1e-6
        )
        self._timestamp_us += dt_us

        # Update trajectory
        logging.debug(
            "pose_rig_t0_to_rig_t1: %s, %s",
            pose_rig_t0_to_rig_t1.vec3,
            pose_rig_t0_to_rig_t1.quat,
        )
        self._trajectory.update_relative(self._timestamp_us, pose_rig_t0_to_rig_t1)
        logging.debug(
            "current pose local to rig: %s, %s",
            self._trajectory.last_pose.vec3,
            self._trajectory.last_pose.quat,
        )

        self._log()

    def _get_reference_in_rig_frame(self) -> Trajectory | None:
        """Transform reference trajectory to current rig frame.

        Returns:
            Reference trajectory transformed to rig frame, or None if no reference.
        """
        if self._reference_trajectory is None:
            return None

        # Transform from local frame to current rig frame
        pose_local_to_rig_at_ref_start = self._trajectory.interpolate_pose(
            self._reference_trajectory.timestamps_us[0]
        )
        pose_local_to_rig_now = self._trajectory.last_pose
        pose_rig_now_to_rig_at_traj_time = (
            pose_local_to_rig_now.inverse() @ pose_local_to_rig_at_ref_start
        )

        # Transform all poses
        transformed_poses = []
        for i in range(len(self._reference_trajectory)):
            pose = self._reference_trajectory.get_pose(i)
            transformed_poses.append(pose_rig_now_to_rig_at_traj_time @ pose)

        return Trajectory.from_poses(
            timestamps=self._reference_trajectory.timestamps_us.copy(),
            poses=transformed_poses,
        )

    def _log_header(self) -> None:
        """Write CSV header."""
        self._log_file_handle.write("timestamp_us,")
        self._log_file_handle.write("x,y,z,")
        self._log_file_handle.write("qx,qy,qz,qw,")
        self._log_file_handle.write("vx,vy,wz,")
        self._log_file_handle.write("u_steering_angle,")
        self._log_file_handle.write("u_longitudinal_actuation,")
        self._log_file_handle.write("ref_traj_0_x,ref_traj_0_y,")
        self._log_file_handle.write("front_steering_angle,")
        self._log_file_handle.write("acceleration,")
        self._log_file_handle.write("x_ref_0,y_ref_0,")
        self._log_file_handle.write("yaw_ref_0\n")

    def _log(self) -> None:
        """Write CSV row."""
        self._log_file_handle.write(f"{self._timestamp_us},")
        for i in range(3):
            self._log_file_handle.write(f"{self._trajectory.last_pose.vec3[i]},")
        for i in range(4):
            self._log_file_handle.write(f"{self._trajectory.last_pose.quat[i]},")
        for i in range(3):
            self._log_file_handle.write(f"{self._vehicle_model.state[i + 3]},")
        for i in range(2):
            self._log_file_handle.write(f"{self.control_input[i]},")
        if self._reference_trajectory is not None:
            first_pos = self._reference_trajectory.positions[0]
            for i in range(2):
                self._log_file_handle.write(f"{first_pos[i]},")
        else:
            self._log_file_handle.write("0.0,0.0,")
        self._log_file_handle.write(f"{self._vehicle_model.front_steering_angle},")
        self._log_file_handle.write(f"{self._vehicle_model.state[7]},")
        for i in range(2):
            self._log_file_handle.write(f"{self._first_reference_pose_rig.vec3[i]},")
        self._log_file_handle.write(f"{self._first_reference_pose_rig.yaw}\n")


def create_system(
    log_file: str,
    initial_state: common_pb2.StateAtTime,
    mpc_implementation: MPCImplementation = MPCImplementation.LINEAR,
) -> System:
    """Create a System with the specified MPC implementation.

    This is a convenience factory that creates the appropriate controller
    and passes it to System.

    Args:
        log_file: Path to CSV log file
        initial_state: Initial vehicle state from gRPC
        mpc_implementation: MPCImplementation.LINEAR (default) or MPCImplementation.NONLINEAR

    Returns:
        System instance with the specified controller

    Example:
        system = create_system("log.csv", initial_state, MPCImplementation.LINEAR)
    """
    # Create vehicle model to get parameters for controller
    vehicle_model = VehicleModel(
        initial_velocity=np.array(
            [
                initial_state.state.linear_velocity.x,
                initial_state.state.linear_velocity.y,
            ]
        ),
        initial_yaw_rate=initial_state.state.angular_velocity.z,
    )

    # Create controller based on mpc_implementation
    if mpc_implementation == "linear":
        from alpasim_controller.mpc_impl import LinearMPC

        controller = LinearMPC(vehicle_model.parameters)
    elif mpc_implementation == "nonlinear":
        from alpasim_controller.mpc_impl import NonlinearMPC

        controller = NonlinearMPC(vehicle_model.parameters)
    else:
        raise ValueError(
            f"Unknown mpc_implementation: {mpc_implementation}. "
            "Use 'linear' or 'nonlinear'."
        )

    return System(
        log_file=log_file,
        initial_state_grpc=initial_state,
        controller=controller,
    )
