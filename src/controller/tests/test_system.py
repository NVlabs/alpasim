# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import argparse
import faulthandler
import math

import pytest
from alpasim_controller.mpc_controller import MPCImplementation
from alpasim_controller.system_manager import SystemManager
from alpasim_grpc.v0 import common_pb2, controller_pb2

X_ORIGINAL = 100.0
Y_ORIGINAL = 200.0
DT = 100000  # 0.1 seconds
SESSION_UUID = "session_uuid"


def get_vx(slow: bool = False) -> float:
    return 0.75 if slow else 20.0


def run_controller_and_vehicle_model_request(
    dt_propagation_us: int = 100000,
) -> controller_pb2.RunControllerAndVehicleModelRequest:
    t0 = 123423458384  # sim init time, e.g.
    tf = t0 + dt_propagation_us

    request = controller_pb2.RunControllerAndVehicleModelRequest()

    request.session_uuid = SESSION_UUID
    request.state.pose.vec.x = X_ORIGINAL
    request.state.pose.vec.y = Y_ORIGINAL
    request.state.pose.vec.z = 0.1
    request.state.pose.quat.x = 0.0
    request.state.pose.quat.y = 0.0
    request.state.pose.quat.z = 0.0
    request.state.pose.quat.w = 1.0
    request.state.timestamp_us = t0
    request.state.state.linear_velocity.x = get_vx()

    # generate trajectory
    for i in range(51):
        request.planned_trajectory_in_rig.poses.add()
        pose_at_time = request.planned_trajectory_in_rig.poses[i]
        pose_at_time.timestamp_us = t0 + i * DT
        pose_at_time.pose.vec.x = get_vx() * i * DT / 1e6
        pose_at_time.pose.vec.y = 0.2
        pose_at_time.pose.quat.w = 1.0

    request.future_time_us = tf
    return request


@pytest.mark.parametrize("dt_propagation_us", [100000, 500000])
def test_alpasimvdc_one_step(dt_propagation_us) -> None:
    "Run a single step of the controller and vehicle model simulation."
    system_manager = SystemManager(".")

    # Must start session before running controller
    system_manager.start_session(SESSION_UUID)

    response = system_manager.run_controller_and_vehicle_model(
        run_controller_and_vehicle_model_request(dt_propagation_us)
    )

    scale = dt_propagation_us / 100000.0
    TOLERANCE_GT = 1e-2 * scale  # note, noise is present => non-zero control error
    TOLERANCE_EST = 1e-1 * scale  # higher noise for estimation

    # sanity check that the integration is approximately working
    assert response.HasField("pose_local_to_rig")
    assert response.pose_local_to_rig.pose.vec.x == pytest.approx(
        X_ORIGINAL + get_vx() * dt_propagation_us / 1e6, abs=TOLERANCE_GT
    )
    assert response.pose_local_to_rig.pose.vec.y == pytest.approx(
        Y_ORIGINAL, abs=TOLERANCE_GT
    )

    assert response.pose_local_to_rig_estimated.pose.vec.x == pytest.approx(
        X_ORIGINAL + get_vx() * dt_propagation_us / 1e6, abs=TOLERANCE_EST
    )
    assert response.pose_local_to_rig_estimated.pose.vec.y == pytest.approx(
        Y_ORIGINAL, abs=TOLERANCE_EST
    )

    # check that we can remove the actor/close the session
    close_session_request = controller_pb2.VDCSessionCloseRequest(
        session_uuid=SESSION_UUID
    )
    response = system_manager.close_session(close_session_request)
    assert response == common_pb2.Empty()

    # closing the same session again should fail
    with pytest.raises(KeyError):
        system_manager.close_session(close_session_request)


def test_session_lifecycle() -> None:
    """Test the full session lifecycle including edge cases."""
    system_manager = SystemManager(".")

    # Can't run controller without starting session
    with pytest.raises(KeyError):
        system_manager.run_controller_and_vehicle_model(
            run_controller_and_vehicle_model_request()
        )

    # Can't close a session that doesn't exist
    with pytest.raises(KeyError):
        system_manager.close_session(
            controller_pb2.VDCSessionCloseRequest(session_uuid="nonexistent")
        )

    # Start session, then close without running controller (early cleanup case)
    system_manager.start_session(SESSION_UUID)
    close_request = controller_pb2.VDCSessionCloseRequest(session_uuid=SESSION_UUID)
    response = system_manager.close_session(close_request)
    assert response == common_pb2.Empty()

    # Session is now closed, can't close again
    with pytest.raises(KeyError):
        system_manager.close_session(close_request)


def generate_run_controller_request(
    timestamp: int,
    previous_state: common_pb2.StateAtTime,
    slow: bool,
) -> controller_pb2.RunControllerAndVehicleModelRequest:
    # Note: assumes planar motion with constant velocity
    tf = timestamp + 100000  # propagate for 0.1 seconds

    request = controller_pb2.RunControllerAndVehicleModelRequest()

    request.session_uuid = SESSION_UUID
    request.state.CopyFrom(previous_state)

    yaw = 2.0 * previous_state.pose.quat.z

    # generate trajectory
    for i in range(51):
        request.planned_trajectory_in_rig.poses.add()
        pose_at_time = request.planned_trajectory_in_rig.poses[i]

        pose_at_time.timestamp_us = timestamp + i * DT

        desired_position_local = [get_vx(slow) * pose_at_time.timestamp_us / 1.0e6, 0.0]
        relative_position_local = [
            desired_position_local[0] - previous_state.pose.vec.x,
            desired_position_local[1] - previous_state.pose.vec.y,
        ]
        desired_position_rig = [
            math.cos(yaw) * relative_position_local[0]
            + math.sin(yaw) * relative_position_local[1],
            -math.sin(yaw) * relative_position_local[0]
            + math.cos(yaw) * relative_position_local[1],
        ]

        pose_at_time.pose.vec.x = desired_position_rig[0]
        pose_at_time.pose.vec.y = desired_position_rig[1]
        pose_at_time.pose.quat.z = -previous_state.pose.quat.z
        pose_at_time.pose.quat.w = math.sqrt(1.0 - pose_at_time.pose.quat.z**2)

    request.future_time_us = tf
    return request


@pytest.mark.parametrize(
    "slow,mpc_impl",
    [
        (False, MPCImplementation.LINEAR),
        (True, MPCImplementation.LINEAR),
        (False, MPCImplementation.NONLINEAR),
        (True, MPCImplementation.NONLINEAR),
    ],
)
def test_mini_sim(slow: bool, mpc_impl: MPCImplementation) -> None:
    "Run multiple steps of the controller and vehicle model simulation."
    run_mini_sim(slow, mpc_impl)


def run_mini_sim(slow: bool, mpc_impl: MPCImplementation) -> None:
    """
    Simulate multiple steps of the simulation, with a constant velocity trajectory.
    """
    system_manager = SystemManager(".", mpc_implementation=mpc_impl)
    system_manager.start_session(SESSION_UUID)

    timestamp = 0
    state = common_pb2.StateAtTime()
    if mpc_impl == MPCImplementation.NONLINEAR:
        state.pose.vec.y = 0.6  # small y offset
    elif mpc_impl == MPCImplementation.LINEAR:
        # TODO(mwatson): Linear MPC requires some gain scheduling for low speed tight maneuvers
        state.pose.vec.y = 0.15  # small y offset
    state.pose.quat.w = 1.0
    state.state.linear_velocity.x = get_vx(slow)  # Initialize to reference velocity

    IDX_KICK = 1

    for i in range(100):  # 10 seconds
        run_controller_and_vehicle_model_request = generate_run_controller_request(
            timestamp, state, slow
        )
        if i == IDX_KICK:
            # provide a kick to the system to:
            # 1. prove that coerce_dynamic_state is working
            # 2. prove that the closed-loop control is working
            KICK_VELOCITY = 0.9 * get_vx(slow)
            run_controller_and_vehicle_model_request.state.state.linear_velocity.x = (
                KICK_VELOCITY
            )
            run_controller_and_vehicle_model_request.coerce_dynamic_state = True
        response = system_manager.run_controller_and_vehicle_model(
            run_controller_and_vehicle_model_request
        )

        timestamp = response.pose_local_to_rig.timestamp_us

        state = common_pb2.StateAtTime()
        state.timestamp_us = response.pose_local_to_rig.timestamp_us
        state.pose.CopyFrom(response.pose_local_to_rig.pose)
        if i == IDX_KICK:
            # Note: the API doesn't return the velocity, so we have to check the
            # displacement
            dt_us = (
                response.pose_local_to_rig.timestamp_us
                - run_controller_and_vehicle_model_request.state.timestamp_us
            )
            dx = (
                response.pose_local_to_rig.pose.vec.x
                - run_controller_and_vehicle_model_request.state.pose.vec.x
            )
            approx_velocity = dx / (dt_us * 1e-6)
            assert approx_velocity == pytest.approx(KICK_VELOCITY, abs=0.1)

    # sanity check that closed loop control is working
    # Note: somewhat relaxed longitudinal position tracking due to gains/competing goals
    #       and the fact that acceleration commands are not provided
    assert state.pose.vec.x == pytest.approx(get_vx(slow) * timestamp / 1e6, abs=2.0)
    assert state.pose.vec.y == pytest.approx(0.0, abs=0.1)

    # sanity check noise effects
    gt_position_in_local = response.pose_local_to_rig.pose.vec
    estimated_position_in_local = response.pose_local_to_rig_estimated.pose.vec
    assert gt_position_in_local.x == pytest.approx(
        estimated_position_in_local.x, abs=1.0e-5
    )
    assert gt_position_in_local.y == pytest.approx(
        estimated_position_in_local.y, abs=1.0e-5
    )


if __name__ == "__main__":
    faulthandler.enable()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--slow", action="store_true", help="Run the slow mini simulation test"
    )
    arg_parser.add_argument(
        "--mpc-implementation",
        type=MPCImplementation,
        choices=list(MPCImplementation),
        default=MPCImplementation.LINEAR,
        help="MPC implementation: linear or nonlinear",
    )
    args = arg_parser.parse_args()
    run_mini_sim(args.slow, args.mpc_implementation)
