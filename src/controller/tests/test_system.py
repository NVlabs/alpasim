# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import argparse
import faulthandler
import math

import numpy as np
import pytest
from alpasim_controller.mpc_controller import (
    ControllerConfig,
    ControllerOutput,
    MPCImplementation,
)
from alpasim_controller.mpc_impl import SQPMPC, LinearMPC, NonlinearMPC
from alpasim_controller.system import System, create_system
from alpasim_controller.system_manager import SystemManager
from alpasim_grpc.v0 import common_pb2, controller_pb2
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_utils.geometry import pose_from_grpc
from alpasim_utils.logs import LogWriter, async_read_pb_log

X_ORIGINAL = 100.0
Y_ORIGINAL = 200.0
DT = 100000  # 0.1 seconds
SESSION_UUID = "session_uuid"


@pytest.mark.parametrize(
    "statuses,failed_count",
    [
        (["solved"], 0),
        (["failed: primal infeasible"], 1),
        (["failed: primal infeasible", "solved inaccurate", "solved"], 1),
    ],
)
def test_response_counts_each_mpc_update(tmp_path, monkeypatch, statuses, failed_count):
    request = run_controller_and_vehicle_model_request(len(statuses) * DT)
    request.pose_reporting_interval_us = DT // 10
    controller = LinearMPC()
    outcomes = iter(statuses)

    def compute(_):
        status = next(outcomes)
        return ControllerOutput(
            None if status.startswith("failed:") else np.zeros(2), 0.0, status
        )

    monkeypatch.setattr(controller, "compute_control", compute)
    system = System(str(tmp_path / "controller.csv"), request.state, controller)
    response = system.run_controller_and_vehicle_model(request)
    assert response.HasField("solver_stats")
    assert response.solver_stats.solve_count == len(statuses)
    assert response.solver_stats.failed_solve_count == failed_count


def test_held_command_reports_zero_solves_after_failed_request(tmp_path, monkeypatch):
    request = run_controller_and_vehicle_model_request(DT // 2)
    controller = LinearMPC()
    monkeypatch.setattr(
        controller,
        "compute_control",
        lambda _: ControllerOutput(None, 0.0, "failed: primal infeasible"),
    )
    system = System(str(tmp_path / "controller.csv"), request.state, controller)
    first = system.run_controller_and_vehicle_model(request)
    assert first.solver_stats.solve_count == first.solver_stats.failed_solve_count == 1
    state = first.states[-1]
    request.state.timestamp_us = state.timestamp_us
    request.state.pose.CopyFrom(state.pose_local_to_rig)
    request.state.state.CopyFrom(state.dynamic_state)
    request.future_time_us = state.timestamp_us + DT // 10
    response = system.run_controller_and_vehicle_model(request)
    assert response.HasField("solver_stats")
    assert (
        response.solver_stats.solve_count
        == response.solver_stats.failed_solve_count
        == 0
    )


@pytest.mark.parametrize("backend", [LinearMPC, NonlinearMPC, SQPMPC])
@pytest.mark.parametrize("coerce", [False, True])
def test_failed_solves_retain_active_command_after_optional_coercion(
    tmp_path, monkeypatch, backend, coerce
):
    request = run_controller_and_vehicle_model_request(DT)
    controller = backend(n_horizon=5)
    system = System(str(tmp_path / "controller.csv"), request.state, controller)
    state = system.run_controller_and_vehicle_model(request).states[-1]
    request.state.timestamp_us = state.timestamp_us
    request.state.pose.CopyFrom(state.pose_local_to_rig)
    request.state.state.CopyFrom(state.dynamic_state)
    request.future_time_us = state.timestamp_us + 2 * DT
    request.coerce_dynamic_state = coerce
    system.control_input = np.array([0.12, 1.0])
    expected = system.current_control_input
    if coerce:
        request.state.state.linear_velocity.x = 10.0
        request.state.state.angular_velocity.z = 0.4
        request.state.state.linear_acceleration.x = 2.0
        expected = np.array(
            [
                math.atan(
                    request.state.state.angular_velocity.z
                    / request.state.state.linear_velocity.x
                    * system._vehicle_model.parameters.wheelbase
                ),
                2.0,
            ]
        )
    resets = []
    native_reset = controller.reset_warm_start

    def reset():
        resets.append(True)
        native_reset()

    monkeypatch.setattr(controller, "reset_warm_start", reset)
    monkeypatch.setattr(
        controller, "_compute_control", lambda _: (None, "native failure")
    )
    response = system.run_controller_and_vehicle_model(request)
    np.testing.assert_array_equal(system.current_control_input, expected)
    assert system.last_controller_status == "failed: native failure"
    assert (
        response.solver_stats.solve_count
        == response.solver_stats.failed_solve_count
        == 2
    )
    assert len(resets) == 2 + coerce


@pytest.mark.parametrize("counts", [None, (0, 0), (3, 1)])
async def test_asl_preserves_solver_stats_presence_and_counts(tmp_path, counts):
    response = controller_pb2.RunControllerAndVehicleModelResponse()
    if counts is not None:
        response.solver_stats.CopyFrom(
            controller_pb2.RunControllerAndVehicleModelResponse.SolverStats(
                solve_count=counts[0], failed_solve_count=counts[1]
            )
        )
    asl_path = str(tmp_path / "controller.asl")
    async with LogWriter(asl_path) as writer:
        await writer.on_message(LogEntry(controller_return=response))
    entries = [entry async for entry in async_read_pb_log(asl_path)]
    assert len(entries) == 1
    logged_response = entries[0].controller_return
    assert logged_response.HasField("solver_stats") == (counts is not None)
    assert logged_response == response


@pytest.mark.parametrize("mpc_impl", list(MPCImplementation))
def test_held_plan_keeps_first_waypoint_frame_between_requests(tmp_path, mpc_impl):
    """Reusing a plan preserves its world targets as the ego advances."""
    config = ControllerConfig(mpc_implementation=mpc_impl)
    request = run_controller_and_vehicle_model_request(dt_propagation_us=500000)
    request.pose_reporting_interval_us = 100000
    bulk = create_system(str(tmp_path / "bulk.csv"), request.state, config)
    expected = bulk.run_controller_and_vehicle_model(request)

    ticks = create_system(str(tmp_path / "ticks.csv"), request.state, config)
    for state_expected in expected.states:
        request.future_time_us = request.state.timestamp_us + 100000
        observed = ticks.run_controller_and_vehicle_model(request).states[-1]
        np.testing.assert_allclose(
            pose_from_grpc(observed.pose_local_to_rig).vec3,
            pose_from_grpc(state_expected.pose_local_to_rig).vec3,
            rtol=0,
            atol=1e-5,
        )
        assert observed.dynamic_state.angular_velocity.z == pytest.approx(
            state_expected.dynamic_state.angular_velocity.z, abs=1e-4
        )
        request.state.timestamp_us = observed.timestamp_us
        request.state.pose.CopyFrom(observed.pose_local_to_rig)
        request.state.state.CopyFrom(observed.dynamic_state)


@pytest.mark.parametrize("mpc_impl", list(MPCImplementation))
def test_coercion_discards_previous_optimizer_state(tmp_path, mpc_impl) -> None:
    """After coercion, a used optimizer behaves exactly like a fresh optimizer."""
    config = ControllerConfig(mpc_implementation=mpc_impl)
    stop = run_controller_and_vehicle_model_request()
    del stop.planned_trajectory_in_rig.poses[2:]
    used = create_system(str(tmp_path / "used.csv"), stop.state, config)
    first = used.run_controller_and_vehicle_model(stop).states[-1]
    request = run_controller_and_vehicle_model_request()
    delta = first.timestamp_us - request.state.timestamp_us
    request.state.timestamp_us += delta
    request.future_time_us += delta
    for sample in request.planned_trajectory_in_rig.poses:
        sample.timestamp_us += delta
    request.coerce_dynamic_state = True
    observed = used.run_controller_and_vehicle_model(request)

    fresh = create_system(str(tmp_path / "fresh.csv"), request.state, config)
    expected = fresh.run_controller_and_vehicle_model(request)
    np.testing.assert_array_equal(
        used.current_control_input, fresh.current_control_input
    )
    assert (
        observed.states[-1].SerializeToString()
        == expected.states[-1].SerializeToString()
    )


@pytest.mark.parametrize("controller_type", [LinearMPC, NonlinearMPC, SQPMPC])
def test_coercion_replaces_held_commands_between_mpc_ticks(
    tmp_path, monkeypatch, controller_type
) -> None:
    """No pre-coercion steering or brake command survives until the next solve."""
    request = run_controller_and_vehicle_model_request(dt_propagation_us=50000)
    controller = controller_type()
    monkeypatch.setattr(
        controller,
        "compute_control",
        lambda _: ControllerOutput(np.array([0.5, -9.0]), 0.0, "solved"),
    )
    system = System(str(tmp_path / "controller.csv"), request.state, controller)
    first = system.run_controller_and_vehicle_model(request).states[-1]
    request.state.timestamp_us = first.timestamp_us
    request.state.pose.CopyFrom(first.pose_local_to_rig)
    request.state.state.linear_velocity.x = get_vx()
    request.state.state.linear_acceleration.x = 2.0
    request.future_time_us = first.timestamp_us + 10000
    request.coerce_dynamic_state = True
    response = system.run_controller_and_vehicle_model(request)
    np.testing.assert_array_equal(system.current_control_input, [0.0, 2.0])
    assert response.states[-1].dynamic_state.linear_acceleration.x == pytest.approx(2.0)


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
    """Run a single step of the controller and vehicle model simulation."""
    system_manager = SystemManager(".", controller_config=ControllerConfig())

    # Must start session before running controller
    system_manager.start_session(SESSION_UUID)

    response = system_manager.run_controller_and_vehicle_model(
        run_controller_and_vehicle_model_request(dt_propagation_us)
    )

    scale = dt_propagation_us / 100000.0
    TOLERANCE_GT = 1e-2 * scale  # note, noise is present => non-zero control error
    TOLERANCE_EST = 1e-1 * scale  # higher noise for estimation

    # sanity check that the integration is approximately working
    assert len(response.states) > 0
    final = response.states[-1]
    assert final.pose_local_to_rig.vec.x == pytest.approx(
        X_ORIGINAL + get_vx() * dt_propagation_us / 1e6, abs=TOLERANCE_GT
    )
    assert final.pose_local_to_rig.vec.y == pytest.approx(Y_ORIGINAL, abs=TOLERANCE_GT)

    assert final.pose_local_to_rig_estimated.vec.x == pytest.approx(
        X_ORIGINAL + get_vx() * dt_propagation_us / 1e6, abs=TOLERANCE_EST
    )
    assert final.pose_local_to_rig_estimated.vec.y == pytest.approx(
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


def test_coerce_dynamic_state_preserves_yaw_rate(tmp_path) -> None:
    """Coercing dynamic state initializes the model from ground-truth yaw rate."""
    system_manager = SystemManager(str(tmp_path), controller_config=ControllerConfig())
    system_manager.start_session(SESSION_UUID)
    request = run_controller_and_vehicle_model_request(dt_propagation_us=1)
    request.state.state.angular_velocity.z = 0.2
    request.coerce_dynamic_state = True

    response = system_manager.run_controller_and_vehicle_model(request)

    assert response.states[-1].dynamic_state.angular_velocity.z == pytest.approx(0.2)


def test_session_lifecycle() -> None:
    """Test the full session lifecycle including edge cases."""
    system_manager = SystemManager(".", controller_config=ControllerConfig())

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
    """Run multiple steps of the controller and vehicle model simulation."""
    run_mini_sim(slow, mpc_impl)


def run_mini_sim(slow: bool, mpc_impl: MPCImplementation) -> None:
    """Simulate multiple steps of the simulation, with a constant velocity trajectory."""
    system_manager = SystemManager(
        ".", controller_config=ControllerConfig(mpc_implementation=str(mpc_impl))
    )
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

        final = response.states[-1]
        timestamp = final.timestamp_us

        state = common_pb2.StateAtTime()
        state.timestamp_us = final.timestamp_us
        state.pose.CopyFrom(final.pose_local_to_rig)
        if i == IDX_KICK:
            # Note: the API doesn't return the velocity, so we have to check the
            # displacement
            dt_us = (
                final.timestamp_us
                - run_controller_and_vehicle_model_request.state.timestamp_us
            )
            dx = (
                final.pose_local_to_rig.vec.x
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
    gt_position_in_local = final.pose_local_to_rig.vec
    estimated_position_in_local = final.pose_local_to_rig_estimated.vec
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
