# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Native solver outcomes and fresh-optimizer recovery across MPC backends."""

from types import SimpleNamespace

import numpy as np
import osqp
import pytest
from alpasim_controller.mpc_controller import ControllerInput, MPCGains
from alpasim_controller.mpc_impl import SQPMPC, LinearMPC, NonlinearMPC
from alpasim_utils.geometry import Trajectory


def _input():
    return ControllerInput(
        np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
        Trajectory(
            np.arange(11, dtype=np.uint64) * 100_000,
            np.column_stack([np.arange(11), np.full(11, 0.2), np.zeros(11)]).astype(
                np.float32
            ),
            np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (11, 1)),
        ),
        0,
    )


def _assert_fresh_command(controller, request):
    fresh = type(controller)(n_horizon=5, gains=MPCGains(idx_start_penalty=0))
    observed = controller.compute_control(request)
    expected = fresh.compute_control(request)
    assert observed.status.startswith("solved") and expected.status.startswith("solved")
    np.testing.assert_array_equal(observed.control, expected.control)


@pytest.fixture(params=[LinearMPC, NonlinearMPC, SQPMPC])
def controller(request):
    return request.param(n_horizon=5, gains=MPCGains(idx_start_penalty=0))


@pytest.mark.parametrize(
    "outcome",
    [
        "primal infeasible",
        "exception",
        "nan",
        "inf",
        "solved inaccurate",
        "input_error",
    ],
)
def test_solver_outcome_and_recovery(controller, monkeypatch, outcome):
    nonlinear = isinstance(controller, NonlinearMPC)
    if nonlinear and outcome == "solved inaccurate":
        pytest.skip("IPOPT uses its native success flag")
    request = _input()
    assert controller.compute_control(request).status.startswith("solved")

    def solve(*args, **kwargs):
        if outcome == "exception":
            raise RuntimeError("native failure") if nonlinear else osqp.OSQPException(4)
        if outcome == "input_error":
            raise ValueError("invalid formulation")
        status = "solved" if outcome in ("nan", "inf") else outcome
        values = np.full(
            (2, 1) if nonlinear else args[0].n,
            {"nan": np.nan, "inf": np.inf}.get(outcome, 0.3),
        )
        if nonlinear:
            controller._mpc.solver_stats = {
                "success": status == "solved",
                "return_status": status,
            }
            return values
        return SimpleNamespace(info=SimpleNamespace(status=status), x=values)

    target, method = (
        (controller._mpc, "make_step") if nonlinear else (osqp.OSQP, "solve")
    )
    with monkeypatch.context() as patch:
        patch.setattr(target, method, solve)
        if outcome == "input_error":
            with pytest.raises(ValueError, match="invalid formulation"):
                controller.compute_control(request)
            return
        output = controller.compute_control(request)
    if outcome == "solved inaccurate":
        assert output.status.startswith("solved inaccurate")
        assert np.any(output.control != 0.0)
        return
    assert output.status.startswith("failed:")
    assert output.control is None
    _assert_fresh_command(controller, request)


def test_native_infeasible_solve_recovers_like_a_fresh_optimizer(controller):
    request = _input()
    assert controller.compute_control(request).status.startswith("solved")
    # At 60 m/s, the next state cannot satisfy the 45 m/s speed bound.
    request.state[3] = 60.0
    failed = controller.compute_control(request)
    assert failed.status.startswith("failed:")
    assert failed.control is None
    request.state[3] = 10.0
    _assert_fresh_command(controller, request)


def test_sqp_later_qp_failure_is_not_reported_solved(monkeypatch):
    controller = SQPMPC(n_horizon=5, gains=MPCGains(idx_start_penalty=0))
    native_solve = controller._qp_solver.solve
    calls = 0

    def solve(**kwargs):
        nonlocal calls
        calls += 1
        return (
            native_solve(**kwargs)
            if calls == 1
            else SimpleNamespace(
                info=SimpleNamespace(status="primal infeasible"), x=None
            )
        )

    monkeypatch.setattr(controller._qp_solver, "solve", solve)
    output = controller.compute_control(_input())
    assert calls == 2
    assert output.status == "failed: primal infeasible"
    assert output.control is None
    assert controller._previous_control is None
    assert controller._last_qp_solution is None


def test_sqp_reset_clears_native_primal_and_dual_warm_starts():
    controller = SQPMPC(n_horizon=5)
    controller._qp_solver.warm_start(
        x=np.full(controller._num_vars, 1000.0),
        y=np.full(controller._num_constraints, 1000.0),
    )
    controller.reset_warm_start()
    fresh = SQPMPC(n_horizon=5)
    # One iteration exposes the native starting iterate instead of converging past it.
    for candidate in (controller, fresh):
        candidate._qp_solver.update_settings(max_iter=1)
    observed = controller._qp_solver.solve(raise_error=False)
    expected = fresh._qp_solver.solve(raise_error=False)
    assert observed.info.status == expected.info.status == "solved"
    np.testing.assert_array_equal(observed.x, expected.x)


@pytest.mark.parametrize("converged", [True, False])
def test_sqp_preserves_earlier_inaccurate_qp_status(monkeypatch, converged):
    controller = SQPMPC(
        n_horizon=5,
        gains=MPCGains(idx_start_penalty=0, sqp_max_iterations=2),
    )
    calls = 0

    def solve(*args):
        nonlocal calls
        calls += 1
        step = 0.0 if converged and calls == 2 else 0.01
        return (
            np.full((6, 8), step),
            np.full((5, 2), step),
            "solved inaccurate" if calls == 1 else "solved",
        )

    monkeypatch.setattr(controller, "_solve_qp_step", solve)
    output = controller.compute_control(_input())
    assert calls == 2
    diagnostic = "2 sqp iterations" if converged else "max sqp iterations"
    assert output.status == f"solved inaccurate ({diagnostic})"
    assert np.all(np.isfinite(output.control))
    assert np.any(output.control != 0.0)
