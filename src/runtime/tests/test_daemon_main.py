# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from alpasim_grpc.v0 import common_pb2, runtime_pb2
from alpasim_runtime.daemon.app import RuntimeDaemonApp
from alpasim_runtime.daemon.engine import UnknownSceneError
from alpasim_runtime.daemon.servicer import RuntimeDaemonServicer
from alpasim_runtime.simulate.__main__ import _serve, create_arg_parser, run_simulation

import grpc


class _AbortContext:
    def __init__(self) -> None:
        self.code = None
        self.details = None

    async def abort(self, code, details):
        self.code = code
        self.details = details
        raise RuntimeError("aborted")


@pytest.mark.asyncio
async def test_servicer_maps_unknown_scene_to_invalid_argument() -> None:
    engine = SimpleNamespace(
        simulate=AsyncMock(side_effect=UnknownSceneError("scene-x"))
    )
    servicer = RuntimeDaemonServicer(engine=engine)
    context = _AbortContext()

    with pytest.raises(RuntimeError, match="aborted"):
        await servicer.simulate(runtime_pb2.SimulationRequest(), context)

    assert context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "scene-x" in context.details


@pytest.mark.asyncio
async def test_servicer_returns_rollout_returns_in_request_order() -> None:
    response = runtime_pb2.SimulationReturn(
        rollout_returns=[
            runtime_pb2.SimulationReturn.RolloutReturn(
                rollout_spec=runtime_pb2.RolloutSpec(
                    scenario_id="clipgt-a", nr_rollouts=1
                )
            ),
            runtime_pb2.SimulationReturn.RolloutReturn(
                rollout_spec=runtime_pb2.RolloutSpec(
                    scenario_id="clipgt-b", nr_rollouts=2
                )
            ),
        ]
    )

    engine = SimpleNamespace(
        simulate=AsyncMock(return_value=response),
    )
    servicer = RuntimeDaemonServicer(engine=engine)

    response = await servicer.simulate(runtime_pb2.SimulationRequest(), _AbortContext())
    assert [item.rollout_spec.scenario_id for item in response.rollout_returns] == [
        "clipgt-a",
        "clipgt-b",
    ]


@pytest.mark.asyncio
async def test_servicer_shutdown_rpc_acknowledges_and_triggers_callback() -> None:
    on_shutdown_requested = Mock()
    servicer = RuntimeDaemonServicer(
        engine=SimpleNamespace(),
        on_shutdown_requested=on_shutdown_requested,
    )

    response = await servicer.shut_down(common_pb2.Empty(), _AbortContext())

    assert response == common_pb2.Empty()
    on_shutdown_requested.assert_called_once_with()


@pytest.mark.asyncio
async def test_servicer_shutdown_rpc_is_safe_for_repeated_requests() -> None:
    app = RuntimeDaemonApp(
        engine=SimpleNamespace(startup=AsyncMock(), shutdown=AsyncMock()),
        listen_address="[::]:50051",
    )
    servicer = RuntimeDaemonServicer(
        engine=SimpleNamespace(),
        on_shutdown_requested=app.request_shutdown,
    )

    response_1 = await servicer.shut_down(common_pb2.Empty(), _AbortContext())
    response_2 = await servicer.shut_down(common_pb2.Empty(), _AbortContext())
    await asyncio.wait_for(app._wait_for_shutdown_request(), timeout=0.5)

    assert response_1 == common_pb2.Empty()
    assert response_2 == common_pb2.Empty()


@pytest.mark.asyncio
async def test_daemon_main_serve_starts_and_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace()
    app_instances = []

    class _FakeApp:
        def __init__(
            self,
            engine,
            listen_address: str,
        ) -> None:
            self.engine = engine
            self.listen_address = listen_address
            self.shutdown_requested = asyncio.Event()
            app_instances.append(self)

        async def wait_for_shutdown_request(self) -> None:
            await self.shutdown_requested.wait()

        async def run(self) -> None:
            return None

    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.DaemonEngine",
        lambda **kwargs: engine,
    )
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.RuntimeDaemonApp",
        _FakeApp,
    )

    args = Namespace(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        usdz_glob="/tmp/*.usdz",
        log_dir="/tmp/log",
        listen_address="[::]:50051",
    )

    await asyncio.wait_for(_serve(args), timeout=0.5)

    assert len(app_instances) == 1
    app = app_instances[0]
    assert app.engine is engine
    assert app.listen_address == "[::]:50051"


@pytest.mark.asyncio
async def test_daemon_main_serve_stops_on_app_shutdown_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace()
    app_instances = []

    class _FakeApp:
        def __init__(
            self,
            engine,
            listen_address: str,
        ) -> None:
            self.engine = engine
            self.listen_address = listen_address
            self.shutdown_requested = asyncio.Event()
            app_instances.append(self)

        async def wait_for_shutdown_request(self) -> None:
            await self.shutdown_requested.wait()

        async def run(self) -> None:
            self.shutdown_requested.set()
            await asyncio.wait_for(self.wait_for_shutdown_request(), timeout=0.5)

    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.DaemonEngine",
        lambda **kwargs: engine,
    )
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.RuntimeDaemonApp",
        _FakeApp,
    )

    args = Namespace(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        usdz_glob="/tmp/*.usdz",
        log_dir="/tmp/log",
        listen_address="[::]:50051",
    )

    await asyncio.wait_for(_serve(args), timeout=0.5)

    assert len(app_instances) == 1


def test_simulate_parser_supports_serve_mode_args() -> None:
    parser = create_arg_parser()
    args = parser.parse_args(
        [
            "--user-config=u.yaml",
            "--network-config=n.yaml",
            "--eval-config=e.yaml",
            "--usdz-glob=/tmp/*.usdz",
            "--log-dir=/tmp/log",
            "--serve",
            "--listen-address=[::]:50060",
        ]
    )

    assert args.serve is True
    assert args.listen_address == "[::]:50060"


def test_simulate_parser_rejects_removed_grpc_shutdown_flag() -> None:
    parser = create_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--user-config=u.yaml",
                "--network-config=n.yaml",
                "--eval-config=e.yaml",
                "--usdz-glob=/tmp/*.usdz",
                "--log-dir=/tmp/log",
                "--serve",
                "--grpc-graceful-shutdown-s=7.0",
            ]
        )


@pytest.mark.asyncio
async def test_runtime_daemon_app_run_starts_and_stops_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace(startup=AsyncMock(), shutdown=AsyncMock())

    class _FakeServer:
        def __init__(self) -> None:
            self.started = False
            self.stop_grace = None
            self.listen_address = None

        def add_insecure_port(self, listen_address: str) -> None:
            self.listen_address = listen_address

        async def start(self) -> None:
            self.started = True

        async def stop(self, grace: float) -> None:
            self.stop_grace = grace

    fake_server = _FakeServer()
    monkeypatch.setattr(
        "alpasim_runtime.daemon.app.grpc.aio.server",
        lambda: fake_server,
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.app.runtime_pb2_grpc.add_RuntimeServiceServicer_to_server",
        lambda _servicer, _server: None,
    )

    app = RuntimeDaemonApp(
        engine=engine,
        listen_address="[::]:50051",
    )

    app.request_shutdown()
    await app.run()

    engine.startup.assert_awaited_once()
    engine.shutdown.assert_awaited_once()
    assert fake_server.started is True
    assert fake_server.listen_address == "[::]:50051"
    assert fake_server.stop_grace == 10.0


@pytest.mark.asyncio
async def test_runtime_daemon_app_servicer_shutdown_request_stops_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace(startup=AsyncMock(), shutdown=AsyncMock())

    class _FakeServer:
        def __init__(self) -> None:
            self.stop_grace = None

        def add_insecure_port(self, listen_address: str) -> None:
            return None

        async def start(self) -> None:
            return None

        async def stop(self, grace: float) -> None:
            self.stop_grace = grace

    fake_server = _FakeServer()
    captured_servicer: RuntimeDaemonServicer | None = None
    servicer_registered = asyncio.Event()

    def register_servicer(servicer: RuntimeDaemonServicer, _server: object) -> None:
        nonlocal captured_servicer
        captured_servicer = servicer
        servicer_registered.set()

    monkeypatch.setattr(
        "alpasim_runtime.daemon.app.grpc.aio.server",
        lambda: fake_server,
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.app.runtime_pb2_grpc.add_RuntimeServiceServicer_to_server",
        register_servicer,
    )

    app = RuntimeDaemonApp(
        engine=engine,
        listen_address="[::]:50051",
    )

    run_task = asyncio.create_task(app.run())
    await asyncio.wait_for(servicer_registered.wait(), timeout=0.5)

    if captured_servicer is None:
        raise AssertionError("servicer was not registered")
    await captured_servicer.shut_down(common_pb2.Empty(), _AbortContext())
    await run_task

    engine.shutdown.assert_awaited_once()
    assert fake_server.stop_grace == 10.0


@pytest.mark.asyncio
async def test_runtime_daemon_app_shutdowns_engine_when_server_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace(startup=AsyncMock(), shutdown=AsyncMock())

    class _FakeServer:
        def add_insecure_port(self, listen_address: str) -> None:
            return None

        async def start(self) -> None:
            raise RuntimeError("start failed")

        async def stop(self, grace: float) -> None:
            return None

    monkeypatch.setattr(
        "alpasim_runtime.daemon.app.grpc.aio.server",
        lambda: _FakeServer(),
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.app.runtime_pb2_grpc.add_RuntimeServiceServicer_to_server",
        lambda _servicer, _server: None,
    )

    app = RuntimeDaemonApp(
        engine=engine,
        listen_address="[::]:50051",
    )

    with pytest.raises(RuntimeError, match="start failed"):
        await app.run()

    engine.startup.assert_awaited_once()
    engine.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_daemon_app_shutdowns_engine_when_server_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace(startup=AsyncMock(), shutdown=AsyncMock())

    class _FakeServer:
        def add_insecure_port(self, listen_address: str) -> None:
            return None

        async def start(self) -> None:
            return None

        async def stop(self, grace: float) -> None:
            raise RuntimeError("stop failed")

    monkeypatch.setattr(
        "alpasim_runtime.daemon.app.grpc.aio.server",
        lambda: _FakeServer(),
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.app.runtime_pb2_grpc.add_RuntimeServiceServicer_to_server",
        lambda _servicer, _server: None,
    )

    app = RuntimeDaemonApp(
        engine=engine,
        listen_address="[::]:50051",
    )

    app.request_shutdown()
    with pytest.raises(RuntimeError, match="stop failed"):
        await app.run()

    engine.startup.assert_awaited_once()
    engine.shutdown.assert_awaited_once()


def _make_one_shot_args() -> Namespace:
    return Namespace(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        usdz_glob="/tmp/*.usdz",
        log_dir="/tmp/log",
        array_job_dir=None,
    )


def _patch_one_shot_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_config: SimpleNamespace,
    fake_eval_config: SimpleNamespace,
    request: runtime_pb2.SimulationRequest,
) -> None:
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.parse_simulator_config",
        Mock(return_value=fake_config),
    )
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.typed_parse_config",
        Mock(return_value=fake_eval_config),
    )
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.build_simulation_request",
        Mock(return_value=request),
    )


def _make_simulation_return(
    results: list[tuple[str, bool, str | None]],
) -> runtime_pb2.SimulationReturn:
    return runtime_pb2.SimulationReturn(
        rollout_returns=[
            runtime_pb2.SimulationReturn.RolloutReturn(
                rollout_spec=runtime_pb2.RolloutSpec(
                    scenario_id=scenario_id,
                    nr_rollouts=1,
                ),
                success=success,
                rollout_uuid="",
                error=error or "",
            )
            for scenario_id, success, error in results
        ]
    )


@pytest.mark.asyncio
async def test_run_simulation_one_shot_uses_daemon_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config = SimpleNamespace(
        user=SimpleNamespace(nr_workers=1),
    )
    fake_eval_config = SimpleNamespace(run_in_runtime=False, enabled=False)
    _patch_one_shot_inputs(
        monkeypatch,
        fake_config=fake_config,
        fake_eval_config=fake_eval_config,
        request=runtime_pb2.SimulationRequest(
            rollout_specs=[
                runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=2),
                runtime_pb2.RolloutSpec(scenario_id="clipgt-b", nr_rollouts=1),
            ]
        ),
    )

    simulation_return = _make_simulation_return(
        [
            ("clipgt-a", True, None),
            ("clipgt-a", True, None),
            ("clipgt-b", True, None),
        ]
    )
    fake_engine = SimpleNamespace(
        startup=AsyncMock(),
        simulate=AsyncMock(return_value=simulation_return),
        shutdown=AsyncMock(),
    )

    engine_cls = Mock(return_value=fake_engine)
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.DaemonEngine",
        engine_cls,
    )

    shutdown_services = AsyncMock()
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.shutdown_services",
        shutdown_services,
    )
    merge_metrics_files = Mock()
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.merge_metrics_files",
        merge_metrics_files,
    )
    generate_metrics_plot = Mock(return_value="/tmp/log/metrics_plot.png")
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.generate_metrics_plot",
        generate_metrics_plot,
    )
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.get_run_name",
        Mock(return_value="run-name"),
    )

    args = _make_one_shot_args()

    success = await run_simulation(args)

    assert success is True
    engine_cls.assert_called_once_with(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        usdz_glob="/tmp/*.usdz",
        log_dir="/tmp/log",
    )
    fake_engine.startup.assert_awaited_once()
    fake_engine.shutdown.assert_awaited_once()
    shutdown_services.assert_awaited_once_with(fake_config)
    merge_metrics_files.assert_called_once_with("/tmp/log/telemetry")
    generate_metrics_plot.assert_called_once()

    request = fake_engine.simulate.await_args.args[0]
    assert isinstance(request, runtime_pb2.SimulationRequest)
    assert [
        (rollout_spec.scenario_id, rollout_spec.nr_rollouts)
        for rollout_spec in request.rollout_specs
    ] == [("clipgt-a", 2), ("clipgt-b", 1)]


@pytest.mark.asyncio
async def test_run_simulation_one_shot_fails_when_result_count_mismatches_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config = SimpleNamespace(
        user=SimpleNamespace(nr_workers=1),
    )
    fake_eval_config = SimpleNamespace(run_in_runtime=False, enabled=False)
    _patch_one_shot_inputs(
        monkeypatch,
        fake_config=fake_config,
        fake_eval_config=fake_eval_config,
        request=runtime_pb2.SimulationRequest(
            rollout_specs=[
                runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=2),
                runtime_pb2.RolloutSpec(scenario_id="clipgt-b", nr_rollouts=1),
            ]
        ),
    )

    simulation_return = _make_simulation_return(
        [
            ("clipgt-a", True, None),
            ("clipgt-b", True, None),
        ]
    )
    fake_engine = SimpleNamespace(
        startup=AsyncMock(),
        simulate=AsyncMock(return_value=simulation_return),
        shutdown=AsyncMock(),
    )

    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.DaemonEngine",
        Mock(return_value=fake_engine),
    )

    shutdown_services = AsyncMock()
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.shutdown_services",
        shutdown_services,
    )
    merge_metrics_files = Mock()
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.merge_metrics_files",
        merge_metrics_files,
    )
    generate_metrics_plot = Mock()
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.generate_metrics_plot",
        generate_metrics_plot,
    )
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.get_run_name",
        Mock(return_value="run-name"),
    )

    args = _make_one_shot_args()

    with pytest.raises(RuntimeError, match="expected 3 results, got 2"):
        await run_simulation(args)

    fake_engine.shutdown.assert_awaited_once()
    shutdown_services.assert_awaited_once_with(fake_config)
    merge_metrics_files.assert_not_called()
    generate_metrics_plot.assert_not_called()


@pytest.mark.asyncio
async def test_run_simulation_one_shot_shutdowns_when_engine_simulate_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config = SimpleNamespace(
        user=SimpleNamespace(nr_workers=1),
    )
    fake_eval_config = SimpleNamespace(run_in_runtime=False, enabled=False)
    _patch_one_shot_inputs(
        monkeypatch,
        fake_config=fake_config,
        fake_eval_config=fake_eval_config,
        request=runtime_pb2.SimulationRequest(
            rollout_specs=[
                runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=1),
            ]
        ),
    )

    fake_engine = SimpleNamespace(
        startup=AsyncMock(),
        simulate=AsyncMock(side_effect=RuntimeError("simulate failed")),
        shutdown=AsyncMock(),
    )

    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.DaemonEngine",
        Mock(return_value=fake_engine),
    )

    shutdown_services = AsyncMock()
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.shutdown_services",
        shutdown_services,
    )

    args = _make_one_shot_args()

    with pytest.raises(RuntimeError, match="simulate failed"):
        await run_simulation(args)

    fake_engine.startup.assert_awaited_once()
    fake_engine.simulate.assert_awaited_once()
    fake_engine.shutdown.assert_awaited_once()
    shutdown_services.assert_awaited_once_with(fake_config)


@pytest.mark.asyncio
async def test_run_simulation_one_shot_shutdowns_when_engine_startup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config = SimpleNamespace(
        user=SimpleNamespace(nr_workers=1),
    )
    fake_eval_config = SimpleNamespace(run_in_runtime=False, enabled=False)
    _patch_one_shot_inputs(
        monkeypatch,
        fake_config=fake_config,
        fake_eval_config=fake_eval_config,
        request=runtime_pb2.SimulationRequest(
            rollout_specs=[
                runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=1),
            ]
        ),
    )

    fake_engine = SimpleNamespace(
        startup=AsyncMock(side_effect=RuntimeError("startup failed")),
        simulate=AsyncMock(),
        shutdown=AsyncMock(),
    )

    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.DaemonEngine",
        Mock(return_value=fake_engine),
    )

    shutdown_services = AsyncMock()
    monkeypatch.setattr(
        "alpasim_runtime.simulate.__main__.shutdown_services",
        shutdown_services,
    )

    args = _make_one_shot_args()

    with pytest.raises(RuntimeError, match="startup failed"):
        await run_simulation(args)

    fake_engine.simulate.assert_not_awaited()
    fake_engine.shutdown.assert_awaited_once()
    shutdown_services.assert_awaited_once_with(fake_config)
