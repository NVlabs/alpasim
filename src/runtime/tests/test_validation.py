# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for parent-canonical version probing in validation.py."""

import asyncio
import logging

import pytest
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.config import (
    EndpointAddresses,
    NetworkSimulatorConfig,
    PhysicsUpdateMode,
    RendererConfig,
    RendererKind,
    RuntimeCameraConfig,
    SceneConfig,
    ServiceEndpoint,
    SimulationConfig,
    SimulatorConfig,
    SingleUserEndpointConfig,
    UserEndpointConfig,
    UserSimulatorConfig,
)
from alpasim_runtime.validation import (
    _log_awaitable_progress,
    gather_versions_from_addresses,
    validate_scenarios,
)


def _make_network_config() -> NetworkSimulatorConfig:
    """Create a minimal NetworkSimulatorConfig with one address per service."""
    return NetworkSimulatorConfig(
        driver=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50051")]),
        renderer=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50052")]),
        physics=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50053")]),
        trafficsim=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50054")]),
        controller=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50055")]),
    )


def _make_network_config_with_two_driver_addresses() -> NetworkSimulatorConfig:
    """Create a config where driver has two addresses (for mismatch testing)."""
    return NetworkSimulatorConfig(
        driver=EndpointAddresses(
            endpoints=[
                ServiceEndpoint("localhost:50051"),
                ServiceEndpoint("localhost:50061"),
            ]
        ),
        renderer=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50052")]),
        physics=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50053")]),
        trafficsim=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50054")]),
        controller=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50055")]),
    )


def _make_user_endpoints(
    skip_physics: bool = False,
) -> UserEndpointConfig:
    """Create a UserEndpointConfig with optional skip flags."""
    return UserEndpointConfig(
        driver=SingleUserEndpointConfig(skip=False, n_concurrent_rollouts=1),
        renderer=SingleUserEndpointConfig(skip=False, n_concurrent_rollouts=1),
        physics=SingleUserEndpointConfig(skip=skip_physics, n_concurrent_rollouts=1),
        trafficsim=SingleUserEndpointConfig(skip=False, n_concurrent_rollouts=1),
        controller=SingleUserEndpointConfig(skip=False, n_concurrent_rollouts=1),
    )


def _make_simulator_config(
    *,
    physics_update_mode: PhysicsUpdateMode = PhysicsUpdateMode.EGO_ONLY,
    force_gt_duration_us: int = 100_000,
    skip_physics: bool = False,
    cameras: list[RuntimeCameraConfig] | None = None,
) -> SimulatorConfig:
    if cameras is None:
        cameras = [RuntimeCameraConfig(logical_id="camera_front")]

    return SimulatorConfig(
        user=UserSimulatorConfig(
            simulation_config=SimulationConfig(
                n_sim_steps=1,
                n_rollouts=1,
                physics_update_mode=physics_update_mode,
                force_gt_duration_us=force_gt_duration_us,
                cameras=cameras,
            ),
            scenes=[SceneConfig(scene_id="clipgt-test", n_rollouts=1)],
            endpoints=_make_user_endpoints(skip_physics=skip_physics),
            renderer=RendererConfig(kind=RendererKind.sensorsim),
            nr_workers=1,
        ),
        network=NetworkSimulatorConfig(
            driver=EndpointAddresses(endpoints=[]),
            renderer=EndpointAddresses(endpoints=[]),
            physics=EndpointAddresses(endpoints=[]),
            trafficsim=EndpointAddresses(endpoints=[]),
            controller=EndpointAddresses(endpoints=[]),
        ),
    )


@pytest.mark.asyncio
async def test_await_with_pending_log_reports_slow_operation(caplog):
    release_slow_probe = asyncio.Event()

    async def fast_probe() -> str:
        return "fast"

    async def slow_probe() -> str:
        await release_slow_probe.wait()
        return "slow"

    with caplog.at_level(logging.INFO, logger="alpasim_runtime.validation"):
        task = asyncio.gather(
            _log_awaitable_progress(
                fast_probe(),
                label="fast service",
                log_interval_s=0.01,
            ),
            _log_awaitable_progress(
                slow_probe(),
                label="slow service",
                log_interval_s=0.01,
            ),
        )
        await asyncio.sleep(0.03)
        release_slow_probe.set()
        fast_result, slow_result = await task

    assert fast_result == "fast"
    assert slow_result == "slow"
    assert "slow service" in caplog.text
    assert "fast service" not in caplog.text


@pytest.mark.asyncio
async def test_validate_scenarios_accepts_physics_enabled_config():
    await validate_scenarios(_make_simulator_config())


@pytest.mark.asyncio
async def test_validate_scenarios_rejects_physics_update_without_physics_service():
    with pytest.raises(AssertionError, match="requires the physics service"):
        await validate_scenarios(
            _make_simulator_config(
                skip_physics=True,
            )
        )


@pytest.mark.asyncio
async def test_validate_scenarios_rejects_running_physics_with_no_update():
    with pytest.raises(AssertionError, match="Physics is disabled"):
        await validate_scenarios(
            _make_simulator_config(
                physics_update_mode=PhysicsUpdateMode.NONE,
            )
        )


@pytest.mark.asyncio
async def test_validate_scenarios_accepts_physics_enabled_without_force_gt_duration():
    await validate_scenarios(
        _make_simulator_config(
            force_gt_duration_us=0,
        )
    )


@pytest.mark.asyncio
async def test_validate_scenarios_accepts_headless_rollout() -> None:
    """``_build_rollout_timing`` handles ``cameras=[]``, so validation no longer
    rejects headless rollouts at the pre-flight stage."""
    await validate_scenarios(
        _make_simulator_config(
            skip_physics=True,
            physics_update_mode=PhysicsUpdateMode.NONE,
            cameras=[],
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("renderer_kind", "expected_sensorsim_version", "expected_video_model_version"),
    [
        (RendererKind.sensorsim, "renderer-v1", "<skip>"),
        (RendererKind.video_model, "<skip>", "renderer-v1"),
    ],
)
async def test_gather_versions_returns_rollout_version_ids(
    monkeypatch,
    renderer_kind,
    expected_sensorsim_version,
    expected_video_model_version,
):
    """gather_versions_from_addresses should return a populated VersionIds proto."""

    async def fake_probe(svc_name, stub_class, address, timeout_s):
        del stub_class, timeout_s
        return (
            svc_name,
            address,
            VersionId(
                version_id=f"{svc_name}-v1",
                git_hash="abc",
            ),
        )

    monkeypatch.setattr(
        "alpasim_runtime.validation._probe_version_for_address", fake_probe
    )

    version_ids = await gather_versions_from_addresses(
        _make_network_config(),
        _make_user_endpoints(),
        renderer_kind=renderer_kind,
    )

    assert isinstance(version_ids, RolloutMetadata.VersionIds)
    assert version_ids.egodriver_version.version_id == "driver-v1"
    assert version_ids.sensorsim_version.version_id == expected_sensorsim_version
    assert version_ids.video_model_version.version_id == expected_video_model_version
    assert version_ids.physics_version.version_id == "physics-v1"
    assert version_ids.traffic_version.version_id == "trafficsim-v1"
    assert version_ids.controller_version.version_id == "controller-v1"
    # runtime version should be set from the runtime package
    assert version_ids.runtime_version.version_id != ""


@pytest.mark.asyncio
async def test_gather_versions_fails_on_mixed_service_versions(monkeypatch):
    """If the same service returns different versions from different addresses, fail."""

    call_count = {}

    async def fake_probe(svc_name, stub_class, address, timeout_s):
        del stub_class, timeout_s
        call_count.setdefault(svc_name, 0)
        call_count[svc_name] += 1
        # Second driver address returns a different version
        suffix = "v1" if call_count[svc_name] == 1 else "v2"
        return (
            svc_name,
            address,
            VersionId(
                version_id=f"{svc_name}-{suffix}",
                git_hash="abc",
            ),
        )

    monkeypatch.setattr(
        "alpasim_runtime.validation._probe_version_for_address", fake_probe
    )

    with pytest.raises(AssertionError, match="mixed versions"):
        await gather_versions_from_addresses(
            _make_network_config_with_two_driver_addresses(),
            _make_user_endpoints(),
            renderer_kind=RendererKind.sensorsim,
        )


@pytest.mark.asyncio
async def test_gather_versions_uses_skip_version_without_probing(monkeypatch):
    """Skipped services should get a '<skip>' VersionId without making gRPC calls."""

    probed_services = set()

    async def fake_probe(svc_name, stub_class, address, timeout_s):
        del stub_class, timeout_s
        probed_services.add(svc_name)
        return (
            svc_name,
            address,
            VersionId(
                version_id=f"{svc_name}-v1",
                git_hash="abc",
            ),
        )

    monkeypatch.setattr(
        "alpasim_runtime.validation._probe_version_for_address", fake_probe
    )

    version_ids = await gather_versions_from_addresses(
        _make_network_config(),
        _make_user_endpoints(skip_physics=True),
        renderer_kind=RendererKind.sensorsim,
    )

    assert "physics" not in probed_services
    assert version_ids.physics_version.version_id == "<skip>"
    assert version_ids.physics_version.grpc_api_version == API_VERSION_MESSAGE
    # Other services should still be probed normally
    assert version_ids.egodriver_version.version_id == "driver-v1"


@pytest.mark.asyncio
async def test_gather_versions_fails_on_mixed_git_hash_with_same_version_id(
    monkeypatch,
):
    """Mismatch in git_hash across addresses should fail even with same version_id."""

    call_count = {}

    async def fake_probe(svc_name, stub_class, address, timeout_s):
        del stub_class, timeout_s
        call_count.setdefault(svc_name, 0)
        call_count[svc_name] += 1
        git_hash = "aaa" if call_count[svc_name] == 1 else "bbb"
        return (
            svc_name,
            address,
            VersionId(
                version_id=f"{svc_name}-v1",
                git_hash=git_hash,
            ),
        )

    monkeypatch.setattr(
        "alpasim_runtime.validation._probe_version_for_address", fake_probe
    )

    with pytest.raises(AssertionError, match="mixed versions"):
        await gather_versions_from_addresses(
            _make_network_config_with_two_driver_addresses(),
            _make_user_endpoints(),
            renderer_kind=RendererKind.sensorsim,
        )


@pytest.mark.asyncio
async def test_gather_versions_probes_all_addresses_per_service(monkeypatch):
    """When a service has multiple addresses, all should be probed for consistency."""

    probed_addresses = []

    async def fake_probe(svc_name, stub_class, address, timeout_s):
        del stub_class, timeout_s
        probed_addresses.append((svc_name, address))
        return (
            svc_name,
            address,
            VersionId(
                version_id=f"{svc_name}-v1",
                git_hash="abc",
            ),
        )

    monkeypatch.setattr(
        "alpasim_runtime.validation._probe_version_for_address", fake_probe
    )

    await gather_versions_from_addresses(
        _make_network_config_with_two_driver_addresses(),
        _make_user_endpoints(),
        renderer_kind=RendererKind.sensorsim,
    )

    driver_probes = [
        (name, addr) for name, addr in probed_addresses if name == "driver"
    ]
    assert len(driver_probes) == 2
    assert driver_probes[0][1] == "localhost:50051"
    assert driver_probes[1][1] == "localhost:50061"


@pytest.mark.asyncio
async def test_gather_versions_fails_when_non_skipped_service_has_no_addresses(
    monkeypatch,
):
    """Non-skipped services must provide at least one endpoint address."""

    async def fake_probe(svc_name, stub_class, address, timeout_s):
        del stub_class, timeout_s
        return (
            svc_name,
            address,
            VersionId(
                version_id=f"{svc_name}-v1",
                git_hash="abc",
            ),
        )

    monkeypatch.setattr(
        "alpasim_runtime.validation._probe_version_for_address", fake_probe
    )

    network_config = _make_network_config()
    network_config.driver.endpoints = []

    with pytest.raises(AssertionError, match="driver"):
        await gather_versions_from_addresses(
            network_config,
            _make_user_endpoints(),
            renderer_kind=RendererKind.sensorsim,
        )
