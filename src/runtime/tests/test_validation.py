# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for parent-canonical version probing in validation.py."""

import pytest
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.config import (
    EndpointAddresses,
    NetworkSimulatorConfig,
    ServiceEndpoint,
    SingleUserEndpointConfig,
    UserEndpointConfig,
)
from alpasim_runtime.validation import gather_versions_from_addresses


def _make_network_config() -> NetworkSimulatorConfig:
    """Create a minimal NetworkSimulatorConfig with one address per service."""
    return NetworkSimulatorConfig(
        driver=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50051")]),
        sensorsim=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50052")]),
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
        sensorsim=EndpointAddresses(endpoints=[ServiceEndpoint("localhost:50052")]),
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
        sensorsim=SingleUserEndpointConfig(skip=False, n_concurrent_rollouts=1),
        physics=SingleUserEndpointConfig(skip=skip_physics, n_concurrent_rollouts=1),
        trafficsim=SingleUserEndpointConfig(skip=False, n_concurrent_rollouts=1),
        controller=SingleUserEndpointConfig(skip=False, n_concurrent_rollouts=1),
    )


@pytest.mark.asyncio
async def test_gather_versions_returns_rollout_version_ids(monkeypatch):
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
    )

    assert isinstance(version_ids, RolloutMetadata.VersionIds)
    assert version_ids.egodriver_version.version_id == "driver-v1"
    assert version_ids.sensorsim_version.version_id == "sensorsim-v1"
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
        await gather_versions_from_addresses(network_config, _make_user_endpoints())
