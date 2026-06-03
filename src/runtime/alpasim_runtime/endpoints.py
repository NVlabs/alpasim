# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Central registry of gRPC service stubs and endpoint helpers."""

from typing import Type

from alpasim_grpc.v0.controller_pb2_grpc import VDCServiceStub
from alpasim_grpc.v0.egodriver_pb2_grpc import EgodriverServiceStub
from alpasim_grpc.v0.physics_pb2_grpc import PhysicsServiceStub
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceStub
from alpasim_grpc.v0.traffic_pb2_grpc import TrafficServiceStub
from alpasim_grpc.v0.video_model_pb2_grpc import WorldModelServiceStub
from alpasim_runtime.config import (
    EndpointAddresses,
    NetworkSimulatorConfig,
    RendererKind,
)

# Central mapping of service names to their gRPC stub classes.
# The keys match the attribute names in NetworkSimulatorConfig.
SERVICE_STUBS: dict[str, Type] = {
    "driver": EgodriverServiceStub,
    "physics": PhysicsServiceStub,
    "trafficsim": TrafficServiceStub,
    "controller": VDCServiceStub,
}


def get_renderer_stub_class(renderer_kind: RendererKind) -> Type:
    if renderer_kind == RendererKind.sensorsim:
        return SensorsimServiceStub
    if renderer_kind == RendererKind.video_model:
        return WorldModelServiceStub
    raise ValueError(f"Unknown renderer kind: {renderer_kind!r}")


def get_endpoint_addresses(
    endpoint_config: EndpointAddresses,
    *,
    managed_only: bool = False,
) -> list[str]:
    """Return endpoint address strings, optionally filtering to managed endpoints.

    "Managed" here means the runtime started up the endpoint and is responsible for shutting it down.
    """
    return [
        endpoint.address
        for endpoint in endpoint_config.endpoints
        if not managed_only or endpoint.managed
    ]


def get_service_endpoints(
    network_config: NetworkSimulatorConfig,
    services: list[str] | None = None,
    *,
    renderer_kind: RendererKind,
    managed_only: bool = False,
) -> dict[str, tuple[Type, list[str]]]:
    """
    Get service stubs paired with their addresses from the network config.

    Args:
        network_config: The network configuration containing service addresses.
        services: Optional list of service names to include. If None, includes all.

        managed_only: If True, return only wizard-managed endpoints, i.e. those
        started by the runtime and which the runtime is responsible for shutting
        down.

    Returns:
        Dict mapping service name -> (stub_class, addresses list).
    """
    if services is None:
        services = [*SERVICE_STUBS.keys(), "renderer"]

    endpoints = {}
    for name in services:
        stub_class = (
            get_renderer_stub_class(renderer_kind)
            if name == "renderer"
            else SERVICE_STUBS[name]
        )
        addresses = get_endpoint_addresses(
            getattr(network_config, name),
            managed_only=managed_only,
        )
        endpoints[name] = (stub_class, addresses)
    return endpoints
