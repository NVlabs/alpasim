# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Central registry of gRPC service stubs and endpoint helpers."""

from typing import Type

from alpasim_grpc.v0.controller_pb2_grpc import VDCServiceStub
from alpasim_grpc.v0.egodriver_pb2_grpc import EgodriverServiceStub
from alpasim_grpc.v0.physics_pb2_grpc import PhysicsServiceStub
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceStub
from alpasim_grpc.v0.traffic_pb2_grpc import TrafficServiceStub
from alpasim_runtime.config import NetworkSimulatorConfig

# Central mapping of service names to their gRPC stub classes.
# The keys match the attribute names in NetworkSimulatorConfig.
SERVICE_STUBS: dict[str, Type] = {
    "driver": EgodriverServiceStub,
    "sensorsim": SensorsimServiceStub,
    "physics": PhysicsServiceStub,
    "trafficsim": TrafficServiceStub,
    "controller": VDCServiceStub,
}


def get_service_endpoints(
    network_config: NetworkSimulatorConfig,
    services: list[str] | None = None,
) -> dict[str, tuple[Type, list[str]]]:
    """
    Get service stubs paired with their addresses from the network config.

    Args:
        network_config: The network configuration containing service addresses.
        services: Optional list of service names to include. If None, includes all.

    Returns:
        Dict mapping service name -> (stub_class, addresses list).
    """
    if services is None:
        services = list(SERVICE_STUBS.keys())

    return {
        name: (SERVICE_STUBS[name], getattr(network_config, name).addresses)
        for name in services
    }
