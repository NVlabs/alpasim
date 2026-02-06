# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Integration tests for AlpaSim runtime using pytest-grpc with separate servers for each service.
This properly tests the runtime's ability to connect to different services on
different ports.

See alpasim_runtime/replay_services/README.md for instructions on how to generate
the needed files to be replayed.

To run the test, run:
```
pytest -m 'manual'
```
"""

import argparse
import logging
import subprocess
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
from unittest.mock import MagicMock, patch

import pytest
import yaml
from alpasim_grpc.v0 import (
    controller_pb2_grpc,
    egodriver_pb2_grpc,
    physics_pb2_grpc,
    sensorsim_pb2_grpc,
    traffic_pb2_grpc,
)
from alpasim_runtime.replay_services import (
    ControllerReplayService,
    DriverReplayService,
    PhysicsReplayService,
    SensorsimReplayService,
    TrafficReplayService,
)
from alpasim_runtime.replay_services.asl_reader import ASLReader
from alpasim_runtime.simulate.__main__ import run_simulation
from rich import print

import grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s:\t%(message)s",
    datefmt="%H:%M:%S",
    force=True,  # Override any existing configuration
)

# Required test data files - these are stored in Git LFS and will be
# automatically downloaded if missing when tests are run
REQUIRED_TEST_FILES = {
    "asl": "rollout.asl",
    "network_config": "generated-network-config.yaml",
    "user_config": "generated-user-config-0.yaml",
    "usdz": "6ea1c7a3-98b7-4adc-b774-4d9526371a0b.usdz",  # At least one USDZ file
}


# Service configuration registry
SERVICE_CONFIG: Dict[str, Tuple[Type[Any], Any]] = {
    "physics": (
        PhysicsReplayService,
        physics_pb2_grpc.add_PhysicsServiceServicer_to_server,
    ),
    "driver": (
        DriverReplayService,
        egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server,
    ),
    "trafficsim": (
        TrafficReplayService,
        traffic_pb2_grpc.add_TrafficServiceServicer_to_server,
    ),
    "controller": (
        ControllerReplayService,
        controller_pb2_grpc.add_VDCServiceServicer_to_server,
    ),
    "sensorsim": (
        SensorsimReplayService,
        sensorsim_pb2_grpc.add_SensorsimServiceServicer_to_server,
    ),
}


@pytest.fixture(scope="module")
def test_data_dir() -> Path:
    """Provide test data directory path, downloading via Git LFS if necessary."""
    # Define paths
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent.parent.parent
    integration_data_dir = test_dir / "data" / "integration"

    # Check if required test files exist
    required_files = [
        integration_data_dir / filename for filename in REQUIRED_TEST_FILES.values()
    ]

    missing_files = [f for f in required_files if not f.exists()]

    if missing_files:
        print(f"\nMissing test data files: {', '.join(str(f) for f in missing_files)}")
        print("Trying to download test data from git LFS...")

        # Ensure the integration directory exists
        integration_data_dir.mkdir(parents=True, exist_ok=True)

        # Pull LFS files if they're not already present
        subprocess.run(
            [
                "git",
                "lfs",
                "pull",
                "--include",
                "src/runtime/tests/data/integration/*",
            ],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        print("Test data downloaded successfully via git LFS")

    return integration_data_dir


@pytest.fixture(scope="function")
async def asl_reader(test_data_dir: Path) -> Optional[ASLReader]:
    """Provide ASL reader for all services."""
    asl_file = test_data_dir / REQUIRED_TEST_FILES["asl"]
    reader = ASLReader(str(asl_file))
    await reader.load_exchanges()
    return reader


@pytest.fixture
def runtime_configs(test_data_dir: Path, tmp_path: Path) -> Dict[str, str]:
    """Create runtime configuration files for testing.

    Uses pytest's tmp_path as log_dir to ensure test outputs are
    automatically cleaned up after test execution.
    """
    user_config_path = test_data_dir / REQUIRED_TEST_FILES["user_config"]
    user_config = yaml.safe_load(user_config_path.read_text(encoding="utf-8"))

    test_user_config = tmp_path / "test-user-config.yaml"
    test_user_config.write_text(yaml.dump(user_config), encoding="utf-8")

    network_config_path = test_data_dir / REQUIRED_TEST_FILES["network_config"]
    network_config = yaml.safe_load(network_config_path.read_text(encoding="utf-8"))

    # Replace all addresses with localhost instead of the docker bridge address
    for service_name, service_config in network_config.items():
        for address_idx in range(len(service_config["addresses"])):
            unused_hostname, port = service_config["addresses"][address_idx].split(":")
            service_config["addresses"][address_idx] = f"localhost:{port}"

    test_network_config = tmp_path / "test-network-config.yaml"
    test_network_config.write_text(yaml.dump(network_config), encoding="utf-8")

    # log_dir is where all outputs go (asl/, metrics/, txt-logs/)
    log_dir = tmp_path / "output"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("network_config:")
    print(yaml.dump(network_config, default_flow_style=False, indent=2))
    print("user_config:")
    print(yaml.dump(user_config, default_flow_style=False, indent=2))

    return {
        "user_config": str(test_user_config),
        "network_config": str(test_network_config),
        "usdz_glob": str(test_data_dir / "*.usdz"),
        "log_dir": str(log_dir),
    }


@pytest.fixture(scope="function")
def all_services(
    asl_reader: Optional[ASLReader], runtime_configs: Dict[str, str]
) -> None:
    """Create and start all service servers."""
    # Load network config to get service ports
    network_config = yaml.safe_load(open(runtime_configs["network_config"]))

    servers = {}

    # Create and start all services
    for service_name, (service_class, add_servicer_func) in SERVICE_CONFIG.items():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        service = service_class(asl_reader)
        add_servicer_func(service, server)
        address = network_config[service_name]["addresses"][0]
        # `unused_hostname` could be something like `physics-0` from the
        # docker network.
        hostname, unused_port = address.split(":")
        if hostname != "localhost":
            raise ValueError(
                f"Service {service_name} is configured to run on {hostname}, "
                "but we only support localhost for testing."
                "Manually change the network-config file."
            )
        logging.info(f"Starting {service_name} service on {address}")
        server.add_insecure_port(address)
        server.start()
        servers[service_name] = server

    yield

    # Stop all servers
    for server in servers.values():
        server.stop(0)


@patch("concurrent.futures.ProcessPoolExecutor")
@pytest.mark.manual
async def test_run_simulation_full(
    mock_executor_class: MagicMock,
    runtime_configs: Dict[str, str],
    asl_reader: Any,
    all_services: None,  # Needed to start the services
) -> None:
    """Test the complete run_simulation flow with multiple servers."""
    # Mock the process pool to run in-thread for testing
    mock_executor = MagicMock()
    mock_executor_class.return_value.__enter__.return_value = mock_executor

    # Make submit() run the function immediately in the same thread
    def immediate_submit(func: Any, *args: Any, **kwargs: Any) -> MagicMock:
        future = MagicMock()
        try:
            result = func(*args, **kwargs)
            future.result.return_value = result
        except Exception as e:
            future.result.side_effect = e
        return future

    mock_executor.submit.side_effect = immediate_submit

    # Create args namespace as if from command line
    args = argparse.Namespace(
        user_config=runtime_configs["user_config"],
        network_config=runtime_configs["network_config"],
        usdz_glob=runtime_configs["usdz_glob"],
        log_dir=runtime_configs["log_dir"],
    )

    # Run the full simulation
    success = await run_simulation(args)

    # Verify success
    assert success is True
    assert asl_reader.is_complete()
