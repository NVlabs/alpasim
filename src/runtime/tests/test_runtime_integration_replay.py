# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Manual integration replay test for runtime request determinism.

TODO: This test needs to be updated for the new trajdata-based data flow.
The test currently references usdz_glob which has been removed in favor of
DataSourceConfig. This test is marked as @pytest.mark.manual and is not
run in regular CI, so it can be updated as part of a follow-up task.

This test boots one replay gRPC server per service and re-runs runtime against
recorded artifacts. It asserts that runtime emits requests matching the ASL
recording exactly (ignoring expected dynamic fields).

How to refresh replay artifacts (ASL/USDZ/YAML)
------------------------------------------------
Regenerate these files in `src/runtime/tests/data/integration/` when runtime
behavior intentionally changes and this test needs a new baseline:

- `rollout.asl`
- `generated-network-config.yaml`
- `generated-user-config-0.yaml`
- `eval-config.yaml`
- one scene USDZ file (UUID-named, e.g. `9c264c49-82a8-4344-ada1-d93791120c27.usdz`)

1) Generate a fresh local rollout (from `src/wizard`):

```bash
RUN_DIR=/path/to/output/.wizard
uv run python -m alpasim_wizard \
    wizard.log_dir=${RUN_DIR} \
    deploy=local topology=1gpu driver=vavam \
    runtime.endpoints.trafficsim.skip=False \
    scenes.scene_ids="[clipgt-c14c031a-8c17-4d08-aa4d-23c020a6871e]" \
    runtime.simulation_config.n_sim_steps=60
```

2) Copy the generated artifacts from the run output into
`src/runtime/tests/data/integration/`, using the file names listed above.

3) If the scene changed, update the USDZ file accordingly:

- Add the new `.usdz` matching the scene/map UUID.
- Remove the old one if obsolete.
- Update `REQUIRED_TEST_FILES["usdz"]` in this file to the new name.

3a) Trim the USDZ to the minimum runtime inputs (recommended):

The raw wizard USDZ can be very large because it includes neural rendering
artifacts that replay tests do not use. For this test, runtime only needs:

- `metadata.yaml` (scene_id and metadata)
- `rig_trajectories.json` (ego trajectory and XODR transform)
- `sequence_tracks.json` (traffic trajectories)
- map data (`map.xodr` for this scene; alternatively `map_data/` or `clipgt/`)

Example repack command (replace `SRC_USDZ`/`DST_USDZ`):

```bash
python - <<'PY'
import zipfile

SRC_USDZ = "/path/to/full.usdz"
DST_USDZ = "/path/to/trimmed.usdz"
KEEP = [
    "metadata.yaml",
    "rig_trajectories.json",
    "sequence_tracks.json",
    "map.xodr",
]

with zipfile.ZipFile(SRC_USDZ, "r") as zin, zipfile.ZipFile(
    DST_USDZ, "w", compression=zipfile.ZIP_STORED
) as zout:
    for name in KEEP:
        data = zin.read(name)
        info = zin.getinfo(name)
        new_info = zipfile.ZipInfo(filename=name, date_time=info.date_time)
        new_info.external_attr = info.external_attr
        new_info.compress_type = zipfile.ZIP_STORED
        zout.writestr(new_info, data)
PY
```

If your scene has `map_data/` or `clipgt/` but no `map.xodr`, keep those map
directories instead.

4) Track large artifacts with Git LFS:

```bash
git lfs track "src/runtime/tests/data/integration/*.asl"
git lfs track "src/runtime/tests/data/integration/*.usdz"
```

5) Validate the updated baseline:

```bash
uv run pytest src/runtime/tests/test_runtime_integration_replay.py -m manual
```

Reference: `src/runtime/alpasim_runtime/replay_services/README.md`.
"""

import argparse
import logging
import subprocess
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, Generator, Tuple, Type

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
    "eval_config": "eval-config.yaml",
    "usdz": "9c264c49-82a8-4344-ada1-d93791120c27.usdz",  # At least one USDZ file
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
async def asl_reader(test_data_dir: Path) -> ASLReader | None:
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

    # Create run_metadata.yaml required by get_run_name()
    run_metadata = log_dir / "run_metadata.yaml"
    run_metadata.write_text("run_name: integration_replay_test\n")

    eval_config_path = test_data_dir / REQUIRED_TEST_FILES["eval_config"]

    print("network_config:")
    print(yaml.dump(network_config, default_flow_style=False, indent=2))
    print("user_config:")
    print(yaml.dump(user_config, default_flow_style=False, indent=2))

    return {
        "user_config": str(test_user_config),
        "network_config": str(test_network_config),
        "eval_config": str(eval_config_path),
        "usdz_glob": str(test_data_dir / "*.usdz"),
        "log_dir": str(log_dir),
    }


@pytest.fixture(scope="function")
def all_services(
    asl_reader: ASLReader | None, runtime_configs: Dict[str, str]
) -> Generator[None, None, None]:
    """Create and start all service servers."""
    # Load network config to get service ports
    network_config = yaml.safe_load(Path(runtime_configs["network_config"]).read_text())

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


@pytest.mark.manual
async def test_run_simulation_full(
    runtime_configs: Dict[str, str],
    asl_reader: Any,
    all_services: None,  # Needed to start the services
) -> None:
    """Test the complete run_simulation flow with replay servers.

    Verifies that re-running the runtime against recorded ASL responses
    produces identical requests, confirming deterministic behavior after
    code changes.
    """
    # Create args namespace matching what create_arg_parser() produces
    args = argparse.Namespace(
        user_config=runtime_configs["user_config"],
        network_config=runtime_configs["network_config"],
        usdz_glob=runtime_configs["usdz_glob"],
        log_dir=runtime_configs["log_dir"],
        eval_config=runtime_configs["eval_config"],
        array_job_dir=None,
        log_level="INFO",
    )

    # Run the full simulation
    success = await run_simulation(args)

    # Verify success
    assert success is True
    assert asl_reader.is_complete()
