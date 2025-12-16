# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Context management for AlpasimWizard."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .compatibility import CompatibilityMatrix
from .scenes import LOCAL_SUITE_ID, SceneIdAndUuid, USDZManager
from .schema import AlpasimConfig
from .utils import nre_image_to_nre_version

logger = logging.getLogger(__name__)


def fetch_artifacts(cfg: AlpasimConfig) -> tuple[list[SceneIdAndUuid], str]:
    """Fetch artifacts using the scene manager."""

    compatibility_matrix = CompatibilityMatrix.from_config(
        cfg.scenes.artifact_compatibility_matrix
    )
    Path(cfg.scenes.scene_cache).mkdir(parents=True, exist_ok=True)

    # Get compatible NRE versions
    if cfg.scenes.nre_version_string is not None:
        nre_version = cfg.scenes.nre_version_string
    else:
        assert cfg.services.sensorsim is not None  # For mypy
        nre_version = nre_image_to_nre_version(cfg.services.sensorsim.image)
    compatible_versions = list(compatibility_matrix.lookup(nre_version))

    # Query scenes
    manager = USDZManager.from_cfg(cfg.scenes)

    # Determine which selection method to use
    test_suite_id = cfg.scenes.test_suite_id
    scene_ids = cfg.scenes.scene_ids
    use_local_scenes = cfg.scenes.local_usdz_dir is not None

    if test_suite_id is not None and scene_ids is not None:
        raise ValueError(
            "Both scene_ids and test_suite_id are set; only one can be set."
        )
    elif use_local_scenes and test_suite_id not in (None, LOCAL_SUITE_ID):
        raise ValueError(
            "When using local_usdz_dir, test_suite_id must be None or 'local'."
        )

    # If local_usdz_dir is set and neither scene_ids nor test_suite_id is provided,
    # default to using the "local" test suite (all scenes in the directory)
    if use_local_scenes:
        if test_suite_id is None and scene_ids is None:
            test_suite_id = LOCAL_SUITE_ID
            logger.info(
                f"Using local USDZ directory: {cfg.scenes.local_usdz_dir}. "
                f"Defaulting to test_suite_id='{test_suite_id}' (all scenes)."
            )

    if test_suite_id is not None:
        if scene_ids is not None:
            logger.warning(
                "Both scene_ids and test_suite_id are set; using test_suite_id=%s",
                test_suite_id,
            )
        artifacts = manager.query_by_suite_id(test_suite_id, compatible_versions)
    elif scene_ids is not None:
        artifacts = manager.query_by_scene_ids(scene_ids, compatible_versions)
    else:
        raise ValueError("Either scene_ids or test_suite_id must be set")

    # Create sceneset directory if not using local USDZ directory
    if use_local_scenes:
        sceneset_dir_abs_path = os.path.abspath(str(cfg.scenes.local_usdz_dir))
        sceneset_dir_relative_path = "."
        # Note: for local USDZ directories, override the scene cache directory for proper mounting
        cfg.scenes.scene_cache = sceneset_dir_abs_path
        logger.info(
            f"Using local files--overriding scene_cache to: {sceneset_dir_abs_path}"
        )
    else:
        sceneset_dir_abs_path = manager.create_sceneset_directory(
            [a.uuid for a in artifacts]
        )
        sceneset_dir_relative_path = str(
            Path(sceneset_dir_abs_path).relative_to(cfg.scenes.scene_cache)
        )
        logger.info(f"Relative sceneset path: {sceneset_dir_relative_path}")

    return artifacts, sceneset_dir_relative_path


def detect_gpus() -> int:
    """Detect number of GPUs on the system."""
    try:
        num_gpus = int(
            subprocess.check_output(
                "nvidia-smi -i 0 --query-gpu=count --format=csv,noheader",
                shell=True,
            )
        )
        logger.debug(f"Found {num_gpus} GPUs on system.")
    except subprocess.CalledProcessError as exc:
        logger.warning(
            f"Failed to determine GPU count via 'nvidia-smi' (code {exc.returncode}). "
            "Defaulting to 0 GPUs."
        )
        return 0
    return num_gpus


def setup_directories(cfg: AlpasimConfig) -> None:
    """Create necessary directories and symlinks."""
    log_dir = Path(cfg.wizard.log_dir)

    logger.debug(f"Creating log directory at path: {log_dir}")

    # Create subdirectories
    for subdir in ("asl", "metrics", "txt-logs", "controller"):
        subdir_path = log_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True, mode=0o777)
        os.chmod(subdir_path, 0o777)


@dataclass
class WizardContext:
    """Unified context for all wizard operations.

    Combines configuration access with runtime state,
    eliminating the need for a separate GlobalContext.
    """

    cfg: AlpasimConfig
    port_assigner: Iterator[int]

    # Expensive operations (only loaded when needed for actual execution)
    artifact_list: list[SceneIdAndUuid] = field(default_factory=list)
    num_gpus: int = 0
    sceneset_path: str | None = None

    def get_num_gpus(self) -> int:
        """Get GPU count with fallback to 0."""
        return self.num_gpus if self.num_gpus is not None else 0

    def get_artifacts(self) -> list[SceneIdAndUuid]:
        """Get artifacts with fallback to empty list."""
        return self.artifact_list if self.artifact_list is not None else []

    @property
    def all_services_to_run(self) -> list[str]:
        """Get all services that should be run."""
        result = []
        if self.cfg.wizard.run_sim_services:
            result.extend(self.cfg.wizard.run_sim_services)
        if self.cfg.wizard.run_eval_services:
            result.extend(self.cfg.wizard.run_eval_services)
        if self.cfg.wizard.run_aggregation_services:
            result.extend(self.cfg.wizard.run_aggregation_services)
        return result

    @staticmethod
    def create(cfg: AlpasimConfig) -> WizardContext:
        """Build context."""

        # Always set these basic attributes
        artifact_list, sceneset_path = fetch_artifacts(cfg)
        context = WizardContext(
            cfg=cfg,
            port_assigner=create_port_assigner(cfg.wizard.baseport),
            artifact_list=artifact_list,
            num_gpus=detect_gpus(),
            sceneset_path=sceneset_path,
        )

        setup_directories(cfg)

        return context


def _is_port_open(port: int) -> bool:
    """Check if a port is available (not in use)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def _find_next_open_port(start_port: int, max_search: int = 1000) -> int:
    """Find the next available port starting from start_port.

    Args:
        start_port: Port number to start searching from
        max_search: Maximum number of ports to check

    Returns:
        The next available port number

    Raises:
        RuntimeError: If no available port is found within the search range
    """
    for port in range(start_port, start_port + max_search):
        if _is_port_open(port):
            logger.debug(f"Found available port: {port}")
            return port
    raise RuntimeError(
        f"Could not find an available port in range {start_port}-{start_port + max_search - 1}"
    )


def create_port_assigner(baseport: int) -> Iterator[int]:
    """Create an iterator over port numbers starting from the first available port >= baseport."""

    def port_assigner() -> Iterator[int]:
        ports_assigned = 0
        max_ports = 1000
        # Start from the first available port >= baseport
        next_port = _find_next_open_port(baseport)

        while ports_assigned < max_ports:
            yield next_port
            ports_assigned += 1
            # Find the next available port after the one we just yielded
            next_port = _find_next_open_port(next_port + 1)

        raise AssertionError(
            f"Handed out {max_ports} different port numbers - something's fishy."
        )

    return port_assigner()
