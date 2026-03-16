# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Metadata utilities for reading run configuration files."""

import json
import logging
from glob import glob
from pathlib import Path
from typing import Any

from alpasim_utils.yaml_utils import load_yaml_dict

logger = logging.getLogger("alpasim_eval.metadata")


def get_metadata(config_dir: Path) -> dict[str, Any]:
    """Load run metadata from the configuration directory.

    Reads run_metadata.yaml and collects all other YAML files in the directory,
    serializing them as a JSON string in the 'yamls' field.

    Args:
        config_dir: Path to the directory containing configuration files.

    Returns:
        Dictionary containing run metadata with collected YAML configs.
    """
    # Read run metadata from same directory as config
    run_metadata_path = config_dir / "run_metadata.yaml"
    run_metadata = load_yaml_dict(run_metadata_path, missing_ok=True)
    if not run_metadata_path.exists():
        logger.warning("File not found at %s", run_metadata_path)
    logger.debug("Loaded run metadata: %s", run_metadata)

    yamls_to_upload = glob(f"{config_dir}/*.yaml", recursive=True)
    # Convert string paths from glob to Path objects consistently
    yaml_paths = [
        Path(path) for path in yamls_to_upload if Path(path) != run_metadata_path
    ]

    logger.debug("Yamls to collect: %s", yaml_paths)
    yaml_dict = {
        path.name: load_yaml_dict(path, missing_ok=True) for path in yaml_paths
    }

    # Serialize the dictionary to JSON string
    run_metadata["yamls"] = json.dumps(yaml_dict)
    return run_metadata
