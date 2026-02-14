# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Unified data source loader: supports loading data from USDZ or trajdata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from alpasim_utils.artifact import Artifact
from alpasim_utils.scene_data_source import SceneDataSource

logger = logging.getLogger(__name__)

try:
    from trajdata.dataset import UnifiedDataset
    from alpasim_utils.trajdata_data_source import discover_from_trajdata_dataset
    TRAJDATA_AVAILABLE = True
except ImportError as e:
    TRAJDATA_AVAILABLE = False
    UnifiedDataset = None
    logger.debug(f"trajdata not available: {e}. Trajdata loading will be disabled.")


def load_data_sources(
    usdz_glob: Optional[str] = None,
    trajdata_config_path: Optional[str] = None,
    smooth_trajectories: bool = True,
) -> dict[str, SceneDataSource]:
    """
    Unified data source loading function that supports loading from USDZ or trajdata.
    
    Args:
        usdz_glob: Glob pattern for USDZ files (mutually exclusive with trajdata_config_path)
        trajdata_config_path: Path to JSON/YAML file containing trajdata configuration
        smooth_trajectories: Whether to smooth trajectories
    
    Returns:
        dict[scene_id, SceneDataSource]
    
    Raises:
        ValueError: If both or neither of usdz_glob and trajdata_config_path are provided
        ImportError: If trajdata is not installed but trajdata_config_path is provided
    """
    if usdz_glob is not None and trajdata_config_path is not None:
        raise ValueError(
            "Cannot specify both usdz_glob and trajdata_config_path. Choose one."
        )
    
    if usdz_glob is None and trajdata_config_path is None:
        raise ValueError(
            "Either usdz_glob or trajdata_config_path must be provided."
        )
    
    if usdz_glob is not None:
        # Load from USDZ files
        logger.info(f"Loading artifacts from USDZ files: {usdz_glob}")
        artifacts = Artifact.discover_from_glob(
            usdz_glob, smooth_trajectories=smooth_trajectories
        )
        logger.info(f"Loaded {len(artifacts)} scenes from USDZ files")
        return artifacts
    
    # Load from trajdata
    if not TRAJDATA_AVAILABLE:
        raise ImportError(
            "trajdata is not installed. Install trajdata to use trajdata_config_path option."
        )
    
    logger.info(f"Loading artifacts from trajdata config: {trajdata_config_path}")
    
    # Load config file
    config_path = Path(trajdata_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Trajdata config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            config = yaml.safe_load(f)
        else:
            # Assume JSON
            config = json.load(f)
    
    # Extract scene_indices if provided
    scene_indices = config.pop("scene_indices", None)
    
    # Create UnifiedDataset
    dataset = UnifiedDataset(**config)
    
    # Discover data sources
    artifacts = discover_from_trajdata_dataset(
        dataset=dataset,
        scene_indices=scene_indices,
        smooth_trajectories=smooth_trajectories,
    )
    logger.info(f"Loaded {len(artifacts)} scenes from trajdata dataset")
    return artifacts
