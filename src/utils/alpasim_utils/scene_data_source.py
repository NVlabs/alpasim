# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
SceneDataSource Protocol for abstracting scene data loading.

This Protocol allows Runtime to work with different data sources (USDZ, Nuplan, Waymo, etc.)
without being tied to a specific implementation. Any class that implements this Protocol
can be used as a data source for alpasim Runtime.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

try:
    from trajdata.maps import VectorMap
except ImportError:
    VectorMap = None  # type: ignore

from alpasim_utils.artifact import Metadata
from alpasim_utils.scenario import Rig, TrafficObjects


@runtime_checkable
class SceneDataSource(Protocol):
    """
    Protocol defining the interface for scene data sources.
    
    Any class implementing this protocol can be used as a data source for alpasim Runtime.
    This allows supporting multiple data formats (USDZ, Nuplan, Waymo, etc.) without
    modifying Runtime code.
    
    Attributes:
        scene_id: Unique identifier for the scene
    """
    
    scene_id: str
    
    @property
    def rig(self) -> Rig:
        """
        Get the rig (ego vehicle) trajectory and configuration.
        
        Returns:
            Rig object containing trajectory, camera IDs, and vehicle config
        """
        ...
    
    @property
    def traffic_objects(self) -> TrafficObjects:
        """
        Get traffic objects (vehicles, pedestrians, etc.) in the scene.
        
        Returns:
            TrafficObjects dictionary mapping track_id to TrafficObject
        """
        ...
    
    @property
    def map(self) -> Optional[VectorMap]:
        """
        Get the vector map for the scene.
        
        Returns:
            VectorMap object or None if map data is not available
        """
        ...
    
    @property
    def metadata(self) -> Metadata:
        """
        Get metadata about the scene.
        
        Returns:
            Metadata object containing scene information
        """
        ...
