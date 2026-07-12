# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from __future__ import annotations

import glob
import logging
import os
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field

import numpy as np
import yaml
from alpasim_utils.ply_io import load_mesh_vf
from alpasim_utils.scenario import Rig, TrafficObjects
from alpasim_utils.scene_data_source import SceneDataSource
from alpasim_utils.scene_metadata import Metadata

logger = logging.getLogger(__name__)

try:
    from alpasim_utils.ground_snap import snap_vector_map_to_ground
    from trajdata.dataset_specific.mads.mads_utils import populate_vector_map
    from trajdata.dataset_specific.xodr.geo_transform import get_t_rig_enu_from_ecef
    from trajdata.dataset_specific.xodr.vector_map_export import (
        populate_vector_map_from_xodr,
    )
    from trajdata.maps import VectorMap
    from trajdata.maps.vec_map_elements import MapElementType
except ImportError:
    logger.warning("Could not import trajdata (missing). Map loading will be disabled.")
    VectorMap = None

    def populate_vector_map(dum_one, dum_two):
        """
        Dummy function to avoid ImportError when trajdata is not installed.
        """
        raise FileNotFoundError(
            "Map loading is disabled because trajdata is not installed."
        )

    def get_t_rig_enu_from_ecef(dum_one, dum_two):
        """
        Dummy function to avoid ImportError when trajdata is not installed.
        """
        raise FileNotFoundError(
            "XODR coordinate transformation is disabled because trajdata is not installed."
        )

    def populate_vector_map_from_xodr(dum_one, dum_two, **kwargs):
        """
        Dummy function to avoid ImportError when trajdata is not installed.
        """
        raise FileNotFoundError(
            "XODR map loading is disabled because trajdata is not installed."
        )


@dataclass
class Artifact(SceneDataSource):
    source: str
    use_ground_mesh: bool = False

    # for caching
    _metadata: Metadata | None = None
    _rig: Rig | None = None
    _traffic_objects: TrafficObjects | None = None
    _smooth_trajectories: bool = True
    _map: VectorMap | None = None
    _map_source: str | None = None
    _attempted_map_load: bool = False
    _mesh_ply: bytes | None = None

    # Thread-safety lock for lazy-loaded properties (especially map loading)
    # Using field(default_factory=...) to create a lock per instance
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        assert self.source.endswith(".usdz")

    def __getstate__(self) -> dict:
        """Return state for pickling, excluding the non-picklable lock."""
        state = self.__dict__.copy()
        # Remove the lock from the state - it cannot be pickled
        del state["_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling, recreating the lock."""
        self.__dict__.update(state)
        # Recreate the lock in the new process
        self._lock = threading.Lock()

    def clear_cache(self) -> None:
        """Clear cached data"""
        with self._lock:
            self._metadata = None
            self._rig = None
            self._traffic_objects = None
            self._map = None
            self._map_source = None
            self._attempted_map_load = False
            self._mesh_ply = None

    @staticmethod
    def discover_from_glob(
        glob_query: str,
        recursive: bool = True,
        smooth_trajectories: bool = True,
        use_ground_mesh: bool = False,
    ) -> dict[str, Artifact]:
        """
        A factory method to create artifact instances
        """
        assert glob_query.endswith(
            ".usdz"
        ), f"glob query needs to end in .usdz to find valid artifacts (got {glob_query=})."
        artifacts = {
            path: Artifact(
                path,
                _smooth_trajectories=smooth_trajectories,
                use_ground_mesh=use_ground_mesh,
            )
            for path in glob.glob(glob_query, recursive=recursive)
        }

        scene_id_to_artifact_paths: dict[str, list[str]] = {}
        for path, artifact in artifacts.items():
            scene_id_to_artifact_paths.setdefault(artifact.scene_id, []).append(path)

        duplicates = {
            artifact: paths
            for artifact, paths in scene_id_to_artifact_paths.items()
            if len(paths) >= 2
        }
        if duplicates:
            raise AssertionError(
                f"Duplicate scene IDs found. Duplicates (scene_id: artifact paths): {duplicates}."
            )

        return {artifact.scene_id: artifact for artifact in artifacts.values()}

    @property
    def metadata(self) -> Metadata:
        if self._metadata is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._metadata = Metadata.from_dict(
                    yaml.safe_load(zip_file.open("metadata.yaml"))
                )

        return self._metadata

    @property
    def rig(self) -> Rig:
        if self._rig is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                json_str = zip_file.open("rig_trajectories.json").read().decode("utf-8")
                (self._rig,) = Rig.load_from_json(json_str)  # for now there's only one

        return self._rig

    @property
    def traffic_objects(self) -> TrafficObjects:
        if self._traffic_objects is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                json_str = zip_file.open("sequence_tracks.json").read().decode("utf-8")
                sequence_id_to_traffic_objects = TrafficObjects.load_from_json(
                    json_str, smooth=self._smooth_trajectories
                )
                assert (
                    len(sequence_id_to_traffic_objects) == 1
                )  # we don't support multi-sequence reconstructions yet
                (self._traffic_objects,) = sequence_id_to_traffic_objects.values()

        return self._traffic_objects

    @property
    def scene_id(self) -> str:
        """
        The name used to identify the scene via `scene_id=` when requesting
        """
        return self.metadata.scene_id

    @property
    def map(self) -> VectorMap | None:
        """Load and return the map data from the USDZ file.

        Attempts to load map data in the following order:
        1. map_data or clipgt parquet directories
        2. fastmap parquet directory
        3. XODR file

        Returns:
            VectorMap instance or None if no map data is available
        """
        if VectorMap is None:
            logger.warning(
                "Map loading is disabled because trajdata is not installed. "
                "Install trajdata to enable map loading."
            )
            return None

        # Fast path: if already loaded, return cached result (no lock needed)
        if self._attempted_map_load:
            return self._map

        # Slow path: acquire lock to ensure thread-safe lazy loading
        with self._lock:
            # Double-check after acquiring lock (another thread may have loaded it)
            if self._attempted_map_load:
                return self._map

            logger.info(
                "Loading USDZ map data into memory. This will take a few seconds..."
            )

            vector_map = None
            map_directory = None
            with zipfile.ZipFile(self.source, "r") as zip_file:
                for candidate in ("map_data", "clipgt", "fastmap"):
                    vector_map = self._load_parquet_map(zip_file, candidate)
                    if vector_map is not None:
                        map_directory = candidate
                        logger.info("Successfully loaded map from %s", candidate)
                        break
                if vector_map is None:
                    vector_map = self._load_xodr_map(zip_file)
                    if vector_map is not None:
                        map_directory = "xodr"
                        logger.info("Successfully loaded map from XODR")

            if vector_map is None:
                logger.warning(
                    f"No map data (map_data, clipgt, fastmap, or XODR) found in {self.source}. "
                    "Skipping map loading."
                )
                self._map = None
                self._map_source = None
                # Mark as attempted AFTER setting _map to None (load complete)
                self._attempted_map_load = True
                return None

            self._map = vector_map
            if map_directory == "fastmap":
                self._snap_map_z()

            # Post-process the loaded map (builds KDTree search indices)
            self._finalize_map()
            self._map_source = map_directory

            # Mark as attempted AFTER map is fully loaded and finalized
            self._attempted_map_load = True
            return self._map

    @property
    def map_source(self) -> str | None:
        """Return the source used by :attr:`map`.

        Values are ``map_data``, ``clipgt``, ``fastmap``, or ``xodr``. The
        property loads the map if needed and returns ``None`` when no usable
        map is available.
        """
        _ = self.map
        return self._map_source

    def _load_parquet_map(
        self, zip_file: zipfile.ZipFile, map_directory: str
    ) -> VectorMap | None:
        """Load a complete parquet map directory.

        Args:
            zip_file: Open ZipFile instance
            map_directory: Top-level directory containing map parquet files

        Returns:
            The loaded map, or None if this directory cannot be loaded
        """
        prefix = f"{map_directory}/"
        members = [name for name in zip_file.namelist() if name.startswith(prefix)]
        if not members:
            return None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_file.extractall(temp_dir, members)
                map_root = os.path.join(temp_dir, map_directory)
                vector_map = VectorMap(map_id=f"alpasim_usdz:{self.metadata.scene_id}")
                populate_vector_map(vector_map, map_root)
                if not vector_map.elements.get(MapElementType.ROAD_LANE):
                    logger.warning("Could not load map from %s: no road lanes", prefix)
                    return None
                return vector_map
        except (FileNotFoundError, ValueError, AttributeError) as e:
            logger.warning("Could not load map from %s: %s", prefix, e)
            return None

    def _load_xodr_map(self, zip_file: zipfile.ZipFile) -> VectorMap | None:
        """Load map from XODR file.

        Args:
            zip_file: Open ZipFile instance

        Returns:
            The loaded map, or None if the archive has no usable XODR map
        """
        try:
            # Open XODR file
            with zip_file.open("map.xodr", "r") as xodr_file:
                xodr_xml = xodr_file.read().decode("utf-8")

            # Get coordinate transformation if available
            t_xodr_enu_to_sim = self._get_xodr_transform(zip_file, xodr_xml)

            # Load the XODR map
            vector_map = VectorMap(map_id=f"alpasim_usdz:{self.metadata.scene_id}")
            populate_vector_map_from_xodr(
                vector_map, xodr_xml, t_xodr_enu_to_sim=t_xodr_enu_to_sim
            )
            return vector_map
        except (KeyError, FileNotFoundError) as e:
            logger.debug(f"Could not load XODR map: {e}")
            return None

    def _get_xodr_transform(
        self, zip_file: zipfile.ZipFile, xodr_xml: str
    ) -> np.ndarray | None:
        """Get coordinate transformation matrix for XODR map.

        Transforms map from OpenDRIVE ENU to Simulation space for trajectory alignment.

        Args:
            zip_file: Open ZipFile instance
            xodr_xml: XODR XML content

        Returns:
            4x4 transformation matrix from XODR ENU to Simulation space

        Raises:
            RuntimeError: If rig_trajectories.json is missing or transformation
                         cannot be computed
        """
        # Check if trajectory data exists
        if "rig_trajectories.json" not in zip_file.namelist():
            # For simulation, trajectory data is required for proper coordinate alignment
            # between the neural reconstruction space and the OpenDRIVE map
            raise RuntimeError(
                "Missing rig_trajectories.json: Cannot compute XODR ENU to Simulation "
                "coordinate transformation. This transformation is required for simulation to "
                "ensure map and trajectory coordinates align."
            )

        try:
            with zip_file.open("rig_trajectories.json", "r") as rig_file:
                import json

                rig_data = json.load(rig_file)

            t_world_base = np.asarray(rig_data.get("T_world_base"))
            if t_world_base.shape != (4, 4):
                raise ValueError(
                    f"Invalid T_world_base shape: expected (4, 4), got {t_world_base.shape}"
                )

            # Apply Simulation coordinate transform: Map ENU → Simulation space
            t_sim_map = get_t_rig_enu_from_ecef(t_world_base, xodr_xml)
            t_xodr_enu_to_sim = np.linalg.inv(t_sim_map)
            logger.info(
                "Applied XODR ENU to Simulation coordinate transform for trajectory alignment"
            )
            return t_xodr_enu_to_sim

        except Exception as e:
            logger.warning(
                f"Failed to compute XODR ENU to Simulation transform: {e}. "
                "Map and trajectory coordinates may not align properly, "
                "potentially causing incorrect vehicle positioning and lane associations."
            )
            # Re-raise the exception to ensure users are aware of the potential issue
            raise RuntimeError(
                f"Critical: Unable to compute coordinate transformation for trajectory alignment. "
                f"This will result in misaligned map and trajectory data. Error: {e}"
            ) from e

    def _snap_map_z(self) -> None:
        """Align FastMap geometry with the artifact's ground mesh."""
        assert self._map is not None
        logger.info("Snapping FastMap geometry to the ground mesh...")
        with zipfile.ZipFile(self.source, "r") as zip_file:
            mesh_bytes = zip_file.read("mesh_ground.ply")
        vertices, _ = load_mesh_vf(mesh_bytes)
        snap_vector_map_to_ground(self._map, vertices)

    def _finalize_map(self) -> None:
        """Finalize the loaded map by setting up data structures."""
        self._map.__post_init__()
        self._map.compute_search_indices()

        # Fix data types - trajdata uses lists but we need sets
        for lane in self._map.lanes:
            lane.next_lanes = set(lane.next_lanes)
            lane.prev_lanes = set(lane.prev_lanes)
            lane.adj_lanes_right = set(lane.adj_lanes_right)
            lane.adj_lanes_left = set(lane.adj_lanes_left)

    @property
    def mesh_ply(self) -> bytes:
        if self._mesh_ply is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._mesh_ply = (
                    zip_file.open("mesh.ply", "r").read()
                    if not self.use_ground_mesh
                    else zip_file.open("mesh_ground.ply", "r").read()
                )

        return self._mesh_ply
