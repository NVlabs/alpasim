# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
TrajdataDataSource: Implementation for loading scene data directly from trajdata

This class demonstrates how to create a SceneDataSource implementation that loads
data directly from trajdata converted data without requiring USDZ format. This is
useful for researchers using trajdata datasets.

Usage example:
    from trajdata import UnifiedDataset
    from alpasim_utils.trajdata_data_source import TrajdataDataSource

    # Load trajdata dataset
    dataset = UnifiedDataset(
        desired_data=["nusc_mini"],
        data_dirs={"/path/to/trajdata/data"},
        ...
    )

    # Get a scene
    scene = dataset.get_scene("nusc_mini", "scene-0001")

    # Create data source
    data_source = TrajdataDataSource.from_trajdata_scene(scene)

    # Now can be used in Runtime
    # artifacts = {data_source.scene_id: data_source}
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from trajdata import AgentBatch
    from trajdata.caching import EnvCache
    from trajdata.data_structures.agent import AgentMetadata
    from trajdata.data_structures.scene_metadata import Scene
    from trajdata.dataset import UnifiedDataset
    from trajdata.maps import VectorMap
except ImportError:
    AgentBatch = None
    Scene = None
    UnifiedDataset = None
    EnvCache = None
    AgentMetadata = None
    VectorMap = None

from alpasim_utils.artifact import Metadata
from alpasim_utils.geometry import Trajectory
from alpasim_utils.scenario import (
    AABB,
    CameraId,
    Rig,
    TrafficObject,
    TrafficObjects,
    VehicleConfig,
)
from alpasim_utils.scene_data_source import SceneDataSource

logger = logging.getLogger(__name__)


@dataclass
class TrajdataDataSource(SceneDataSource):
    """
    Implementation for loading scene data directly from trajdata.

    This class implements the SceneDataSource protocol, allowing direct loading
    from trajdata Scene or AgentBatch objects without requiring USDZ format.
    """

    _scene: Scene | None = None
    _scene_cache: EnvCache | None = None
    _dataset: UnifiedDataset | None = None
    _rig: Rig | None = None
    _traffic_objects: TrafficObjects | None = None
    _map: VectorMap | None = None
    _metadata: Metadata | None = None
    _smooth_trajectories: bool = True
    _scene_id: str = ""
    _asset_base_path: str | None = None  # Base path for rendering assets

    @classmethod
    def from_trajdata_scene(
        cls,
        scene: Scene,
        dataset: Optional[UnifiedDataset] = None,
        scene_cache: Optional[EnvCache] = None,
        scene_id: Optional[str] = None,
        smooth_trajectories: bool = True,
        base_timestamp_us: int = 0,
        asset_base_path: Optional[str] = None,
    ) -> TrajdataDataSource:
        """
        Create TrajdataDataSource from trajdata Scene object.

        Args:
            scene: trajdata Scene object
            dataset: UnifiedDataset instance (for getting scene_cache and map)
            scene_cache: Optional EnvCache instance (if not provided, will be created from dataset)
            scene_id: Optional scene ID (if not provided, uses scene.name)
            smooth_trajectories: Whether to smooth trajectories
            base_timestamp_us: Base timestamp in microseconds, starts from 0 if None
            asset_base_path: Base path for rendering assets (e.g., MTGS assets)

        Returns:
            TrajdataDataSource instance
        """
        if Scene is None:
            raise ImportError(
                "trajdata is not installed. Please install it to use TrajdataDataSource."
            )

        data_source = cls(
            _scene=scene,
            _dataset=dataset,
            _scene_cache=scene_cache,
            _scene_id=scene_id or scene.name,
            _smooth_trajectories=smooth_trajectories,
            _asset_base_path=asset_base_path,
        )
        data_source._base_timestamp_us = base_timestamp_us
        return data_source

    @classmethod
    def from_agent_batch(
        cls,
        batch: AgentBatch,
        scene_id: str,
        smooth_trajectories: bool = True,
    ) -> TrajdataDataSource:
        """
        Create TrajdataDataSource from trajdata AgentBatch object.

        Note: This method requires the batch to contain complete scene information.
        It is generally recommended to use from_trajdata_scene instead.

        Args:
            batch: trajdata AgentBatch object
            scene_id: Scene ID
            smooth_trajectories: Whether to smooth trajectories

        Returns:
            TrajdataDataSource instance
        """
        if AgentBatch is None:
            raise ImportError(
                "trajdata is not installed. Please install it to use TrajdataDataSource."
            )

        data_source = cls(scene_id=scene_id)
        data_source._smooth_trajectories = smooth_trajectories
        # Extract data from batch
        data_source._load_from_batch(batch)
        return data_source

    def _load_from_batch(self, batch: AgentBatch) -> None:
        """Load data from AgentBatch (internal method)"""
        # Need to extract data based on batch structure
        # Specific implementation depends on your trajdata data format
        raise NotImplementedError(
            "from_agent_batch needs to be implemented based on your trajdata data format. "
            "It is recommended to use from_trajdata_scene method instead."
        )

    @property
    def scene_id(self) -> str:
        """Scene ID"""
        if self._scene_id:
            return self._scene_id
        if self._scene is not None:
            return self._scene.name
        raise ValueError("scene_id is not set and cannot be obtained from scene")

    @scene_id.setter
    def scene_id(self, value: str) -> None:
        self._scene_id = value

    @property
    def asset_path(self) -> str | None:
        """
        Resolve asset folder path for this scene.

        The asset path is constructed by appending the scene name to _asset_base_path.
        The _asset_base_path should already contain any dataset-specific subdirectories
        (e.g., it might be /data/WE_processed/navtest/assets for MTGS).

        Returns:
            Resolved asset folder path, or None if _asset_base_path is not set
        """
        if self._asset_base_path is None:
            return None

        # Extract asset folder name from scene metadata
        scene_name = self._extract_asset_folder_name()

        # Simple join: asset_base_path already contains dataset-specific subdirs
        return os.path.join(self._asset_base_path, scene_name)

    def _extract_asset_folder_name(self) -> str:
        """
        Extract the asset folder name from scene metadata.

        This method attempts to determine the appropriate asset folder name
        based on scene metadata. Override this in subclasses if needed.

        Resolution order:
        1. USDZ: Use usdz_stem from data_access_info
        2. Other datasets: Use log_id or asset_folder from data_access_info
        3. Fallback: Use scene_id with common suffixes removed

        Returns:
            Asset folder name (defaults to scene_id if no specific name found)
        """
        # Try to get from scene data_access_info
        if self._scene is not None and hasattr(self._scene, "data_access_info"):
            data_access_info = self._scene.data_access_info or {}

            # USDZ: Use usdz_stem (filename without .usdz extension)
            if "usdz_stem" in data_access_info:
                return data_access_info["usdz_stem"]

            # Look for asset_folder or similar keys
            if "asset_folder" in data_access_info:
                return data_access_info["asset_folder"]

            # NuPlan and other datasets: use log_id
            if "log_id" in data_access_info:
                return data_access_info["log_id"]

        # Default: use scene_id (potentially with suffix removed)
        scene_name = self.scene_id
        # Remove common suffixes like "-001"
        if len(scene_name) > 4 and scene_name[-4] == "-" and scene_name[-3:].isdigit():
            scene_name = scene_name[:-4]
        return scene_name

    def set_asset_base_path(self, path: str | None) -> None:
        """Set the base path for rendering assets."""
        self._asset_base_path = path

    def _get_scene_cache(self) -> EnvCache:
        """Get or create scene_cache"""
        if self._scene_cache is not None:
            return self._scene_cache

        if self._scene is None:
            raise ValueError("Cannot create scene_cache: scene is not set")

        # Try to create from dataset if available
        if self._dataset is not None:
            logger.debug(f"Creating scene_cache for scene: {self._scene.name}")
            try:
                self._scene_cache = self._dataset.cache_class(
                    self._dataset.cache_path, self._scene, self._dataset.augmentations
                )
                self._scene_cache.set_obs_format(self._dataset.obs_format)
                logger.debug("Scene cache created successfully")
                return self._scene_cache
            except Exception as e:
                logger.error(f"Failed to create scene_cache: {e}")
                raise

        # If dataset is not set, scene_cache must be provided externally
        raise ValueError(
            "Cannot create scene_cache: dataset is not set and scene_cache was not provided. "
            "Either pass 'dataset' parameter or pre-create 'scene_cache' when creating TrajdataDataSource. "
            "Example: TrajdataDataSource.from_trajdata_scene(scene, dataset=your_dataset) "
            "or TrajdataDataSource.from_trajdata_scene(scene, scene_cache=your_cache)"
        )

    def _extract_agent_trajectory(
        self,
        agent: AgentMetadata,
    ) -> tuple[Optional[Trajectory], Optional[VehicleConfig]]:
        """Extract complete trajectory for agent (refer to trajdata_artifact_converter.py implementation)"""
        if self._scene is None:
            return None, None

        scene_cache = self._get_scene_cache()
        dt = self._scene.dt
        base_timestamp_us = getattr(self, "_base_timestamp_us", None)

        try:
            timestamps_us = []
            positions_agent_world = []
            quaternions_agent_world = []

            # Iterate through all timesteps
            for ts in range(agent.first_timestep, agent.last_timestep + 1):
                try:
                    state = scene_cache.get_raw_state(agent.name, ts)

                    # Get position and orientation
                    x = state.get_attr("x") if hasattr(state, "get_attr") else state.x
                    y = state.get_attr("y") if hasattr(state, "get_attr") else state.y
                    z = (
                        state.get_attr("z")
                        if hasattr(state, "get_attr")
                        else (state.z if hasattr(state, "z") else 0.0)
                    )
                    heading = (
                        state.get_attr("h") if hasattr(state, "get_attr") else state.h
                    )

                    # Convert to numpy array (handle scalar case)
                    if isinstance(x, (int, float)):
                        x = np.array([x])
                    if isinstance(y, (int, float)):
                        y = np.array([y])
                    if isinstance(z, (int, float)):
                        z = np.array([z])
                    if isinstance(heading, (int, float)):
                        heading = np.array([heading])

                    # Take first element (if array)
                    x_val = float(x[0] if x.ndim > 0 else x)
                    y_val = float(y[0] if y.ndim > 0 else y)
                    z_val = float(z[0] if z.ndim > 0 else z)
                    heading_val = float(heading[0] if heading.ndim > 0 else heading)

                    # Calculate timestamp
                    if base_timestamp_us is None:
                        timestamp_us = int(ts * dt * 1e6)
                    else:
                        timestamp_us = int(base_timestamp_us + ts * dt * 1e6)

                    timestamps_us.append(timestamp_us)
                    positions_agent_world.append([x_val, y_val, z_val])

                    # Convert heading to quaternion
                    quat = R.from_euler("z", heading_val).as_quat()  # [x, y, z, w]
                    quaternions_agent_world.append(quat)

                except Exception as e:
                    logger.debug(
                        f"Failed to get state for agent {agent.name} at ts {ts}: {e}"
                    )
                    continue

            if len(timestamps_us) == 0:
                return None, None

            # Create Trajectory
            trajectory = Trajectory(
                timestamps=np.array(timestamps_us, dtype=np.uint64),
                positions=np.array(positions_agent_world, dtype=np.float32),
                quaternions=np.array(quaternions_agent_world, dtype=np.float32),
            )

            # Create VehicleConfig (extract from extent)
            vehicle_config = None
            if hasattr(agent.extent, "length"):
                vehicle_config = VehicleConfig(
                    aabb_x_m=agent.extent.length,
                    aabb_y_m=agent.extent.width,
                    aabb_z_m=agent.extent.height,
                    aabb_x_offset_m=-agent.extent.length / 2,
                    aabb_y_offset_m=0.0,
                    aabb_z_offset_m=-agent.extent.height / 2,
                )

            return trajectory, vehicle_config

        except Exception as e:
            logger.error(f"Failed to extract trajectory for agent {agent.name}: {e}")
            return None, None

    @property
    def rig(self) -> Rig:
        """Load and return Rig object for ego vehicle"""
        if self._rig is not None:
            return self._rig

        if self._scene is None:
            raise ValueError("Cannot load rig: scene is not set")

        # Get all agents
        all_agents = self._scene.agents if self._scene.agents else []

        # Identify ego agent
        ego_agent = next((a for a in all_agents if a.name == "ego"), None)
        if ego_agent is None and len(all_agents) > 0:
            # If no ego, use first agent
            ego_agent = all_agents[0]
            logger.warning(f"No ego agent found, using first agent: {ego_agent.name}")

        if ego_agent is None:
            raise ValueError("No ego agent found in scene")

        # Extract ego trajectory
        ego_trajectory, ego_vehicle_config = self._extract_agent_trajectory(ego_agent)

        if ego_trajectory is None:
            logger.error(
                f"Failed to extract ego trajectory for agent {ego_agent.name}. "
                f"Check if scene_cache is properly initialized and agent data is available."
            )
            raise ValueError("Cannot extract ego trajectory")

        # Calculate world_to_nre transformation matrix (use first trajectory point as origin)
        world_to_nre = np.eye(4)
        if len(ego_trajectory) > 0:
            position_ego_first_world = ego_trajectory.positions[0]
            world_to_nre[:3, 3] = -position_ego_first_world
            logger.info(
                f"Setting world_to_nre origin at first pose: {position_ego_first_world}, "
                f"translation: {world_to_nre[:3, 3]}"
            )

        # Convert ego trajectory to local coordinates (NRE)
        if len(ego_trajectory) > 0:
            translation = world_to_nre[:3, 3]
            local_positions = ego_trajectory.positions + translation

            # Validate transform
            position_ego_first_local = local_positions[0]
            if np.linalg.norm(position_ego_first_local[:2]) > 1.0:
                logger.warning(
                    f"First pose after transformation is not at origin: {position_ego_first_local}. "
                    f"Expected [0, 0, ~z], got {position_ego_first_local}"
                )

            local_quat = ego_trajectory.quaternions.copy()
            ego_trajectory = Trajectory(
                timestamps=ego_trajectory.timestamps_us.copy(),
                positions=local_positions,
                quaternions=local_quat,
            )

            logger.debug(
                f"Transformed ego trajectory to local coordinates. "
                f"First pose: {ego_trajectory.first_pose}, "
                f"Range: X[{local_positions[:, 0].min():.2f}, {local_positions[:, 0].max():.2f}], "
                f"Y[{local_positions[:, 1].min():.2f}, {local_positions[:, 1].max():.2f}], "
                f"Z[{local_positions[:, 2].min():.2f}, {local_positions[:, 2].max():.2f}]"
            )

        # Extract camera information (refer to trajdata_artifact_converter.py)
        camera_ids, _ = self._extract_camera_info_from_scene()

        self._rig = Rig(
            sequence_id=self.scene_id,
            trajectory=ego_trajectory,
            camera_ids=camera_ids,
            world_to_nre=world_to_nre,
            vehicle_config=ego_vehicle_config,
        )

        return self._rig

    def _extract_camera_info_from_scene(self) -> tuple[list[CameraId], dict]:
        """Extract camera information from scene (refer to trajdata_artifact_converter.py)"""
        camera_ids = []
        camera_calibrations = {}

        if self._scene is None:
            return camera_ids, camera_calibrations

        # Check if sensor_calibration information exists
        if (
            not hasattr(self._scene, "data_access_info")
            or not self._scene.data_access_info
        ):
            logger.warning(
                "scene.data_access_info does not exist, skipping camera information extraction"
            )
            return camera_ids, camera_calibrations

        sensor_calibration = self._scene.data_access_info.get("sensor_calibration")
        if not sensor_calibration or not isinstance(sensor_calibration, dict):
            logger.warning(
                "sensor_calibration does not exist or has incorrect format, skipping camera information extraction"
            )
            return camera_ids, camera_calibrations

        unique_sensor_idx = 0
        for camera_name, calibration_info in sensor_calibration.get(
            "cameras", {}
        ).items():
            try:
                unique_camera_id = f"{camera_name}@{self.scene_id}"

                position_sensor_to_ego = calibration_info.get(
                    "sensor2ego_translation", [0.0, 0.0, 0.0]
                )
                rotation_sensor_to_ego = calibration_info.get(
                    "sensor2ego_rotation", [0.0, 0.0, 0.0, 1.0]
                )

                if isinstance(position_sensor_to_ego, (int, float)):
                    position_sensor_to_ego = [float(position_sensor_to_ego), 0.0, 0.0]
                elif len(position_sensor_to_ego) < 3:
                    position_sensor_to_ego = list(position_sensor_to_ego) + [0.0] * (
                        3 - len(position_sensor_to_ego)
                    )

                if isinstance(rotation_sensor_to_ego, (int, float)):
                    rotation_sensor_to_ego = [0.0, 0.0, 0.0, 1.0]
                elif len(rotation_sensor_to_ego) < 4:
                    if len(rotation_sensor_to_ego) == 3:
                        r = R.from_euler("xyz", rotation_sensor_to_ego)
                        rotation_sensor_to_ego = r.as_quat()
                    else:
                        rotation_sensor_to_ego = [0.0, 0.0, 0.0, 1.0]

                camera_id = CameraId(
                    logical_name=camera_name,
                    trajectory_idx=0,
                    sequence_id=self.scene_id,
                    unique_id=unique_camera_id,
                )
                camera_ids.append(camera_id)
                unique_sensor_idx += 1

            except Exception as e:
                logger.warning(
                    f"Error extracting camera {camera_name} information: {e}"
                )
                continue

        if len(camera_ids) == 0:
            # If no camera information, create a default one
            logger.warning(
                f"Scene {self.scene_id} has no camera information, using default camera"
            )
            camera_ids.append(
                CameraId(
                    logical_name="camera_front",
                    trajectory_idx=0,
                    sequence_id=self.scene_id,
                    unique_id="0@camera_front",
                )
            )

        return camera_ids, camera_calibrations

    def _is_static_object(
        self, trajectory: Trajectory, velocity_threshold: float = 0.1
    ) -> bool:
        """Determine if object is static (based on velocity)"""
        if len(trajectory) < 2:
            return True

        positions = trajectory.positions
        timestamps = trajectory.timestamps_us.astype(np.float64) / 1e6

        velocities = []
        for i in range(1, len(positions)):
            dt_sec = timestamps[i] - timestamps[i - 1]
            if dt_sec > 0:
                displacement = np.linalg.norm(positions[i] - positions[i - 1])
                velocity = displacement / dt_sec
                velocities.append(velocity)

        if len(velocities) == 0:
            return True

        avg_velocity = np.mean(velocities)
        return avg_velocity < velocity_threshold

    @property
    def traffic_objects(self) -> TrafficObjects:
        """Load and return traffic objects"""
        if self._traffic_objects is not None:
            return self._traffic_objects

        if self._scene is None:
            raise ValueError("Cannot load traffic_objects: scene is not set")

        # Get all agents
        all_agents = self._scene.agents if self._scene.agents else []

        # Identify ego agent
        ego_agent = next((a for a in all_agents if a.name == "ego"), None)
        if ego_agent is None and len(all_agents) > 0:
            ego_agent = all_agents[0]

        traffic_dict = {}
        for agent in all_agents:
            # Skip ego agent
            if agent.name == "ego" or agent == ego_agent:
                continue

            # Extract trajectory
            trajectory, _ = self._extract_agent_trajectory(agent)

            # Filter out empty trajectories or trajectories with only 1 data point
            if trajectory is None or len(trajectory) < 2:
                continue

            # Convert trajectory to local coordinates (NRE) - use rig's world_to_nre
            if self._rig is None:
                # If rig is not loaded yet, load it first
                _ = self.rig

            world_to_nre = self._rig.world_to_nre
            translation = world_to_nre[:3, 3]
            local_positions = trajectory.positions + translation
            local_quat = trajectory.quaternions.copy()
            trajectory = Trajectory(
                timestamps=trajectory.timestamps_us.copy(),
                positions=local_positions,
                quaternions=local_quat,
            )

            # Smooth if needed
            if self._smooth_trajectories:
                try:
                    import csaps

                    css = csaps.CubicSmoothingSpline(
                        trajectory.timestamps_us / 1e6,
                        trajectory.positions.T,
                        normalizedsmooth=True,
                    )
                    filtered_positions = css(trajectory.timestamps_us / 1e6).T
                    max_error = np.max(
                        np.abs(filtered_positions - trajectory.positions)
                    )
                    if max_error > 1.0:
                        logger.warning(
                            f"Max error in cubic spline approximation: {max_error:.6f} m for {agent.name=}"
                        )
                    # Create new trajectory with smoothed positions
                    trajectory = Trajectory(
                        timestamps=trajectory.timestamps_us.copy(),
                        positions=filtered_positions.astype(np.float32),
                        quaternions=trajectory.quaternions.copy(),
                    )
                except ImportError:
                    logger.warning("csaps not installed, skipping trajectory smoothing")

            # Get AABB
            if hasattr(agent.extent, "length"):
                aabb = AABB(
                    x=agent.extent.length, y=agent.extent.width, z=agent.extent.height
                )
            else:
                # Default AABB
                aabb = AABB(x=4.5, y=1.8, z=1.5)

            # Determine if static object
            is_static = self._is_static_object(trajectory)

            # Get category label
            label_class = agent.type.name if hasattr(agent.type, "name") else "UNKNOWN"

            traffic_dict[agent.name] = TrafficObject(
                track_id=agent.name,
                aabb=aabb,
                trajectory=trajectory,
                is_static=is_static,
                label_class=label_class,
            )

        self._traffic_objects = TrafficObjects(**traffic_dict)
        return self._traffic_objects

    @property
    def map(self) -> Optional[VectorMap]:
        """Load and return VectorMap (obtained from dataset._map_api or scene.map_data)"""
        if self._map is not None:
            return self._map

        if VectorMap is None:
            logger.warning("trajdata is not installed, cannot load map")
            return None

        if self._scene is None:
            logger.warning("Cannot load map: scene is not set")
            return None

        # First, try to get map from scene.map_data (for USDZ and other datasets that attach map directly)
        if hasattr(self._scene, "map_data") and self._scene.map_data is not None:
            logger.info(f"Loading map from scene.map_data for {self.scene_id}")
            # Make a deep copy to avoid modifying shared map object
            self._map = copy.deepcopy(self._scene.map_data)

            # Apply coordinate transformation if needed
            if self._rig is None:
                # If rig is not loaded yet, load it first (this will set world_to_nre)
                _ = self.rig

            world_to_nre = self._rig.world_to_nre

            # Check if transformation is needed (if world_to_nre is not identity)
            if world_to_nre is not None and not np.allclose(world_to_nre, np.eye(4)):
                translation = world_to_nre[:3, 3]
                logger.info(
                    f"Transforming map to local coordinates with translation: {translation}"
                )

                # Transform map coordinates
                self._map.translate(translation[0], translation[1], translation[2])

            logger.info(
                f"Successfully loaded map from scene.map_data for {self.scene_id}"
            )
            return self._map

        # Otherwise, try to get map from dataset._map_api (for datasets with map_api)
        if self._dataset is None:
            logger.warning(
                "Cannot load map: dataset is not set and scene.map_data is not available"
            )
            return None

        # Get map from dataset._map_api (refer to trajdata_artifact_converter.py)
        try:
            # Check if dataset includes map support
            if (
                not hasattr(self._dataset, "incl_vector_map")
                or not self._dataset.incl_vector_map
                or not hasattr(self._dataset, "_map_api")
                or self._dataset._map_api is None
            ):
                logger.warning(
                    "Dataset does not have map support enabled or map_api is unavailable"
                )
                return None

            # Build map name: "{env_name}:{location}"
            if not hasattr(self._scene, "location") or not self._scene.location:
                logger.warning(
                    f"Scene {self.scene_id} has no location information, cannot load map"
                )
                return None

            map_name = f"{self._scene.env_name}:{self._scene.location}"

            # Get vector_map_params (if exists)
            vector_map_params = {}
            if hasattr(self._dataset, "vector_map_params"):
                vector_map_params = self._dataset.vector_map_params

            # Get map from map_api
            vec_map = self._dataset._map_api.get_map(map_name, **vector_map_params)

            if vec_map is None:
                logger.debug(
                    f"Scene {self.scene_id} (map_name: {map_name}) has no map data"
                )
                return None

            # Create an independent copy of VectorMap for current scene to avoid modifying
            # map objects in shared cache. This allows continued use of MapAPI's disk cache
            # and index loading capabilities while preventing coordinate system pollution
            # between multiple scenes through shared VectorMap instances.
            self._map = copy.deepcopy(vec_map)

            # Important: Transform map to local coordinate system (NRE)
            # Since trajectories are already converted to local coordinates, map also needs
            # to be converted to match. This is consistent with USDZ format handling:
            # both map and trajectories need to be converted to the same coordinate system
            if self._rig is None:
                # If rig is not loaded yet, load it first (this will set world_to_nre)
                _ = self.rig

            world_to_nre = self._rig.world_to_nre

            # Check if world_to_nre contains rotation
            rotation_matrix = world_to_nre[:3, :3]
            translation = world_to_nre[:3, 3]
            has_rotation = not np.allclose(rotation_matrix, np.eye(3))

            if has_rotation:
                logger.warning(
                    "world_to_nre contains rotation. Map transformation may need rotation handling. "
                    "Currently only applying translation."
                )

            # Transform all points in map (center.points, left_boundary.points, right_boundary.points, etc.)
            # Note: Must transform before finalize, as finalize may rebuild certain data structures
            # Important: Only transform X, Y coordinates, align Z coordinate to trajectory Z baseline
            # Because map Z usually represents height relative to ground (usually 0), while trajectory Z
            # represents altitude. After transformation, map Z should align with trajectory Z
            # (both relative to first trajectory point's Z, usually 0 after transformation)
            translation_xy = translation[:2]  # Only use X, Y translation

            # Get Z coordinate of first trajectory point (transformed baseline, usually 0)
            first_traj_z = (
                self.rig.trajectory.positions[0][2]
                if len(self.rig.trajectory) > 0
                else 0.0
            )

            logger.info(
                f"Map coordinate transformation: "
                f"translation_xy={translation_xy}, "
                f"first_traj_z={first_traj_z:.2f}m, "
                f"map Z will be aligned to trajectory Z baseline"
            )

            def transform_map_points(points: np.ndarray) -> np.ndarray:
                """Transform map points: only transform X, Y, align Z to trajectory Z baseline"""
                if (
                    points is None
                    or len(points) == 0
                    or points.ndim != 2
                    or points.shape[1] < 3
                ):
                    return points

                points_copy = points.copy()

                # Transform X, Y coordinates
                if has_rotation:
                    # If rotation exists, need to rotate X, Y
                    xy_rotated = (
                        points_copy[:, :2] @ rotation_matrix[:2, :2].T
                    ) + translation_xy
                    points_copy[:, 0] = xy_rotated[:, 0]
                    points_copy[:, 1] = xy_rotated[:, 1]
                else:
                    # Only translate X, Y
                    points_copy[:, 0] = points_copy[:, 0] + translation_xy[0]
                    points_copy[:, 1] = points_copy[:, 1] + translation_xy[1]

                # Z coordinate alignment: align map Z to trajectory Z baseline
                # Map Z is usually height relative to ground (usually 0),
                # after transformation should align with trajectory Z (both relative to first
                # trajectory point's Z, usually 0 after transformation)
                # So: new_z = original_z + first_traj_z
                # If map Z=0 (ground), after transformation it becomes first_traj_z (usually 0)
                points_copy[:, 2] = points_copy[:, 2] + first_traj_z

                return points_copy

            if hasattr(self._map, "lanes"):
                for lane_idx, lane in enumerate(self._map.lanes):
                    # Transform center.points
                    if hasattr(lane, "center") and hasattr(lane.center, "points"):
                        points = lane.center.points
                        if points is not None and len(points) > 0:
                            try:
                                transformed_points = transform_map_points(points)
                                lane.center.points = transformed_points
                            except Exception as e:
                                logger.warning(
                                    f"Failed to transform lane {lane_idx} center.points: {e}"
                                )

                    # Transform left_boundary.points
                    if hasattr(lane, "left_boundary") and hasattr(
                        lane.left_boundary, "points"
                    ):
                        points = lane.left_boundary.points
                        if points is not None and len(points) > 0:
                            try:
                                transformed_points = transform_map_points(points)
                                lane.left_boundary.points = transformed_points
                            except Exception as e:
                                logger.warning(
                                    f"Failed to transform lane {lane_idx} left_boundary.points: {e}"
                                )

                    # Transform right_boundary.points
                    if hasattr(lane, "right_boundary") and hasattr(
                        lane.right_boundary, "points"
                    ):
                        points = lane.right_boundary.points
                        if points is not None and len(points) > 0:
                            try:
                                transformed_points = transform_map_points(points)
                                lane.right_boundary.points = transformed_points
                            except Exception as e:
                                logger.warning(
                                    f"Failed to transform lane {lane_idx} right_boundary.points: {e}"
                                )

                    # Transform other possible point attributes (if any)
                    for attr_name in ["intersections", "crosswalks", "stop_lines"]:
                        if hasattr(lane, attr_name):
                            attr_value = getattr(lane, attr_name)
                            if isinstance(attr_value, list):
                                for item in attr_value:
                                    if hasattr(item, "points"):
                                        points = item.points
                                        if points is not None and len(points) > 0:
                                            try:
                                                transformed_points = (
                                                    transform_map_points(points)
                                                )
                                                item.points = transformed_points
                                            except Exception as e:
                                                logger.debug(
                                                    f"Failed to transform {attr_name} points: {e}"
                                                )

            # If map needs finalize, call it (after transformation)
            # Note: finalize may rebuild certain indices but won't change point coordinates
            # But for safety, we verify transformation again after finalize
            if hasattr(self._map, "__post_init__"):
                self._map.__post_init__()
            if hasattr(self._map, "compute_search_indices"):
                self._map.compute_search_indices()

            # Verify again: check if first point's coordinates are still correct after finalize
            if hasattr(self._map, "lanes") and len(self._map.lanes) > 0:
                first_lane = self._map.lanes[0]
                if hasattr(first_lane, "center") and hasattr(
                    first_lane.center, "points"
                ):
                    first_map_point_after_finalize = (
                        first_lane.center.points[0, :3]
                        if len(first_lane.center.points) > 0
                        else None
                    )
                    if first_map_point_after_finalize is not None:
                        # Check if Z coordinate aligns with trajectory (should both be first_traj_z, usually 0)
                        actual_z = first_map_point_after_finalize[2]
                        if abs(actual_z - first_traj_z) > 1.0:  # Allow 1 meter error
                            logger.warning(
                                f"Map Z coordinate may have been reset after finalize. "
                                f"Expected Z≈{first_traj_z:.2f}m (aligned to trajectory Z baseline), "
                                f"got Z={actual_z:.2f}m. "
                                f"This may cause coordinate misalignment."
                            )
                        # If alignment is correct, no need to log (reduce noise)

            # Fix data types (if needed)
            if hasattr(self._map, "lanes"):
                for lane in self._map.lanes:
                    if hasattr(lane, "next_lanes") and isinstance(
                        lane.next_lanes, list
                    ):
                        lane.next_lanes = set(lane.next_lanes)
                    if hasattr(lane, "prev_lanes") and isinstance(
                        lane.prev_lanes, list
                    ):
                        lane.prev_lanes = set(lane.prev_lanes)
                    if hasattr(lane, "adj_lanes_right") and isinstance(
                        lane.adj_lanes_right, list
                    ):
                        lane.adj_lanes_right = set(lane.adj_lanes_right)
                    if hasattr(lane, "adj_lanes_left") and isinstance(
                        lane.adj_lanes_left, list
                    ):
                        lane.adj_lanes_left = set(lane.adj_lanes_left)

            # Verify map transformation: check if first lane's first point is within reasonable range
            if hasattr(self._map, "lanes") and len(self._map.lanes) > 0:
                first_lane = self._map.lanes[0]
                if hasattr(first_lane, "center") and hasattr(
                    first_lane.center, "points"
                ):
                    first_map_point = (
                        first_lane.center.points[0, :3]
                        if len(first_lane.center.points) > 0
                        else None
                    )
                    if first_map_point is not None:
                        # Map point should be near trajectory (within hundreds of meters)
                        distance_from_origin_xy = np.linalg.norm(
                            first_map_point[:2]
                        )  # Only check X, Y
                        distance_from_origin_xyz = np.linalg.norm(
                            first_map_point
                        )  # Check X, Y, Z

                        # Get first trajectory point for comparison
                        first_traj_point = self.rig.trajectory.positions[0]

                        logger.info(
                            f"Map transformation verification: "
                            f"first lane center point: {first_map_point}, "
                            f"first trajectory point: {first_traj_point}, "
                            f"distance (X,Y): {distance_from_origin_xy:.2f}m, "
                            f"distance (X,Y,Z): {distance_from_origin_xyz:.2f}m, "
                            f"Z difference: {abs(first_map_point[2] - first_traj_point[2]):.2f}m"
                        )

                        # Warn if Z coordinate is too far from trajectory Z
                        z_diff = abs(first_map_point[2] - first_traj_point[2])
                        if z_diff > 10.0:  # Z coordinate difference exceeds 10 meters
                            logger.warning(
                                f"Map Z coordinate may not be correctly aligned with trajectory. "
                                f"Map Z={first_map_point[2]:.2f}m, Trajectory Z={first_traj_point[2]:.2f}m, "
                                f"difference={z_diff:.2f}m. This may cause route generation to fail."
                            )

            logger.info(
                f"Successfully loaded map: {map_name} (transformed to local coordinate system)"
            )
            return self._map
        except Exception as e:
            logger.error(f"Error loading map: {e}", exc_info=True)
            return None

    @property
    def metadata(self) -> Metadata:
        """Create and return Metadata object"""
        if self._metadata is not None:
            return self._metadata

        # Extract metadata from scene
        scene_id = self.scene_id

        # Ensure rig is loaded
        rig = self.rig

        # Extract camera ID list from rig
        camera_id_names = []
        if rig and rig.camera_ids:
            camera_id_names = [camera_id.logical_name for camera_id in rig.camera_ids]

        # Calculate time range
        if self._scene is not None:
            dt = self._scene.dt
            length_timesteps = self._scene.length_timesteps
            base_timestamp_us = getattr(self, "_base_timestamp_us", 0.0)
            time_range_start = float(base_timestamp_us) / 1e6
            time_range_end = (
                float(base_timestamp_us + length_timesteps * dt * 1e6) / 1e6
            )
        else:
            time_range_start = float(rig.trajectory.time_range_us.start) / 1e6
            time_range_end = float(rig.trajectory.time_range_us.stop) / 1e6

        # Create metadata
        import uuid
        from datetime import datetime

        self._metadata = Metadata(
            scene_id=scene_id,
            version_string="trajdata_direct",
            training_date=datetime.now().strftime("%Y-%m-%d"),
            dataset_hash=str(uuid.uuid4()),
            uuid=str(uuid.uuid4()),
            is_resumable=False,
            sensors=Metadata.Sensors(
                camera_ids=camera_id_names,
                lidar_ids=[],
            ),
            logger=Metadata.Logger(),
            time_range=Metadata.TimeRange(
                start=time_range_start,
                end=time_range_end,
            ),
        )

        return self._metadata
