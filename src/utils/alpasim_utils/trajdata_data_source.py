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
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Optional

import csaps
import numpy as np
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
from scipy.spatial.transform import Rotation as R
from trajdata.caching import EnvCache
from trajdata.data_structures.agent import AgentMetadata
from trajdata.data_structures.scene_metadata import Scene
from trajdata.dataset import UnifiedDataset
from trajdata.maps import VectorMap

logger = logging.getLogger(__name__)


@dataclass
class TrajdataDataSource(SceneDataSource):
    """
    Implementation for loading scene data directly from trajdata.

    This class implements the SceneDataSource protocol, allowing direct loading
    from trajdata Scene or AgentBatch objects without requiring USDZ format.

    Property loading order dependencies:
    - rig: No dependencies (loads first, sets world_to_nre transformation)
    - traffic_objects: Requires rig (uses world_to_nre for coordinate conversion)
    - map: Requires rig (uses world_to_nre for coordinate conversion)
    - metadata: Requires rig (uses trajectory time range)

    All properties use lazy loading and caching for efficiency.
    """

    _scene: Scene | None = None
    _scene_cache: EnvCache | None = None
    _dataset: UnifiedDataset | None = None
    _map_api: Optional[object] = (
        None  # MapAPI for loading maps (lightweight, can be passed separately)
    )
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
        map_api=None,
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
            map_api: MapAPI instance for loading maps (lightweight alternative to dataset)
            scene_cache: Optional EnvCache instance (if not provided, will be created from dataset)
            scene_id: Optional scene ID (if not provided, uses scene.name)
            smooth_trajectories: Whether to smooth trajectories
            base_timestamp_us: Base timestamp in microseconds, starts from 0 if None
            asset_base_path: Base path for rendering assets (e.g., MTGS assets)

        Returns:
            TrajdataDataSource instance
        """
        data_source = cls(
            _scene=scene,
            _dataset=dataset,
            _map_api=map_api,
            _scene_cache=scene_cache,
            _scene_id=scene_id or scene.name,
            _smooth_trajectories=smooth_trajectories,
            _asset_base_path=asset_base_path,
        )
        data_source._base_timestamp_us = base_timestamp_us
        return data_source

    @property
    def scene_id(self) -> str:
        """Scene ID (immutable identifier)."""
        return self._scene_id

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
        if self._scene is not None and self._scene.data_access_info is not None:
            data_access_info = self._scene.data_access_info

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

    @staticmethod
    def _get_state_value(state, attr_name: str, default=None):
        """Extract scalar value from state attribute using StateArray.get_attr().

        Args:
            state: StateArray object from trajdata cache (from get_raw_state)
            attr_name: Name of attribute to extract (e.g., "x", "y", "h")
            default: Default value if attribute doesn't exist

        Returns:
            Scalar float value

        Raises:
            AttributeError: If attribute doesn't exist and no default provided
        """
        try:
            value = state.get_attr(attr_name)

            if value is None:
                if default is not None:
                    return default
                raise AttributeError(f"Attribute {attr_name} is None")

            # Convert to scalar if needed
            if isinstance(value, np.ndarray):
                return float(value.flat[0])
            return float(value)
        except (KeyError, AttributeError, TypeError, IndexError) as e:
            if default is not None:
                return default
            raise AttributeError(f"Failed to get {attr_name} from state: {e}")

    def _ensure_rig_loaded(self) -> None:
        """Ensure rig is loaded before accessing world_to_nre.

        This method is called by properties that depend on world_to_nre
        transformation (traffic_objects, map helper methods).
        """
        if self._rig is None:
            _ = self.rig

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

                    # Get position and orientation using helper
                    x_val = self._get_state_value(state, "x")
                    y_val = self._get_state_value(state, "y")
                    z_val = self._get_state_value(state, "z", default=0.0)
                    heading_val = self._get_state_value(state, "h")

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
        if not self._scene.data_access_info:
            logger.warning(
                "scene.data_access_info is empty, skipping camera information extraction"
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

    def _transform_map_points(
        self,
        points: np.ndarray,
        translation_xy: np.ndarray,
        first_traj_z: float,
    ) -> np.ndarray:
        """Transform map points to local coordinates (translation only).

        Args:
            points: Nx3 array of map points
            translation_xy: 2-element XY translation
            first_traj_z: Z coordinate of first trajectory point

        Returns:
            Transformed points
        """
        if (
            points is None
            or len(points) == 0
            or points.ndim != 2
            or points.shape[1] < 3
        ):
            return points

        points_copy = points.copy()

        # Apply XY translation
        points_copy[:, 0] = points_copy[:, 0] + translation_xy[0]
        points_copy[:, 1] = points_copy[:, 1] + translation_xy[1]

        # Align Z coordinate to trajectory baseline
        points_copy[:, 2] = points_copy[:, 2] + first_traj_z

        return points_copy

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
            # Explicit dependency: need world_to_nre from rig
            self._ensure_rig_loaded()

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
                except Exception as e:
                    logger.warning(f"Failed to smooth trajectory: {e}")

            # Get AABB
            aabb = AABB(
                x=agent.extent.length, y=agent.extent.width, z=agent.extent.height
            )

            # Determine if static object
            is_static = self._is_static_object(trajectory)

            # Get category label
            label_class = getattr(agent.type, "name", "UNKNOWN")

            traffic_dict[agent.name] = TrafficObject(
                track_id=agent.name,
                aabb=aabb,
                trajectory=trajectory,
                is_static=is_static,
                label_class=label_class,
            )

        self._traffic_objects = TrafficObjects(**traffic_dict)
        return self._traffic_objects

    def _apply_coordinate_transform_to_map(self, vec_map: VectorMap) -> None:
        """Apply world_to_nre coordinate transformation to map in-place.

        Note: world_to_nre only contains translation (no rotation), as the local
        coordinate frame maintains the same orientation as the world frame (ENU).

        Args:
            vec_map: VectorMap to transform
        """
        world_to_nre = self._rig.world_to_nre
        translation = world_to_nre[:3, 3]
        translation_xy = translation[:2]
        first_traj_z = (
            self.rig.trajectory.positions[0][2] if len(self.rig.trajectory) > 0 else 0.0
        )

        logger.info(
            f"Map coordinate transformation: "
            f"translation_xy={translation_xy}, "
            f"first_traj_z={first_traj_z:.2f}m"
        )

        # Transform all lane points
        if vec_map.lanes is None:
            return

        for lane in vec_map.lanes:
            # Transform center (always exists)
            lane.center.points = self._transform_map_points(
                lane.center.points,
                translation_xy,
                first_traj_z,
            )

            # Transform left_edge (optional)
            if lane.left_edge is not None and lane.left_edge.points is not None:
                lane.left_edge.points = self._transform_map_points(
                    lane.left_edge.points,
                    translation_xy,
                    first_traj_z,
                )

            # Transform right_edge (optional)
            if lane.right_edge is not None and lane.right_edge.points is not None:
                lane.right_edge.points = self._transform_map_points(
                    lane.right_edge.points,
                    translation_xy,
                    first_traj_z,
                )

    def _fix_map_datatypes(self, vec_map: VectorMap) -> None:
        """Fix lane connectivity data types (convert lists to sets).

        This is needed because some trajdata loaders may incorrectly create
        these as lists instead of sets.

        Args:
            vec_map: VectorMap to fix
        """
        if vec_map.lanes is None:
            return

        for lane in vec_map.lanes:
            # Convert to set if they are lists (defensive, but based on observed issues)
            if isinstance(lane.next_lanes, list):
                lane.next_lanes = set(lane.next_lanes)
            if isinstance(lane.prev_lanes, list):
                lane.prev_lanes = set(lane.prev_lanes)
            if isinstance(lane.adj_lanes_right, list):
                lane.adj_lanes_right = set(lane.adj_lanes_right)
            if isinstance(lane.adj_lanes_left, list):
                lane.adj_lanes_left = set(lane.adj_lanes_left)

    def _verify_map_transformation(self, vec_map: VectorMap) -> None:
        """Verify map coordinate transformation is correct.

        Args:
            vec_map: Transformed VectorMap
        """
        if vec_map.lanes is None or len(vec_map.lanes) == 0:
            return

        # Get first lane and points (center always exists in RoadLane)
        first_lane = vec_map.lanes[0]
        first_map_point = first_lane.center.points[0, :3]
        first_traj_point = self.rig.trajectory.positions[0]

        distance_xy = np.linalg.norm(first_map_point[:2])
        z_diff = abs(first_map_point[2] - first_traj_point[2])

        logger.info(
            f"Map transformation verification: "
            f"first lane center: {first_map_point}, "
            f"first trajectory: {first_traj_point}, "
            f"XY distance: {distance_xy:.2f}m, "
            f"Z difference: {z_diff:.2f}m"
        )

        if z_diff > 10.0:
            logger.warning(
                f"Map Z coordinate may not be correctly aligned. "
                f"Map Z={first_map_point[2]:.2f}m, Traj Z={first_traj_point[2]:.2f}m"
            )

    def _load_map_from_scene_data(self) -> Optional[VectorMap]:
        """Load map from scene.map_data (USDZ).

        Returns:
            VectorMap if available, None otherwise
        """
        if not hasattr(self._scene, "map_data") or self._scene.map_data is None:
            return None

        logger.info(f"Loading map from scene.map_data for {self.scene_id}")
        vec_map = copy.deepcopy(self._scene.map_data)

        # Ensure rig is loaded (need world_to_nre for transformation)
        self._ensure_rig_loaded()

        # Apply coordinate transformation
        self._apply_coordinate_transform_to_map(vec_map)

        # Fix datatypes and verify
        self._fix_map_datatypes(vec_map)
        self._verify_map_transformation(vec_map)

        logger.info("Successfully loaded map from scene.map_data")
        return vec_map

    def _load_map_from_dataset_api(self) -> Optional[VectorMap]:
        """Load map from dataset._map_api (datasets with map API).

        Returns:
            VectorMap if available, None otherwise
        """
        # Try to get map_api from either self._map_api or self._dataset
        map_api = self._map_api
        if map_api is None and self._dataset is not None:
            map_api = getattr(self._dataset, "_map_api", None)

        if map_api is None:
            logger.warning("Cannot load map: map_api not available")
            return None

        # Get vector_map_params from dataset if available
        vector_map_params = {}
        if self._dataset is not None:
            vector_map_params = getattr(self._dataset, "vector_map_params", {})

        # Build map name
        if not self._scene.location:
            logger.warning(f"Scene {self.scene_id} has no location, cannot load map")
            return None

        map_name = f"{self._scene.env_name}:{self._scene.location}"

        try:
            vec_map = map_api.get_map(map_name, **vector_map_params)
            if vec_map is None:
                logger.debug(f"Scene {self.scene_id} (map: {map_name}) has no map data")
                return None

            # Deep copy to avoid modifying shared cache
            vec_map = copy.deepcopy(vec_map)

            # Ensure rig is loaded (need world_to_nre for transformation)
            self._ensure_rig_loaded()

            # Apply coordinate transformation
            self._apply_coordinate_transform_to_map(vec_map)

            # Finalize map
            vec_map.__post_init__()
            vec_map.compute_search_indices()

            # Fix datatypes and verify
            self._fix_map_datatypes(vec_map)
            self._verify_map_transformation(vec_map)

            logger.info(f"Successfully loaded map: {map_name}")
            return vec_map
        except Exception as e:
            logger.error(f"Error loading map from dataset API: {e}", exc_info=True)
            return None

    @property
    def map(self) -> Optional[VectorMap]:
        """Load and return VectorMap."""
        if self._map is not None:
            return self._map

        if self._scene is None:
            logger.warning("Cannot load map: scene is not set")
            return None

        # Try scene.map_data first (simpler path)
        self._map = self._load_map_from_scene_data()
        if self._map is not None:
            return self._map

        # Fallback to dataset._map_api
        self._map = self._load_map_from_dataset_api()
        return self._map

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

        # Generate deterministic IDs based on scene identifiers
        # Create hash from scene_id and time range for reproducibility
        hash_input = f"{scene_id}_{time_range_start}_{time_range_end}"
        dataset_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        uuid_str = hashlib.sha256(f"{hash_input}_uuid".encode()).hexdigest()[:32]

        # Use fixed training date instead of datetime.now() for determinism
        training_date = "trajdata-generated"

        # Create metadata
        self._metadata = Metadata(
            scene_id=scene_id,
            version_string="trajdata_direct",
            training_date=training_date,
            dataset_hash=dataset_hash,
            uuid=uuid_str,
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
