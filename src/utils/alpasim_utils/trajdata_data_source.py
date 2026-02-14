# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

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
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
logger = logging.getLogger(__name__)

try:
    from trajdata.caching import EnvCache
    from trajdata.data_structures.agent import AgentMetadata
    from trajdata.data_structures.scene_metadata import Scene
    from trajdata.dataset import UnifiedDataset
    from trajdata.maps import VectorMap
    
    # Try to import AgentBatch (may not exist in all versions)
    try:
        from trajdata import AgentBatch
    except ImportError:
        AgentBatch = None
except ImportError as e:
    AgentBatch = None
    Scene = None
    UnifiedDataset = None
    EnvCache = None
    AgentMetadata = None
    VectorMap = None
    # Only log warning at debug level to avoid noise if trajdata is intentionally not installed
    logger.debug(f"trajdata not available: {e}. TrajdataDataSource will not work.")

from alpasim_utils.artifact import Metadata
from alpasim_utils.qvec import QVec
from alpasim_utils.scenario import AABB, CameraId, Rig, TrafficObject, TrafficObjects, VehicleConfig
from alpasim_utils.scene_data_source import SceneDataSource
from alpasim_utils.trajectory import Trajectory




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

    @classmethod
    def from_trajdata_scene(
        cls,
        scene: Scene,
        dataset: Optional[UnifiedDataset] = None,
        scene_cache: Optional[EnvCache] = None,
        scene_id: Optional[str] = None,
        smooth_trajectories: bool = True,
        base_timestamp_us: int = 0,
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
        
        Returns:
            TrajdataDataSource instance
        """
        if Scene is None:
            raise ImportError("trajdata is not installed. Please install it to use TrajdataDataSource.")

        data_source = cls(
            _scene=scene,
            _dataset=dataset,
            _scene_cache=scene_cache,
            _scene_id=scene_id or scene.name,
            _smooth_trajectories=smooth_trajectories,
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
            raise ImportError("trajdata is not installed. Please install it to use TrajdataDataSource.")

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
    
    def _get_scene_cache(self) -> EnvCache:
        """Get or create scene_cache"""
        if self._scene_cache is not None:
            return self._scene_cache
        
        if self._scene is None:
            raise ValueError("Cannot create scene_cache: scene is not set")
        
        if self._dataset is None:
            raise ValueError("Cannot create scene_cache: dataset is not set")
        
        # Create SceneCache
        self._scene_cache = self._dataset.cache_class(
            self._dataset.cache_path, self._scene, self._dataset.augmentations
        )
        self._scene_cache.set_obs_format(self._dataset.obs_format)
        return self._scene_cache

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
            poses_vec3 = []
            poses_quat = []

            # Iterate through all timesteps
            for ts in range(agent.first_timestep, agent.last_timestep + 1):
                try:
                    state = scene_cache.get_raw_state(agent.name, ts)
                    
                    # Get position and orientation
                    x = state.get_attr("x") if hasattr(state, 'get_attr') else state.x
                    y = state.get_attr("y") if hasattr(state, 'get_attr') else state.y
                    z = state.get_attr("z") if hasattr(state, 'get_attr') else (state.z if hasattr(state, 'z') else 0.0)
                    heading = state.get_attr("h") if hasattr(state, 'get_attr') else state.h

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
                    poses_vec3.append([x_val, y_val, z_val])

                    # Convert heading to quaternion
                    quat = R.from_euler('z', heading_val).as_quat()  # [x, y, z, w]
                    poses_quat.append(quat)

                except Exception as e:
                    logger.debug(f"Failed to get state for agent {agent.name} at ts {ts}: {e}")
                    continue

            if len(timestamps_us) == 0:
                return None, None

            # Create QVec
            poses = QVec(
                vec3=np.array(poses_vec3, dtype=np.float64),
                quat=np.array(poses_quat, dtype=np.float64),
            )

            # Create Trajectory
            trajectory = Trajectory(
                timestamps_us=np.array(timestamps_us, dtype=np.uint64),
                poses=poses,
            )

            # Create VehicleConfig (extract from extent)
            vehicle_config = None
            if hasattr(agent.extent, 'length'):
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
            raise ValueError("Cannot extract ego trajectory")

        # Calculate world_to_nre transformation matrix (use first trajectory point as origin)
        world_to_nre = np.eye(4)
        if len(ego_trajectory) > 0:
            first_pose_position = ego_trajectory.poses[0].vec3
            world_to_nre[:3, 3] = -first_pose_position
            logger.info(
                f"Setting world_to_nre origin at first pose: {first_pose_position}, "
                f"translation: {world_to_nre[:3, 3]}"
            )

        # Convert ego trajectory to local coordinates (NRE)
        if len(ego_trajectory) > 0:
            translation = world_to_nre[:3, 3]
            local_positions = ego_trajectory.poses.vec3 + translation
            
            # Validate transform
            first_pose_local = local_positions[0]
            if np.linalg.norm(first_pose_local[:2]) > 1.0: 
                logger.warning(
                    f"First pose after transformation is not at origin: {first_pose_local}. "
                    f"Expected [0, 0, ~z], got {first_pose_local}"
                )
            
            local_quat = ego_trajectory.poses.quat.copy()
            local_poses = QVec(vec3=local_positions, quat=local_quat)
            ego_trajectory = Trajectory(
                timestamps_us=ego_trajectory.timestamps_us.copy(),
                poses=local_poses,
            )
            
            logger.info(
                f"Transformed ego trajectory to local coordinates. "
                f"First pose: {ego_trajectory.poses[0].vec3}, "
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
        if not hasattr(self._scene, 'data_access_info') or not self._scene.data_access_info:
            logger.warning(f"scene.data_access_info does not exist, skipping camera information extraction")
            return camera_ids, camera_calibrations
        
        sensor_calibration = self._scene.data_access_info.get('sensor_calibration')
        if not sensor_calibration or not isinstance(sensor_calibration, dict):
            logger.warning(f"sensor_calibration does not exist or has incorrect format, skipping camera information extraction")
            return camera_ids, camera_calibrations
        
        unique_sensor_idx = 0
        for camera_name, calibration_info in sensor_calibration.get('cameras', {}).items():
            try:
                unique_camera_id = f"{camera_name}@{self.scene_id}"
                
                position = calibration_info.get('sensor2ego_translation', [0.0, 0.0, 0.0])
                rotation = calibration_info.get('sensor2ego_rotation', [0.0, 0.0, 0.0, 1.0])
                
                if isinstance(position, (int, float)):
                    position = [float(position), 0.0, 0.0]
                elif len(position) < 3:
                    position = list(position) + [0.0] * (3 - len(position))
                
                if isinstance(rotation, (int, float)):
                    rotation = [0.0, 0.0, 0.0, 1.0]
                elif len(rotation) < 4:
                    if len(rotation) == 3:
                        r = R.from_euler('xyz', rotation)
                        rotation = r.as_quat()
                    else:
                        rotation = [0.0, 0.0, 0.0, 1.0]
                
                if len(rotation) == 4:
                    qx, qy, qz, qw = rotation[0], rotation[1], rotation[2], rotation[3]
                else:
                    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
                
                camera_id = CameraId(
                    logical_name=camera_name,
                    trajectory_idx=0,
                    sequence_id=self.scene_id,
                    unique_id=unique_camera_id,
                )
                camera_ids.append(camera_id)
                unique_sensor_idx += 1
                
            except Exception as e:
                logger.warning(f"Error extracting camera {camera_name} information: {e}")
                continue
        
        if len(camera_ids) == 0:
            # If no camera information, create a default one
            logger.warning(f"Scene {self.scene_id} has no camera information, using default camera")
            camera_ids.append(
                CameraId(
                    logical_name="camera_front",
                    trajectory_idx=0,
                    sequence_id=self.scene_id,
                    unique_id="0@camera_front",
                )
            )
        
        return camera_ids, camera_calibrations

    def _is_static_object(self, trajectory: Trajectory, velocity_threshold: float = 0.1) -> bool:
        """Determine if object is static (based on velocity)"""
        if len(trajectory) < 2:
            return True

        positions = trajectory.poses.vec3
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
            local_positions = trajectory.poses.vec3 + translation
            local_quat = trajectory.poses.quat.copy()
            local_poses = QVec(vec3=local_positions, quat=local_quat)
            trajectory = Trajectory(
                timestamps_us=trajectory.timestamps_us.copy(),
                poses=local_poses,
            )

            # Smooth if needed
            if self._smooth_trajectories:
                try:
                    import csaps

                    css = csaps.CubicSmoothingSpline(
                        trajectory.timestamps_us / 1e6,
                        trajectory.poses.vec3.T,
                        normalizedsmooth=True,
                    )
                    filtered_positions = css(trajectory.timestamps_us / 1e6).T
                    max_error = np.max(np.abs(filtered_positions - trajectory.poses.vec3))
                    if max_error > 1.0:
                        logger.warning(
                            f"Max error in cubic spline approximation: {max_error:.6f} m for {agent.name=}"
                        )
                    trajectory.poses.vec3 = filtered_positions
                except ImportError:
                    logger.warning("csaps not installed, skipping trajectory smoothing")

            # Get AABB
            if hasattr(agent.extent, 'length'):
                aabb = AABB(x=agent.extent.length, y=agent.extent.width, z=agent.extent.height)
            else:
                # Default AABB
                aabb = AABB(x=4.5, y=1.8, z=1.5)

            # Determine if static object
            is_static = self._is_static_object(trajectory)

            # Get category label
            label_class = agent.type.name if hasattr(agent.type, 'name') else "UNKNOWN"

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
        """Load and return VectorMap (obtained from dataset._map_api)"""
        if self._map is not None:
            return self._map

        if VectorMap is None:
            logger.warning("trajdata is not installed, cannot load map")
            return None

        if self._scene is None:
            logger.warning("Cannot load map: scene is not set")
            return None

        if self._dataset is None:
            logger.warning("Cannot load map: dataset is not set")
            return None

        # Get map from dataset._map_api (refer to trajdata_artifact_converter.py)
        try:
            # Check if dataset includes map support
            if (
                not hasattr(self._dataset, 'incl_vector_map')
                or not self._dataset.incl_vector_map
                or not hasattr(self._dataset, '_map_api')
                or self._dataset._map_api is None
            ):
                logger.debug(f"Dataset does not have map support enabled or map_api is unavailable")
                return None

            # Build map name: "{env_name}:{location}"
            if not hasattr(self._scene, 'location') or not self._scene.location:
                logger.debug(f"Scene {self.scene_id} has no location information, cannot load map")
                return None

            map_name = f"{self._scene.env_name}:{self._scene.location}"
            
            # Get vector_map_params (if exists)
            vector_map_params = {}
            if hasattr(self._dataset, 'vector_map_params'):
                vector_map_params = self._dataset.vector_map_params

            # Get map from map_api
            vec_map = self._dataset._map_api.get_map(map_name, **vector_map_params)
            
            if vec_map is None:
                logger.debug(f"Scene {self.scene_id} (map_name: {map_name}) has no map data")
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
                    f"world_to_nre contains rotation. Map transformation may need rotation handling. "
                    f"Currently only applying translation."
                )
            
            # Transform all points in map (center.points, left_boundary.points, right_boundary.points, etc.)
            # Note: Must transform before finalize, as finalize may rebuild certain data structures
            # Important: Only transform X, Y coordinates, align Z coordinate to trajectory Z baseline
            # Because map Z usually represents height relative to ground (usually 0), while trajectory Z
            # represents altitude. After transformation, map Z should align with trajectory Z
            # (both relative to first trajectory point's Z, usually 0 after transformation)
            translation_xy = translation[:2]  # Only use X, Y translation
            
            # Get Z coordinate of first trajectory point (transformed baseline, usually 0)
            first_traj_z = self.rig.trajectory.poses[0].vec3[2] if len(self.rig.trajectory) > 0 else 0.0
            
            logger.info(
                f"Map coordinate transformation: "
                f"translation_xy={translation_xy}, "
                f"first_traj_z={first_traj_z:.2f}m, "
                f"map Z will be aligned to trajectory Z baseline"
            )
            
            def transform_map_points(points: np.ndarray) -> np.ndarray:
                """Transform map points: only transform X, Y, align Z to trajectory Z baseline"""
                if points is None or len(points) == 0 or points.ndim != 2 or points.shape[1] < 3:
                    return points
                
                points_copy = points.copy()
                
                # Transform X, Y coordinates
                if has_rotation:
                    # If rotation exists, need to rotate X, Y
                    xy_rotated = (points_copy[:, :2] @ rotation_matrix[:2, :2].T) + translation_xy
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
                                logger.warning(f"Failed to transform lane {lane_idx} center.points: {e}")
                    
                    # Transform left_boundary.points
                    if hasattr(lane, "left_boundary") and hasattr(lane.left_boundary, "points"):
                        points = lane.left_boundary.points
                        if points is not None and len(points) > 0:
                            try:
                                transformed_points = transform_map_points(points)
                                lane.left_boundary.points = transformed_points
                            except Exception as e:
                                logger.warning(f"Failed to transform lane {lane_idx} left_boundary.points: {e}")
                    
                    # Transform right_boundary.points
                    if hasattr(lane, "right_boundary") and hasattr(lane.right_boundary, "points"):
                        points = lane.right_boundary.points
                        if points is not None and len(points) > 0:
                            try:
                                transformed_points = transform_map_points(points)
                                lane.right_boundary.points = transformed_points
                            except Exception as e:
                                logger.warning(f"Failed to transform lane {lane_idx} right_boundary.points: {e}")
                    
                    # Transform other possible point attributes (if any)
                    for attr_name in ['intersections', 'crosswalks', 'stop_lines']:
                        if hasattr(lane, attr_name):
                            attr_value = getattr(lane, attr_name)
                            if isinstance(attr_value, list):
                                for item in attr_value:
                                    if hasattr(item, 'points'):
                                        points = item.points
                                        if points is not None and len(points) > 0:
                                            try:
                                                transformed_points = transform_map_points(points)
                                                item.points = transformed_points
                                            except Exception as e:
                                                logger.debug(f"Failed to transform {attr_name} points: {e}")
            
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
                if hasattr(first_lane, "center") and hasattr(first_lane.center, "points"):
                    first_map_point_after_finalize = first_lane.center.points[0, :3] if len(first_lane.center.points) > 0 else None
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
                    if hasattr(lane, "next_lanes") and isinstance(lane.next_lanes, list):
                        lane.next_lanes = set(lane.next_lanes)
                    if hasattr(lane, "prev_lanes") and isinstance(lane.prev_lanes, list):
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
                if hasattr(first_lane, "center") and hasattr(first_lane.center, "points"):
                    first_map_point = first_lane.center.points[0, :3] if len(first_lane.center.points) > 0 else None
                    if first_map_point is not None:
                        # Map point should be near trajectory (within hundreds of meters)
                        distance_from_origin_xy = np.linalg.norm(first_map_point[:2])  # Only check X, Y
                        distance_from_origin_xyz = np.linalg.norm(first_map_point)  # Check X, Y, Z
                        
                        # Get first trajectory point for comparison
                        first_traj_point = self.rig.trajectory.poses[0].vec3
                        
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

            logger.info(f"Successfully loaded map: {map_name} (transformed to local coordinate system)")
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
            time_range_end = float(base_timestamp_us + length_timesteps * dt * 1e6) / 1e6
        else:
            time_range_start = float(rig.trajectory.time_range_us.start) / 1e6
            time_range_end = float(rig.trajectory.time_range_us.stop) / 1e6

        # Create metadata
        from datetime import datetime
        import uuid
        
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


def discover_from_trajdata_dataset(
    dataset: UnifiedDataset,
    scene_indices: Optional[list[int]] = None,
    smooth_trajectories: bool = True,
    base_timestamp_us: int = 0,
) -> dict[str, TrajdataDataSource]:
    """
    Create TrajdataDataSource dictionary from trajdata UnifiedDataset.
    
    Args:
        dataset: UnifiedDataset instance
        scene_indices: List of scene indices to load (if None, load all scenes)
        smooth_trajectories: Whether to smooth trajectories
        base_timestamp_us: Base timestamp in microseconds
    
    Returns:
        dict[scene_id, TrajdataDataSource]
    """
    if UnifiedDataset is None:
        raise ImportError("trajdata is not installed. Please install it to use this function.")
    
    data_sources = {}
    
    if scene_indices is None:
        # Get indices for all scenes
        # Use dataset._scene_index to get scene count (refer to trajdata_artifact_converter.py)
        # len(dataset) may not equal scene count, so use _scene_index
        if hasattr(dataset, '_scene_index'):
            num_scenes = len(dataset._scene_index)
            logger.info(f"Got {num_scenes} scenes from dataset._scene_index")
        elif hasattr(dataset, '__len__'):
            # Fallback: if _scene_index doesn't exist, try using __len__
            try:
                num_scenes = len(dataset)
                logger.warning(
                    "Using len(dataset) to get scene count, this may be inaccurate. "
                    "Consider checking if dataset._scene_index is available."
                )
            except (TypeError, NotImplementedError):
                logger.error("Cannot determine scene count: dataset has neither _scene_index nor supports __len__")
                return data_sources
        else:
            logger.error("Cannot determine scene count: dataset has no _scene_index attribute and doesn't support __len__")
            return data_sources
        
        if num_scenes == 0:
            logger.warning("No scenes found in dataset, please check dataset configuration")
            return data_sources
        
        scene_indices = list(range(num_scenes))
        logger.info(f"Will load all {num_scenes} scenes (indices: 0 to {num_scenes-1})")
    
    for scene_idx in scene_indices:
        try:
            scene = dataset.get_scene(scene_idx)
            scene_id = scene.name
            
            data_source = TrajdataDataSource.from_trajdata_scene(
                scene=scene,
                dataset=dataset,
                scene_id=scene_id,
                smooth_trajectories=smooth_trajectories,
                base_timestamp_us=base_timestamp_us,
            )
            
            data_sources[scene_id] = data_source
            logger.info(f"Loaded scene: {scene_id} (scene index: {scene_idx})")
            
        except Exception as e:
            logger.error(f"Failed to Load {scene_idx}: {e}")
            continue
    
    return data_sources


class LazyTrajdataArtifactLoader(dict[str, TrajdataDataSource]):
    """
    Lazy-loading wrapper for trajdata artifacts.
    
    Instead of loading all scenes at startup (which is slow), this class
    loads scenes on-demand when they are first accessed. This significantly
    speeds up worker startup in multi-worker mode.
    """
    
    def __init__(
        self,
        dataset: UnifiedDataset | None = None,
        dataset_config: dict | None = None,
        trajdata_config_path: str | None = None,
        scene_indices: Optional[list[int]] = None,
        smooth_trajectories: bool = True,
        base_timestamp_us: int = 0,
    ):
        """
        Initialize lazy loader.
        
        Args:
            dataset: UnifiedDataset instance (if provided, use it directly)
            dataset_config: Config dict for creating UnifiedDataset (if dataset not provided)
            trajdata_config_path: Path to trajdata config file (for error messages)
            scene_indices: List of scene indices to load (if None, all scenes)
            smooth_trajectories: Whether to smooth trajectories
            base_timestamp_us: Base timestamp in microseconds
        """
        if dataset is None and dataset_config is None:
            raise ValueError("Either dataset or dataset_config must be provided")
        
        self._dataset = dataset  # Will be created lazily if None
        self._dataset_config = dataset_config
        self._trajdata_config_path = trajdata_config_path
        self._scene_indices = scene_indices
        self._smooth_trajectories = smooth_trajectories
        self._base_timestamp_us = base_timestamp_us
        self._loaded_scenes: dict[str, TrajdataDataSource] = {}
        self._scene_id_to_index: dict[str, int] = {}
        self._lock = threading.Lock()
        self._total_scene_count: int = 0
        
        # Store scene count if scene_indices is provided
        if scene_indices is not None:
            self._total_scene_count = len(scene_indices)
        
        # Don't create dataset or build mapping here - do it lazily on first access
        # This avoids slow UnifiedDataset creation at worker startup
        logger.info(
            f"LazyTrajdataArtifactLoader initialized (UnifiedDataset and mapping will be created on-demand)"
        )
    
    def _ensure_dataset_created(self) -> None:
        """Create UnifiedDataset if not already created."""
        if self._dataset is not None:
            return  # Already created
        
        if self._dataset_config is None:
            raise ValueError("Cannot create UnifiedDataset: no config provided")
        
        logger.info("Creating UnifiedDataset (first access, this may take a while)...")
        import time
        create_start = time.perf_counter()
        self._dataset = UnifiedDataset(**self._dataset_config)
        create_elapsed = time.perf_counter() - create_start
        logger.info(f"UnifiedDataset created in {create_elapsed:.2f} seconds")
        
        # Determine scene_indices if not provided
        if self._scene_indices is None:
            if hasattr(self._dataset, '_scene_index'):
                self._scene_indices = list(range(len(self._dataset._scene_index)))
            else:
                logger.warning("Cannot determine scene indices, lazy loading may not work correctly")
                self._scene_indices = []
        
        # Update scene count for __len__()
        self._total_scene_count = len(self._scene_indices) if self._scene_indices else 0
    
    def _build_mapping_if_needed(self) -> None:
        """Build scene_id to index mapping on first access (lazy, only when needed)."""
        if self._scene_id_to_index:
            return  # Already built
        
        # Ensure dataset is created first
        self._ensure_dataset_created()
        
        logger.debug("Building scene_id to index mapping lazily (will build on-demand)")
        # Don't build mapping here - build it lazily when scenes are accessed
        # This avoids loading all scenes at startup
        # The mapping will be built incrementally as scenes are accessed via __getitem__
        
        # If we have scene_indices, we can pre-populate the mapping structure
        # but we'll only get scene names when scenes are actually accessed
        # For now, we'll build the mapping lazily in __getitem__
    
    def __getitem__(self, scene_id: str) -> TrajdataDataSource:
        """Load scene on-demand when first accessed."""
        with self._lock:
            # Ensure dataset is created (lazy creation)
            self._ensure_dataset_created()
            
            # Check if already loaded
            if scene_id in self._loaded_scenes:
                return self._loaded_scenes[scene_id]
            
            # If mapping has this scene, use it directly
            if scene_id in self._scene_id_to_index:
                scene_idx = self._scene_id_to_index[scene_id]
                logger.debug(f"Lazy loading scene {scene_id} (index {scene_idx})")
                scene = self._dataset.get_scene(scene_idx)
                
                data_source = TrajdataDataSource.from_trajdata_scene(
                    scene=scene,
                    dataset=self._dataset,
                    scene_id=scene_id,
                    smooth_trajectories=self._smooth_trajectories,
                    base_timestamp_us=self._base_timestamp_us,
                )
                
                self._loaded_scenes[scene_id] = data_source
                return data_source
            
            # Mapping doesn't have this scene yet - search lazily through indices
            # This is still lazy - we only search until we find the matching scene
            logger.debug(f"Searching for scene {scene_id} in dataset (lazy search)...")
            for scene_idx in self._scene_indices:
                try:
                    scene = self._dataset.get_scene(scene_idx)
                    found_scene_id = scene.name
                    # Add to mapping for future lookups
                    self._scene_id_to_index[found_scene_id] = scene_idx
                    
                    # If this is the scene we're looking for, use it
                    if found_scene_id == scene_id:
                        data_source = TrajdataDataSource.from_trajdata_scene(
                            scene=scene,
                            dataset=self._dataset,
                            scene_id=scene_id,
                            smooth_trajectories=self._smooth_trajectories,
                            base_timestamp_us=self._base_timestamp_us,
                        )
                        self._loaded_scenes[scene_id] = data_source
                        return data_source
                except Exception as e:
                    logger.warning(f"Failed to get scene {scene_idx} while searching: {e}")
                    continue
            
            raise KeyError(f"Scene {scene_id} not found in dataset")
    
    def __contains__(self, scene_id: str) -> bool:
        """Check if scene_id exists (lazy - only checks already discovered scenes)."""
        # Don't build full mapping - just check what we've discovered so far
        # This avoids loading all scenes
        with self._lock:
            return scene_id in self._scene_id_to_index
    
    def __len__(self) -> int:
        """Return number of available scenes."""
        # #region agent log
        import time
        import os
        import json
        len_start = time.perf_counter()
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "trajdata_data_source.py:1119", "message": "__len__ called", "data": {"worker_id": os.environ.get("WORKER_ID", "unknown"), "pid": os.getpid(), "total_scene_count": self._total_scene_count, "has_dataset": self._dataset is not None, "has_mapping": bool(self._scene_id_to_index)}, "timestamp": int(time.time() * 1000)}) + "\n")
                f.flush()
        except Exception:
            pass
        # #endregion
        # Return stored count if available, otherwise 0 (will be updated when dataset is created)
        if self._total_scene_count > 0:
            return self._total_scene_count
        # If dataset is already created and mapping is built, use that
        if self._scene_id_to_index:
            return len(self._scene_id_to_index)
        # Otherwise return 0 (will be determined when dataset is first accessed)
        return 0
    
    def keys(self):
        """Return scene IDs (lazy - only returns already discovered scenes)."""
        # Don't build full mapping - just return what we've discovered so far
        # This avoids loading all scenes
        with self._lock:
            return self._scene_id_to_index.keys()
    
    def values(self):
        """Return loaded data sources (only returns already loaded scenes)."""
        # Don't load all scenes - just return what's already loaded
        return self._loaded_scenes.values()
    
    def items(self):
        """Return (scene_id, data_source) pairs (only returns already loaded scenes)."""
        # Don't load all scenes - just return what's already loaded
        return self._loaded_scenes.items()
