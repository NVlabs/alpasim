# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Conversion utilities: convert a trajdata UnifiedDataset into the alpasim Artifact format.
"""

from __future__ import annotations

import json
import logging
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from alpasim_utils.artifact import Artifact, Metadata
from alpasim_utils.qvec import QVec
from alpasim_utils.scenario import AABB, CameraId, Rig, TrafficObject, TrafficObjects, VehicleConfig
from alpasim_utils.trajectory import Trajectory

logger = logging.getLogger(__name__)

try:
    from trajdata.caching import EnvCache
    from trajdata.data_structures.agent import AgentMetadata, AgentType
    from trajdata.data_structures.scene_metadata import Scene
    from trajdata.dataset import UnifiedDataset
    from trajdata.maps import VectorMap
except ImportError:
    logger.error("trajdata is required for this conversion function")
    raise


def convert_unified_dataset_to_artifact(
    dataset: UnifiedDataset,
    scene_idx: int,
    output_path: str,
    scene_id: Optional[str] = None,
    version_string: str = "trajdata_converted",
    base_timestamp_us: Optional[int] = None,
    include_map: bool = True,
) -> Artifact:
    """
    Convert a scene from a UnifiedDataset to the Artifact format and save it as a USDZ file.

    Args:
        dataset: UnifiedDataset instance.
        scene_idx: Index of the scene to convert.
        output_path: Output USDZ file path (must end with .usdz).
        scene_id: Scene ID (if None, generated from scene.name).
        version_string: Version string.
        base_timestamp_us: Base timestamp (microseconds); if None, starts from 0.
        include_map: Whether to include map data.

    Returns:
        Artifact instance.
    """
    if not output_path.endswith(".usdz"):
        raise ValueError("output_path must end with .usdz")

    # 1. Get the Scene object
    scene = dataset.get_scene(scene_idx)
    scene_path = dataset._scene_index[scene_idx]
    
    # Create SceneCache
    scene_cache = dataset.cache_class(dataset.cache_path, scene, dataset.augmentations)
    scene_cache.set_obs_format(dataset.obs_format)

    # 2. Generate scene_id
    if scene_id is None:
        scene_id = f"trajdata-{scene.env_name}-{scene.name}"

    # 3. Extract full trajectories for all agents
    dt = scene.dt
    length_timesteps = scene.length_timesteps
    
    # Get all agents
    all_agents = scene.agents if scene.agents else []
    
    # Identify ego agent
    ego_agent = next((a for a in all_agents if a.name == "ego"), None)
    if ego_agent is None and len(all_agents) > 0:
        # If there is no ego, use the first agent
        ego_agent = all_agents[0]
        logger.warning(f"No ego agent found, using first agent: {ego_agent.name}")

    # 4. Extract ego trajectory
    ego_trajectory = None
    ego_vehicle_config = None
    if ego_agent:
        ego_trajectory, ego_vehicle_config = _extract_agent_trajectory(
            scene, scene_cache, ego_agent, dt, base_timestamp_us
        )

    # 5. Compute world_to_nre transform matrix (use first trajectory point as origin)
    world_to_nre = np.eye(4)
    if ego_trajectory and len(ego_trajectory) > 0:
        # Get the position of the first trajectory point (global coordinates)
        first_pose_position = ego_trajectory.poses[0].vec3
        
        # Set world_to_nre matrix: use the first trajectory point as origin.
        # The translation part of world_to_nre = -position of the first trajectory point.
        world_to_nre[:3, 3] = -first_pose_position
        
        logger.info(
            f"Setting world_to_nre origin at first pose: {first_pose_position}, "
            f"translation: {world_to_nre[:3, 3]}"
        )
    else:
        logger.warning("No ego trajectory found, using identity world_to_nre matrix")

    # 6. Transform ego trajectory into local coordinates (NRE)
    if ego_trajectory and len(ego_trajectory) > 0:
        ego_trajectory = _transform_trajectory_to_local(ego_trajectory, world_to_nre)
        logger.info(
            f"Transformed ego trajectory to local coordinates. "
            f"First pose: {ego_trajectory.poses[0].vec3}"
        )

    # 7. Extract trajectories of other agents (traffic objects)
    traffic_objects = {}
    for agent in all_agents:
        if agent.name == "ego" or agent == ego_agent:
            continue
        
        trajectory, _ = _extract_agent_trajectory(
            scene, scene_cache, agent, dt, base_timestamp_us
        )
        
        # Filter out empty trajectories or trajectories with only 1 point (smoothing needs at least 2 points)
        if trajectory is None or len(trajectory) < 2:
            continue

        # Transform trajectory into local coordinates (NRE)
        trajectory = _transform_trajectory_to_local(trajectory, world_to_nre)

        # Get extent of the agent
        extent = agent.extent
        if hasattr(extent, 'length'):
            aabb = AABB(x=extent.length, y=extent.width, z=extent.height)
        else:
            # Default size
            aabb = AABB(x=4.5, y=1.8, z=1.5)

        # Determine whether this is a static object (simple heuristic: very small velocity)
        is_static = _is_static_object(trajectory)

        # Get agent type label
        label_class = agent.type.name if hasattr(agent.type, 'name') else "UNKNOWN"

        traffic_objects[agent.name] = TrafficObject(
            track_id=agent.name,
            aabb=aabb,
            trajectory=trajectory,
            is_static=is_static,
            label_class=label_class,
        )

    # 8. Create Rig object
    rig = None
    if ego_trajectory and len(ego_trajectory) > 0:
        
        # Extract camera information from scene.data_access_info['sensor_calibration']
        camera_ids, camera_calibrations = _extract_camera_info_from_scene(
            scene, scene_id
        )
        
        rig = Rig(
            sequence_id=scene_id,
            trajectory=ego_trajectory,
            camera_ids=camera_ids,
            world_to_nre=world_to_nre,
            vehicle_config=ego_vehicle_config,
        )
        
        # Store camera_calibrations on the rig object (for later writing to USDZ)
        rig._camera_calibrations = camera_calibrations

    # 9. Create TrafficObjects
    traffic_objects_dict = TrafficObjects(traffic_objects)

    # 10. Create Metadata
    if base_timestamp_us is None:
        base_timestamp_us = 0
    
    time_range_start = float(base_timestamp_us) / 1e6
    time_range_end = float(base_timestamp_us + length_timesteps * dt * 1e6) / 1e6

    # Extract camera ID list from rig (if present)
    camera_id_names = []
    if rig and rig.camera_ids:
        camera_id_names = [camera_id.logical_name for camera_id in rig.camera_ids]

    metadata = Metadata(
        scene_id=scene_id,
        version_string=version_string,
        training_date=datetime.now().strftime("%Y-%m-%d"),
        dataset_hash=str(uuid.uuid4()),
        uuid=str(uuid.uuid4()),
        is_resumable=False,
        sensors=Metadata.Sensors(camera_ids=camera_id_names, lidar_ids=[]),
        logger=Metadata.Logger(name=None, run_id=None, run_url=None),
        time_range=Metadata.TimeRange(start=time_range_start, end=time_range_end),
    )

    # 11. Create the USDZ file
    _create_usdz_file(
        output_path,
        metadata,
        rig,
        traffic_objects_dict,
        scene_id,
        scene,
        dataset,
        include_map,
    )

    # 12. Return the Artifact instance
    return Artifact(source=output_path)


def _extract_camera_info_from_scene(
    scene: Scene,
    scene_id: str,
) -> tuple[list[CameraId], dict]:
    """
    Extract camera information from scene.data_access_info['sensor_calibration'].
    
    Args:
        scene: Scene object.
        scene_id: Scene ID.
        
    Returns:
        (camera_ids, camera_calibrations) tuple.
        - camera_ids: list of CameraId objects.
        - camera_calibrations: dict of camera calibration information, in the same format as artifact_2.
    """
    camera_ids = []
    camera_calibrations = {}
    
    # Check whether sensor_calibration exists
    if not hasattr(scene, 'data_access_info') or not scene.data_access_info:
        logger.warning("scene.data_access_info does not exist, skipping camera info extraction")
        return camera_ids, camera_calibrations
    
    sensor_calibration = scene.data_access_info.get('sensor_calibration')
    if not sensor_calibration:
        logger.warning("scene.data_access_info['sensor_calibration'] does not exist, skipping camera info extraction")
        return camera_ids, camera_calibrations
    
    # Iterate over camera information in sensor_calibration.
    # sensor_calibration is expected to be a dict whose keys are camera names and values are calibration info.
    if not isinstance(sensor_calibration, dict):
        logger.warning("sensor_calibration is not a dict, skipping camera info extraction")
        return camera_ids, camera_calibrations
    
    unique_sensor_idx = 0
    for camera_name, calibration_info in sensor_calibration['cameras'].items():
        try:
            # Build unique_camera_id (format: logical_sensor_name@sequence_id)
            unique_camera_id = f"{camera_name}@{scene_id}"
            
            # Extract position and rotation information.
            # Assume calibration_info contains position and rotation fields.
            # rotation is a quaternion in the order [x, y, z, w].
            position = calibration_info.get('sensor2ego_translation', [0.0, 0.0, 0.0])
            rotation = calibration_info.get('sensor2ego_rotation', [0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
            
            # Ensure position and rotation are lists or arrays
            if isinstance(position, (int, float)):
                position = [float(position), 0.0, 0.0]
            elif len(position) < 3:
                position = list(position) + [0.0] * (3 - len(position))
            
            if isinstance(rotation, (int, float)):
                rotation = [0.0, 0.0, 0.0, 1.0]
            elif len(rotation) < 4:
                # If there are only 3 values, assume they are Euler angles and convert to a quaternion
                if len(rotation) == 3:
                    r = R.from_euler('xyz', rotation)
                    rotation = r.as_quat()  # [x, y, z, w]
                else:
                    rotation = [0.0, 0.0, 0.0, 1.0]
            
            # Ensure rotation is in [x, y, z, w] format
            if len(rotation) == 4:
                qx, qy, qz, qw = rotation[0], rotation[1], rotation[2], rotation[3]
            else:
                logger.warning(f"Camera {camera_name} has invalid rotation format, using default value")
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            
            # Build the T_sensor_rig transform matrix (4x4).
            # Build a rotation matrix from the quaternion.
            r_quat = R.from_quat([qx, qy, qz, qw])
            rotation_matrix = r_quat.as_matrix()
            
            # Build the full transform matrix
            T_sensor_rig = np.eye(4)
            T_sensor_rig[:3, :3] = rotation_matrix
            T_sensor_rig[:3, 3] = position[:3]
            
            # Create CameraId object
            camera_id = CameraId(
                logical_name=camera_name,
                trajectory_idx=0,
                sequence_id=scene_id,
                unique_id=unique_camera_id,
            )
            camera_ids.append(camera_id)
            
            # Build camera_calibration dict.
            # Extract camera intrinsics if present.
            camera_model = calibration_info.get('camera_model', {})
            if not camera_model:
                # If no camera model is provided, fall back to default values
                camera_model = {
                    "type": "pinhole",  # Use pinhole model by default
                    "parameters": {
                        "resolution": calibration_info.get('resolution', [1920, 1080]),
                        "focal_length": calibration_info.get('focal_length', [1000.0, 1000.0]),
                        "principal_point": calibration_info.get('principal_point', [960.0, 540.0]),
                    }
                }
            
            camera_calibrations[unique_camera_id] = {
                "sequence_id": scene_id,
                "logical_sensor_name": camera_name,
                "unique_sensor_idx": unique_sensor_idx,
                "T_sensor_rig": T_sensor_rig.tolist(),
                "camera_model": camera_model,
            }
            
            unique_sensor_idx += 1
            logger.info(f"Successfully extracted camera info: {camera_name} (unique_id: {unique_camera_id})")
            
        except Exception as e:
            logger.warning(f"Failed to extract camera info for {camera_name}: {e}")
            continue
    
    logger.info(f"Extracted {len(camera_ids)} cameras in total")
    return camera_ids, camera_calibrations


def _extract_agent_trajectory(
    scene: Scene,
    scene_cache,
    agent: AgentMetadata,
    dt: float,
    base_timestamp_us: Optional[int],
) -> tuple[Optional[Trajectory], Optional[VehicleConfig]]:
    """
    Extract the full trajectory for an agent.

    Returns:
        (Trajectory, VehicleConfig) or (None, None) if extraction fails.
    """
    try:
        timestamps_us = []
        poses_vec3 = []
        poses_quat = []

        # Iterate over all timesteps
        for ts in range(agent.first_timestep, agent.last_timestep + 1):
            try:
                state = scene_cache.get_raw_state(agent.name, ts)
                
                # Get position and heading
                x = state.get_attr("x") if hasattr(state, 'get_attr') else state.x
                y = state.get_attr("y") if hasattr(state, 'get_attr') else state.y
                z = state.get_attr("z") if hasattr(state, 'get_attr') else (state.z if hasattr(state, 'z') else 0.0)
                heading = state.get_attr("h") if hasattr(state, 'get_attr') else state.h

                # Convert to numpy arrays (handling scalar cases)
                if isinstance(x, (int, float)):
                    x = np.array([x])
                if isinstance(y, (int, float)):
                    y = np.array([y])
                if isinstance(z, (int, float)):
                    z = np.array([z])
                if isinstance(heading, (int, float)):
                    heading = np.array([heading])

                # Take the first element (if they are arrays)
                x_val = float(x[0] if x.ndim > 0 else x)
                y_val = float(y[0] if y.ndim > 0 else y)
                z_val = float(z[0] if z.ndim > 0 else z)
                heading_val = float(heading[0] if heading.ndim > 0 else heading)

                # Compute timestamp
                if base_timestamp_us is None:
                    timestamp_us = int(ts * dt * 1e6)
                else:
                    timestamp_us = int(base_timestamp_us + ts * dt * 1e6)

                timestamps_us.append(timestamp_us)
                poses_vec3.append([x_val, y_val, z_val])

                # Convert heading to quaternion.
                # Heading is in radians, rotation about z-axis.
                quat = R.from_euler('z', heading_val).as_quat()  # [x, y, z, w]
                poses_quat.append(quat)

            except Exception as e:
                logger.debug(f"Failed to get state for agent {agent.name} at ts {ts}: {e}")
                continue

        if len(timestamps_us) == 0:
            return None, None

        # Create QVec
        poses = QVec(
            vec3=np.array(poses_vec3, dtype=np.float32),
            quat=np.array(poses_quat, dtype=np.float32),
        )

        # Create Trajectory
        trajectory = Trajectory(
            timestamps_us=np.array(timestamps_us, dtype=np.uint64),
            poses=poses,
        )

        # Create VehicleConfig (from extent)
        vehicle_config = None
        if hasattr(agent.extent, 'length'):
            vehicle_config = VehicleConfig(
                aabb_x_m=agent.extent.length,
                aabb_y_m=agent.extent.width,
                aabb_z_m=agent.extent.height,
                aabb_x_offset_m=-agent.extent.length / 2,  # Simplifying assumption
                aabb_y_offset_m=0.0,
                aabb_z_offset_m=-agent.extent.height / 2,
            )

        return trajectory, vehicle_config

    except Exception as e:
        logger.error(f"Failed to extract trajectory for agent {agent.name}: {e}")
        return None, None


def _transform_trajectory_to_local(
    trajectory: Trajectory,
    world_to_nre: np.ndarray,
) -> Trajectory:
    """
    Transform a trajectory from the global coordinate frame into a local (NRE) frame.
    
    Args:
        trajectory: Trajectory in the global coordinate frame.
        world_to_nre: Transform matrix from global coordinates to NRE coordinates (4x4).
    
    Returns:
        Trajectory in the local coordinate frame.
    """
    # Extract translation component
    translation = world_to_nre[:3, 3]
    
    # Transform positions (vec3)
    local_positions = trajectory.poses.vec3 + translation
    
    # Keep quaternions unchanged (assume pure translation without rotation).
    # If world_to_nre includes rotation, we would need to apply that to the quaternions.
    # For now this is simplified to translation only.
    local_quat = trajectory.poses.quat.copy()
    
    # Create a new QVec
    local_poses = QVec(
        vec3=local_positions,
        quat=local_quat,
    )
    
    # Create a new Trajectory
    return Trajectory(
        timestamps_us=trajectory.timestamps_us.copy(),
        poses=local_poses,
    )


def _write_map_to_usdz(
    zip_file: zipfile.ZipFile,
    scene: Scene,
    dataset: UnifiedDataset,
    logger: logging.Logger,
) -> None:
    """
    Write map information into the USDZ file.

    Prefer to obtain a VectorMap from UnifiedDataset.map_api, then:
    1. Try to serialize the VectorMap to protobuf and store it (recommended).
    2. If that fails, try to locate the raw XODR file as a fallback.

    Note: artifact.py must support loading maps from protobuf, otherwise only XODR can be used.
    """
    map_written = False
    
    # Method 1: directly get a VectorMap from UnifiedDataset and serialize it to protobuf
    try:
        if (
            hasattr(dataset, 'incl_vector_map') 
            and dataset.incl_vector_map 
            and hasattr(dataset, '_map_api') 
            and dataset._map_api
        ):
            if hasattr(scene, 'location') and scene.location:
                map_name = f"{scene.env_name}:{scene.location}"
                vector_map = dataset._map_api.get_map(map_name, **dataset.vector_map_params)
                
                if vector_map:
                    # Serialize VectorMap to protobuf
                    try:
                        import trajdata.proto.vectorized_map_pb2 as map_proto
                        vec_map_proto = vector_map.to_proto()
                        proto_bytes = vec_map_proto.SerializeToString()
                        
                        # Store as map.pb
                        zip_file.writestr("map.pb", proto_bytes)
                        logger.info(f"Successfully wrote VectorMap as protobuf (map.pb) for {map_name}")
                        map_written = True
                    except Exception as e:
                        logger.debug(f"Failed to serialize VectorMap to protobuf: {e}")
    except Exception as e:
        logger.debug(f"Could not get VectorMap from dataset API: {e}")
    
    # Method 2: if protobuf serialization fails, try to find the original XODR file (fallback)
    if not map_written:
        xodr_content = None
        
        # Try to find an XODR file in the raw data directories
        try:
            if hasattr(scene, 'location') and scene.location:
                # nuplan XODR files are typically stored in data_dir.parent / "maps" / "{location}.xodr"
                if hasattr(dataset, 'data_dirs') and dataset.data_dirs:
                    for env_name, data_dir in dataset.data_dirs.items():
                        if env_name == scene.env_name:
                            # According to nuplan_dataset.py:381, map_root = data_dir.parent / "maps"
                            xodr_path = Path(data_dir).parent / "maps" / f"{scene.location}.xodr"
                            if xodr_path.exists():
                                xodr_content = xodr_path.read_text(encoding='utf-8')
                                logger.info(f"Found XODR file at {xodr_path}")
                                break
        except Exception as e:
            logger.debug(f"Could not find XODR file in data directories: {e}")
        
        # If an XODR file is found, write it into the USDZ
        if xodr_content:
            zip_file.writestr("map.xodr", xodr_content.encode('utf-8'))
            logger.info("Successfully wrote map.xodr to USDZ file")
            map_written = True
    
    if not map_written:
        logger.warning(
            f"Could not write map for scene {scene.name} "
            f"(location: {getattr(scene, 'location', 'unknown')}). "
            f"Map will not be available in the USDZ file."
        )




def _is_static_object(trajectory: Trajectory, velocity_threshold: float = 0.1) -> bool:
    """
    Determine whether an object is static (based on its velocity).
    """
    if len(trajectory) < 2:
        return True

    # Compute average velocity
    positions = trajectory.poses.vec3
    timestamps = trajectory.timestamps_us.astype(np.float64) / 1e6  # convert to seconds

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


def _create_usdz_file(
    output_path: str,
    metadata: Metadata,
    rig: Optional[Rig],
    traffic_objects: TrafficObjects,
    sequence_id: str,
    scene: Scene,
    dataset: UnifiedDataset,
    include_map: bool,
) -> None:
    """
    Create a USDZ file (technically a ZIP file).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Write metadata.yaml
        metadata_dict = metadata.to_dict()
        yaml_content = yaml.dump(metadata_dict, default_flow_style=False)
        zip_file.writestr("metadata.yaml", yaml_content)

        # 2. Write rig_trajectories.json
        if rig:
            # Get camera calibrations if present
            camera_calibrations = getattr(rig, '_camera_calibrations', {})
            
            # Build the cameras_frame_timestamps_us dictionary
            cameras_frame_timestamps_us = {}
            if rig.camera_ids:
                # Use the rig trajectory timestamps for each camera
                rig_timestamps = rig.trajectory.timestamps_us.tolist()
                for camera_id in rig.camera_ids:
                    cameras_frame_timestamps_us[camera_id.unique_id] = rig_timestamps
            
            rig_data = {
                "world_to_nre": {
                    "matrix": rig.world_to_nre.tolist()
                },
                "camera_calibrations": camera_calibrations,
                "rig_trajectories": [
                    {
                        "sequence_id": rig.sequence_id,
                        "T_rig_world_timestamps_us": rig.trajectory.timestamps_us.tolist(),
                        "T_rig_worlds": [pose.as_se3().tolist() for pose in rig.trajectory.poses],
                        "cameras_frame_timestamps_us": cameras_frame_timestamps_us,
                        "rig_bbox": None,  # can be added later
                    }
                ],
                "T_world_base": np.eye(4).tolist(),  # identity by default
            }

            # If there is a vehicle_config, add rig_bbox
            if rig.vehicle_config:
                rig_data["rig_trajectories"][0]["rig_bbox"] = {
                    "centroid": [
                        rig.vehicle_config.aabb_x_offset_m + rig.vehicle_config.aabb_x_m / 2,
                        rig.vehicle_config.aabb_y_offset_m,
                        rig.vehicle_config.aabb_z_offset_m + rig.vehicle_config.aabb_z_m / 2,
                    ],
                    "dim": [
                        rig.vehicle_config.aabb_x_m,
                        rig.vehicle_config.aabb_y_m,
                        rig.vehicle_config.aabb_z_m,
                    ],
                    "rot": [0.0, 0.0, 0.0],
                }

            zip_file.writestr("rig_trajectories.json", json.dumps(rig_data, indent=2))

        # 3. Write sequence_tracks.json
        tracks_data = {
            "tracks_id": [],
            "tracks_label_class": [],
            "tracks_flags": [],
            "tracks_timestamps_us": [],
            "tracks_poses": [],
        }
        cuboids_dims = []

        for track_id, traffic_obj in traffic_objects.items():
            tracks_data["tracks_id"].append(track_id)
            tracks_data["tracks_label_class"].append(traffic_obj.label_class)
            tracks_data["tracks_flags"].append("CONTROLLABLE" if not traffic_obj.is_static else "STATIC")
            tracks_data["tracks_timestamps_us"].append(traffic_obj.trajectory.timestamps_us.tolist())
            
            # Convert QVec to [x, y, z, qx, qy, qz, qw] format
            poses_list = []
            for pose in traffic_obj.trajectory.poses:
                pose_array = np.concatenate([pose.vec3, pose.quat])
                poses_list.append(pose_array.tolist())
            tracks_data["tracks_poses"].append(poses_list)
            
            cuboids_dims.append([traffic_obj.aabb.x, traffic_obj.aabb.y, traffic_obj.aabb.z])

        sequence_tracks_data = {
            sequence_id: {
                "tracks_data": tracks_data,
                "cuboidtracks_data": {
                    "cuboids_dims": cuboids_dims,
                },
            }
        }

        zip_file.writestr("sequence_tracks.json", json.dumps(sequence_tracks_data, indent=2))

        # 4. Write map if present
        if include_map:
            _write_map_to_usdz(zip_file, scene, dataset, logger)

    logger.info(f"Created USDZ file: {output_path}")
    

if __name__ == "__main__":
    
    # artifact_2 = Artifact(source='data/nre-artifacts/all-usdzs/sample_set/25.07_release/Batch0001/05bb8212-63e1-40a8-b4fc-3142c0e94646/05bb8212-63e1-40a8-b4fc-3142c0e94646.usdz')
    
    dataset = UnifiedDataset(
        desired_data=["nuplan_train"],
        cache_location="/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/trajdata/datasets",
        incl_vector_map=True,
        rebuild_cache=False,
        rebuild_maps=False,
        require_map_cache=False,
        num_workers=1,
        desired_dt=0.5,
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "nuplan_train": "/inspire/hdd/project/roboticsystem2/public/nuplan/dataset/nuplan-v1.1",
        },
    )
    
    # artifact = convert_unified_dataset_to_artifact(
    #     dataset=dataset,
    #     scene_idx=0,
    #     output_path="output/nuplan_train_1aa44d46e4ab5bc7.usdz",
    #     scene_id="nuplan_train_1aa44d46e4ab5bc7",
    # )
    
    artifact = Artifact(source='output/nuplan_train_1aa44d46e4ab5bc7.usdz')
    artifact_2 = Artifact(source='data/nre-artifacts/all-usdzs/sample_set/25.07_release/Batch0001/05bb8212-63e1-40a8-b4fc-3142c0e94646/05bb8212-63e1-40a8-b4fc-3142c0e94646.usdz')
    
    print("✓ Successfully loaded reference artifact")
    
    # Compare the two artifacts
    if artifact is None:
        print("\n" + "=" * 80)
        print("⚠ Unable to compare: converted artifact does not exist")
        print("=" * 80)
        print("\nShowing information for the reference artifact only:")
        print(f"  Scene ID: {artifact_2.scene_id}")
        print(f"  Rig trajectory length: {len(artifact_2.rig.trajectory)}")
        print(f"  Traffic objects: {len(artifact_2.traffic_objects)}")
        print(f"  World to NRE matrix:\n{artifact_2.rig.world_to_nre}")
        print(f"  First pose position: {artifact_2.rig.trajectory.poses[0].vec3}")
        exit(0)
    
    print("=" * 80)
    print("Artifact comparison analysis")
    print("=" * 80)
    
    print("\n[Basic information]")
    print("Artifact 1 (converted):")
    print(f"  Scene ID: {artifact.scene_id}")
    print(f"  Rig trajectory length: {len(artifact.rig.trajectory)}")
    print(f"  Traffic objects: {len(artifact.traffic_objects)}")
    
    print("\nArtifact 2 (reference case):")
    print(f"  Scene ID: {artifact_2.scene_id}")
    print(f"  Rig trajectory length: {len(artifact_2.rig.trajectory)}")
    print(f"  Traffic objects: {len(artifact_2.traffic_objects)}")
    
    print("\n[Rig trajectory analysis]")
    print("\nArtifact 1 Rig:")
    rig1 = artifact.rig
    print(f"  Sequence ID: {rig1.sequence_id}")
    print(f"  World to NRE matrix shape: {rig1.world_to_nre.shape}")
    print(f"  World to NRE matrix:\n{rig1.world_to_nre}")
    print(f"  Trajectory timestamps range: {rig1.trajectory.timestamps_us[0]} - {rig1.trajectory.timestamps_us[-1]}")
    print(f"  First pose position (vec3): {rig1.trajectory.poses[0].vec3}")
    print(f"  First pose quaternion: {rig1.trajectory.poses[0].quat}")
    print(f"  Last pose position (vec3): {rig1.trajectory.poses[-1].vec3}")
    print(f"  Last pose quaternion: {rig1.trajectory.poses[-1].quat}")
    
    print("\nArtifact 2 Rig:")
    rig2 = artifact_2.rig
    print(f"  Sequence ID: {rig2.sequence_id}")
    print(f"  World to NRE matrix shape: {rig2.world_to_nre.shape}")
    print(f"  World to NRE matrix:\n{rig2.world_to_nre}")
    print(f"  Trajectory timestamps range: {rig2.trajectory.timestamps_us[0]} - {rig2.trajectory.timestamps_us[-1]}")
    print(f"  First pose position (vec3): {rig2.trajectory.poses[0].vec3}")
    print(f"  First pose quaternion: {rig2.trajectory.poses[0].quat}")
    print(f"  Last pose position (vec3): {rig2.trajectory.poses[-1].vec3}")
    print(f"  Last pose quaternion: {rig2.trajectory.poses[-1].quat}")
    
    print("\n[Coordinate frame differences]")
    print("\nArtifact 1:")
    print(f"  World to NRE is identity: {np.allclose(rig1.world_to_nre, np.eye(4))}")
    print("  Trajectory position ranges:")
    positions1 = rig1.trajectory.poses.vec3
    print(f"    X: [{positions1[:, 0].min():.2f}, {positions1[:, 0].max():.2f}]")
    print(f"    Y: [{positions1[:, 1].min():.2f}, {positions1[:, 1].max():.2f}]")
    print(f"    Z: [{positions1[:, 2].min():.2f}, {positions1[:, 2].max():.2f}]")
    
    print("\nArtifact 2:")
    print(f"  World to NRE is identity: {np.allclose(rig2.world_to_nre, np.eye(4))}")
    print("  Trajectory position ranges:")
    positions2 = rig2.trajectory.poses.vec3
    print(f"    X: [{positions2[:, 0].min():.2f}, {positions2[:, 0].max():.2f}]")
    print(f"    Y: [{positions2[:, 1].min():.2f}, {positions2[:, 1].max():.2f}]")
    print(f"    Z: [{positions2[:, 2].min():.2f}, {positions2[:, 2].max():.2f}]")
    
    print("\n[Traffic object trajectory analysis]")
    if len(artifact.traffic_objects) > 0:
        sample_obj1 = list(artifact.traffic_objects.values())[0]
        print("\nArtifact 1 first traffic object:")
        print(f"  Track ID: {sample_obj1.track_id}")
        print(f"  Trajectory length: {len(sample_obj1.trajectory)}")
        print(f"  First pose position: {sample_obj1.trajectory.poses[0].vec3}")
        print(f"  Last pose position: {sample_obj1.trajectory.poses[-1].vec3}")
    
    if len(artifact_2.traffic_objects) > 0:
        sample_obj2 = list(artifact_2.traffic_objects.values())[0]
        print("\nArtifact 2 first traffic object:")
        print(f"  Track ID: {sample_obj2.track_id}")
        print(f"  Trajectory length: {len(sample_obj2.trajectory)}")
        print(f"  First pose position: {sample_obj2.trajectory.poses[0].vec3}")
        print(f"  Last pose position: {sample_obj2.trajectory.poses[-1].vec3}")
    
    print("\n" + "=" * 80)