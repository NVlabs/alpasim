# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
转换函数：将 trajdata UnifiedDataset 转换为 alpasim Artifact 格式
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
    将 UnifiedDataset 中的场景转换为 Artifact 格式并保存为 USDZ 文件。

    Args:
        dataset: UnifiedDataset 实例
        scene_idx: 要转换的场景索引
        output_path: 输出 USDZ 文件路径（必须以 .usdz 结尾）
        scene_id: 场景 ID（如果为 None，则从 scene.name 生成）
        version_string: 版本字符串
        base_timestamp_us: 基础时间戳（微秒），如果为 None 则从 0 开始
        include_map: 是否包含地图数据

    Returns:
        Artifact 实例
    """
    if not output_path.endswith(".usdz"):
        raise ValueError("output_path must end with .usdz")

    # 1. 获取 Scene 对象
    scene = dataset.get_scene(scene_idx)
    scene_path = dataset._scene_index[scene_idx]
    
    # 创建 SceneCache
    scene_cache = dataset.cache_class(dataset.cache_path, scene, dataset.augmentations)
    scene_cache.set_obs_format(dataset.obs_format)

    # 2. 生成 scene_id
    if scene_id is None:
        scene_id = f"trajdata-{scene.env_name}-{scene.name}"

    # 3. 提取所有 agents 的完整轨迹
    dt = scene.dt
    length_timesteps = scene.length_timesteps
    
    # 获取所有 agents
    all_agents = scene.agents if scene.agents else []
    
    # 识别 ego agent
    ego_agent = next((a for a in all_agents if a.name == "ego"), None)
    if ego_agent is None and len(all_agents) > 0:
        # 如果没有 ego，使用第一个 agent
        ego_agent = all_agents[0]
        logger.warning(f"No ego agent found, using first agent: {ego_agent.name}")

    # 4. 提取 ego 轨迹
    ego_trajectory = None
    ego_vehicle_config = None
    if ego_agent:
        ego_trajectory, ego_vehicle_config = _extract_agent_trajectory(
            scene, scene_cache, ego_agent, dt, base_timestamp_us
        )

    # 5. 计算 world_to_nre 变换矩阵（将第一个轨迹点作为原点）
    world_to_nre = np.eye(4)
    if ego_trajectory and len(ego_trajectory) > 0:
        # 获取第一个轨迹点的位置（全局坐标）
        first_pose_position = ego_trajectory.poses[0].vec3
        
        # 设置 world_to_nre 矩阵：将第一个轨迹点作为原点
        # world_to_nre 的平移部分 = -第一个轨迹点的位置
        world_to_nre[:3, 3] = -first_pose_position
        
        logger.info(
            f"Setting world_to_nre origin at first pose: {first_pose_position}, "
            f"translation: {world_to_nre[:3, 3]}"
        )
    else:
        logger.warning("No ego trajectory found, using identity world_to_nre matrix")

    # 6. 将 ego 轨迹转换为局部坐标（NRE）
    if ego_trajectory and len(ego_trajectory) > 0:
        ego_trajectory = _transform_trajectory_to_local(ego_trajectory, world_to_nre)
        logger.info(
            f"Transformed ego trajectory to local coordinates. "
            f"First pose: {ego_trajectory.poses[0].vec3}"
        )

    # 7. 提取其他 agents 的轨迹（traffic objects）
    traffic_objects = {}
    for agent in all_agents:
        if agent.name == "ego" or agent == ego_agent:
            continue
        
        trajectory, _ = _extract_agent_trajectory(
            scene, scene_cache, agent, dt, base_timestamp_us
        )
        
        # 过滤掉空轨迹或只有1个数据点的轨迹（平滑处理需要至少2个点）
        if trajectory is None or len(trajectory) < 2:
            continue

        # 将轨迹转换为局部坐标（NRE）
        trajectory = _transform_trajectory_to_local(trajectory, world_to_nre)

        # 获取 agent 的 extent
        extent = agent.extent
        if hasattr(extent, 'length'):
            aabb = AABB(x=extent.length, y=extent.width, z=extent.height)
        else:
            # 默认尺寸
            aabb = AABB(x=4.5, y=1.8, z=1.5)

        # 判断是否为静态对象（简单启发式：速度很小）
        is_static = _is_static_object(trajectory)

        # 获取 agent 类型标签
        label_class = agent.type.name if hasattr(agent.type, 'name') else "UNKNOWN"

        traffic_objects[agent.name] = TrafficObject(
            track_id=agent.name,
            aabb=aabb,
            trajectory=trajectory,
            is_static=is_static,
            label_class=label_class,
        )

    # 8. 创建 Rig 对象
    rig = None
    if ego_trajectory and len(ego_trajectory) > 0:
        
        # 从 scene.data_access_info['sensor_calibration'] 中提取相机信息
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
        
        # 将 camera_calibrations 存储在 rig 对象中（用于后续写入 USDZ）
        rig._camera_calibrations = camera_calibrations

    # 9. 创建 TrafficObjects
    traffic_objects_dict = TrafficObjects(traffic_objects)

    # 10. 创建 Metadata
    if base_timestamp_us is None:
        base_timestamp_us = 0
    
    time_range_start = float(base_timestamp_us) / 1e6
    time_range_end = float(base_timestamp_us + length_timesteps * dt * 1e6) / 1e6

    # 从 rig 中提取相机 ID 列表（如果存在）
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

    # 11. 创建 USDZ 文件
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

    # 12. 返回 Artifact 实例
    return Artifact(source=output_path)


def _extract_camera_info_from_scene(
    scene: Scene,
    scene_id: str,
) -> tuple[list[CameraId], dict]:
    """
    从 scene.data_access_info['sensor_calibration'] 中提取相机信息。
    
    Args:
        scene: Scene 对象
        scene_id: 场景 ID
        
    Returns:
        (camera_ids, camera_calibrations) 元组
        - camera_ids: CameraId 对象列表
        - camera_calibrations: 相机校准信息字典，格式与 artifact_2 一致
    """
    camera_ids = []
    camera_calibrations = {}
    
    # 检查是否存在 sensor_calibration 信息
    if not hasattr(scene, 'data_access_info') or not scene.data_access_info:
        logger.warning(f"scene.data_access_info 不存在，跳过相机信息提取")
        return camera_ids, camera_calibrations
    
    sensor_calibration = scene.data_access_info.get('sensor_calibration')
    if not sensor_calibration:
        logger.warning(f"scene.data_access_info['sensor_calibration'] 不存在，跳过相机信息提取")
        return camera_ids, camera_calibrations
    
    # 遍历 sensor_calibration 中的相机信息
    # sensor_calibration 可能是一个字典，键是相机名称，值是校准信息
    if not isinstance(sensor_calibration, dict):
        logger.warning(f"sensor_calibration 不是字典类型，跳过相机信息提取")
        return camera_ids, camera_calibrations
    
    unique_sensor_idx = 0
    for camera_name, calibration_info in sensor_calibration['cameras'].items():
        try:
            # 构建 unique_camera_id（格式：logical_sensor_name@sequence_id）
            unique_camera_id = f"{camera_name}@{scene_id}"
            
            # 提取位置和旋转信息
            # 假设 calibration_info 包含 position 和 rotation 字段
            # rotation 是四元数，顺序为 [x, y, z, w]
            position = calibration_info.get('sensor2ego_translation', [0.0, 0.0, 0.0])
            rotation = calibration_info.get('sensor2ego_rotation', [0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
            
            # 确保 position 和 rotation 是列表或数组
            if isinstance(position, (int, float)):
                position = [float(position), 0.0, 0.0]
            elif len(position) < 3:
                position = list(position) + [0.0] * (3 - len(position))
            
            if isinstance(rotation, (int, float)):
                rotation = [0.0, 0.0, 0.0, 1.0]
            elif len(rotation) < 4:
                # 如果只有3个值，假设是欧拉角，转换为四元数
                if len(rotation) == 3:
                    r = R.from_euler('xyz', rotation)
                    rotation = r.as_quat()  # [x, y, z, w]
                else:
                    rotation = [0.0, 0.0, 0.0, 1.0]
            
            # 确保 rotation 是 [x, y, z, w] 格式
            if len(rotation) == 4:
                qx, qy, qz, qw = rotation[0], rotation[1], rotation[2], rotation[3]
            else:
                logger.warning(f"相机 {camera_name} 的 rotation 格式不正确，使用默认值")
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            
            # 构建 T_sensor_rig 变换矩阵（4x4）
            # 从四元数构建旋转矩阵
            r_quat = R.from_quat([qx, qy, qz, qw])
            rotation_matrix = r_quat.as_matrix()
            
            # 构建完整的变换矩阵
            T_sensor_rig = np.eye(4)
            T_sensor_rig[:3, :3] = rotation_matrix
            T_sensor_rig[:3, 3] = position[:3]
            
            # 创建 CameraId 对象
            camera_id = CameraId(
                logical_name=camera_name,
                trajectory_idx=0,
                sequence_id=scene_id,
                unique_id=unique_camera_id,
            )
            camera_ids.append(camera_id)
            
            # 构建 camera_calibration 字典
            # 提取相机内参（如果存在）
            camera_model = calibration_info.get('camera_model', {})
            if not camera_model:
                # 如果没有提供相机模型，使用默认值
                camera_model = {
                    "type": "pinhole",  # 默认使用 pinhole 模型
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
            logger.info(f"成功提取相机信息: {camera_name} (unique_id: {unique_camera_id})")
            
        except Exception as e:
            logger.warning(f"提取相机 {camera_name} 的信息时出错: {e}")
            continue
    
    logger.info(f"共提取 {len(camera_ids)} 个相机信息")
    return camera_ids, camera_calibrations


def _extract_agent_trajectory(
    scene: Scene,
    scene_cache,
    agent: AgentMetadata,
    dt: float,
    base_timestamp_us: Optional[int],
) -> tuple[Optional[Trajectory], Optional[VehicleConfig]]:
    """
    提取 agent 的完整轨迹。

    Returns:
        (Trajectory, VehicleConfig) 或 (None, None) 如果无法提取
    """
    try:
        timestamps_us = []
        poses_vec3 = []
        poses_quat = []

        # 遍历所有时间步
        for ts in range(agent.first_timestep, agent.last_timestep + 1):
            try:
                state = scene_cache.get_raw_state(agent.name, ts)
                
                # 获取位置和朝向
                x = state.get_attr("x") if hasattr(state, 'get_attr') else state.x
                y = state.get_attr("y") if hasattr(state, 'get_attr') else state.y
                z = state.get_attr("z") if hasattr(state, 'get_attr') else (state.z if hasattr(state, 'z') else 0.0)
                heading = state.get_attr("h") if hasattr(state, 'get_attr') else state.h

                # 转换为 numpy 数组（处理标量情况）
                if isinstance(x, (int, float)):
                    x = np.array([x])
                if isinstance(y, (int, float)):
                    y = np.array([y])
                if isinstance(z, (int, float)):
                    z = np.array([z])
                if isinstance(heading, (int, float)):
                    heading = np.array([heading])

                # 取第一个元素（如果是数组）
                x_val = float(x[0] if x.ndim > 0 else x)
                y_val = float(y[0] if y.ndim > 0 else y)
                z_val = float(z[0] if z.ndim > 0 else z)
                heading_val = float(heading[0] if heading.ndim > 0 else heading)

                # 计算时间戳
                if base_timestamp_us is None:
                    timestamp_us = int(ts * dt * 1e6)
                else:
                    timestamp_us = int(base_timestamp_us + ts * dt * 1e6)

                timestamps_us.append(timestamp_us)
                poses_vec3.append([x_val, y_val, z_val])

                # 将 heading 转换为四元数
                # heading 是弧度，绕 z 轴旋转
                quat = R.from_euler('z', heading_val).as_quat()  # [x, y, z, w]
                poses_quat.append(quat)

            except Exception as e:
                logger.debug(f"Failed to get state for agent {agent.name} at ts {ts}: {e}")
                continue

        if len(timestamps_us) == 0:
            return None, None

        # 创建 QVec
        poses = QVec(
            vec3=np.array(poses_vec3, dtype=np.float32),
            quat=np.array(poses_quat, dtype=np.float32),
        )

        # 创建 Trajectory
        trajectory = Trajectory(
            timestamps_us=np.array(timestamps_us, dtype=np.uint64),
            poses=poses,
        )

        # 创建 VehicleConfig（从 extent 提取）
        vehicle_config = None
        if hasattr(agent.extent, 'length'):
            vehicle_config = VehicleConfig(
                aabb_x_m=agent.extent.length,
                aabb_y_m=agent.extent.width,
                aabb_z_m=agent.extent.height,
                aabb_x_offset_m=-agent.extent.length / 2,  # 简化假设
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
    将轨迹从全局坐标系转换为局部坐标系（NRE）。
    
    Args:
        trajectory: 全局坐标系中的轨迹
        world_to_nre: 从全局坐标系到NRE坐标系的变换矩阵 (4x4)
    
    Returns:
        局部坐标系中的轨迹
    """
    # 提取平移部分
    translation = world_to_nre[:3, 3]
    
    # 转换位置（vec3）
    local_positions = trajectory.poses.vec3 + translation
    
    # 四元数保持不变（假设只有平移，没有旋转）
    # 如果 world_to_nre 包含旋转，需要应用旋转到四元数
    # 这里简化处理，假设只有平移
    local_quat = trajectory.poses.quat.copy()
    
    # 创建新的 QVec
    local_poses = QVec(
        vec3=local_positions,
        quat=local_quat,
    )
    
    # 创建新的 Trajectory
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
    将地图信息写入 USDZ 文件。
    
    优先从 UnifiedDataset 的 map_api 获取 VectorMap，然后：
    1. 尝试将 VectorMap 序列化为 protobuf 格式存储（推荐方式）
    2. 如果失败，尝试查找原始 XODR 文件作为备选
    
    注意：artifact.py 需要支持从 protobuf 加载地图，否则只能使用 XODR。
    """
    map_written = False
    
    # 方法1: 直接从 UnifiedDataset 获取 VectorMap 并序列化为 protobuf
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
                    # 将 VectorMap 序列化为 protobuf
                    try:
                        import trajdata.proto.vectorized_map_pb2 as map_proto
                        vec_map_proto = vector_map.to_proto()
                        proto_bytes = vec_map_proto.SerializeToString()
                        
                        # 存储为 map.pb
                        zip_file.writestr("map.pb", proto_bytes)
                        logger.info(f"Successfully wrote VectorMap as protobuf (map.pb) for {map_name}")
                        map_written = True
                    except Exception as e:
                        logger.debug(f"Failed to serialize VectorMap to protobuf: {e}")
    except Exception as e:
        logger.debug(f"Could not get VectorMap from dataset API: {e}")
    
    # 方法2: 如果 protobuf 序列化失败，尝试查找原始 XODR 文件（作为备选）
    if not map_written:
        xodr_content = None
        
        # 尝试从原始数据目录查找 XODR 文件
        try:
            if hasattr(scene, 'location') and scene.location:
                # nuplan 的 XODR 文件通常在 data_dir.parent / "maps" / "{location}.xodr"
                if hasattr(dataset, 'data_dirs') and dataset.data_dirs:
                    for env_name, data_dir in dataset.data_dirs.items():
                        if env_name == scene.env_name:
                            # 根据 nuplan_dataset.py:381，map_root = data_dir.parent / "maps"
                            xodr_path = Path(data_dir).parent / "maps" / f"{scene.location}.xodr"
                            if xodr_path.exists():
                                xodr_content = xodr_path.read_text(encoding='utf-8')
                                logger.info(f"Found XODR file at {xodr_path}")
                                break
        except Exception as e:
            logger.debug(f"Could not find XODR file in data directories: {e}")
        
        # 如果找到 XODR，写入 USDZ
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
    判断对象是否为静态对象（基于速度）。
    """
    if len(trajectory) < 2:
        return True

    # 计算平均速度
    positions = trajectory.poses.vec3
    timestamps = trajectory.timestamps_us.astype(np.float64) / 1e6  # 转换为秒

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
    创建 USDZ 文件（实际上是 ZIP 文件）。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. 写入 metadata.yaml
        metadata_dict = metadata.to_dict()
        yaml_content = yaml.dump(metadata_dict, default_flow_style=False)
        zip_file.writestr("metadata.yaml", yaml_content)

        # 2. 写入 rig_trajectories.json
        if rig:
            # 获取相机校准信息（如果存在）
            camera_calibrations = getattr(rig, '_camera_calibrations', {})
            
            # 构建 cameras_frame_timestamps_us 字典
            cameras_frame_timestamps_us = {}
            if rig.camera_ids:
                # 为每个相机使用 rig 轨迹的时间戳
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
                        "rig_bbox": None,  # 可以后续添加
                    }
                ],
                "T_world_base": np.eye(4).tolist(),  # 默认单位矩阵
            }

            # 如果有 vehicle_config，添加 rig_bbox
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

        # 3. 写入 sequence_tracks.json
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
            
            # 将 QVec 转换为 [x, y, z, qx, qy, qz, qw] 格式
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

        # 4. 写入地图（如果有）
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
    
    print("✓ 成功加载官方用例 artifact")
    
    # 比较两个 artifact
    if artifact is None:
        print("\n" + "=" * 80)
        print("⚠ 无法进行比较：转换得到的 artifact 不存在")
        print("=" * 80)
        print("\n只显示官方用例的信息：")
        print(f"  Scene ID: {artifact_2.scene_id}")
        print(f"  Rig trajectory length: {len(artifact_2.rig.trajectory)}")
        print(f"  Traffic objects: {len(artifact_2.traffic_objects)}")
        print(f"  World to NRE matrix:\n{artifact_2.rig.world_to_nre}")
        print(f"  First pose position: {artifact_2.rig.trajectory.poses[0].vec3}")
        exit(0)
    
    print("=" * 80)
    print("Artifact 比较分析")
    print("=" * 80)
    
    print("\n【基本信息】")
    print(f"Artifact 1 (转换得到的):")
    print(f"  Scene ID: {artifact.scene_id}")
    print(f"  Rig trajectory length: {len(artifact.rig.trajectory)}")
    print(f"  Traffic objects: {len(artifact.traffic_objects)}")
    
    print(f"\nArtifact 2 (官方用例):")
    print(f"  Scene ID: {artifact_2.scene_id}")
    print(f"  Rig trajectory length: {len(artifact_2.rig.trajectory)}")
    print(f"  Traffic objects: {len(artifact_2.traffic_objects)}")
    
    print("\n【Rig 轨迹分析】")
    print(f"\nArtifact 1 Rig:")
    rig1 = artifact.rig
    print(f"  Sequence ID: {rig1.sequence_id}")
    print(f"  World to NRE matrix shape: {rig1.world_to_nre.shape}")
    print(f"  World to NRE matrix:\n{rig1.world_to_nre}")
    print(f"  Trajectory timestamps range: {rig1.trajectory.timestamps_us[0]} - {rig1.trajectory.timestamps_us[-1]}")
    print(f"  First pose position (vec3): {rig1.trajectory.poses[0].vec3}")
    print(f"  First pose quaternion: {rig1.trajectory.poses[0].quat}")
    print(f"  Last pose position (vec3): {rig1.trajectory.poses[-1].vec3}")
    print(f"  Last pose quaternion: {rig1.trajectory.poses[-1].quat}")
    
    print(f"\nArtifact 2 Rig:")
    rig2 = artifact_2.rig
    print(f"  Sequence ID: {rig2.sequence_id}")
    print(f"  World to NRE matrix shape: {rig2.world_to_nre.shape}")
    print(f"  World to NRE matrix:\n{rig2.world_to_nre}")
    print(f"  Trajectory timestamps range: {rig2.trajectory.timestamps_us[0]} - {rig2.trajectory.timestamps_us[-1]}")
    print(f"  First pose position (vec3): {rig2.trajectory.poses[0].vec3}")
    print(f"  First pose quaternion: {rig2.trajectory.poses[0].quat}")
    print(f"  Last pose position (vec3): {rig2.trajectory.poses[-1].vec3}")
    print(f"  Last pose quaternion: {rig2.trajectory.poses[-1].quat}")
    
    print("\n【坐标系差异分析】")
    print(f"\nArtifact 1:")
    print(f"  World to NRE 是否为单位矩阵: {np.allclose(rig1.world_to_nre, np.eye(4))}")
    print(f"  轨迹位置范围:")
    positions1 = rig1.trajectory.poses.vec3
    print(f"    X: [{positions1[:, 0].min():.2f}, {positions1[:, 0].max():.2f}]")
    print(f"    Y: [{positions1[:, 1].min():.2f}, {positions1[:, 1].max():.2f}]")
    print(f"    Z: [{positions1[:, 2].min():.2f}, {positions1[:, 2].max():.2f}]")
    
    print(f"\nArtifact 2:")
    print(f"  World to NRE 是否为单位矩阵: {np.allclose(rig2.world_to_nre, np.eye(4))}")
    print(f"  轨迹位置范围:")
    positions2 = rig2.trajectory.poses.vec3
    print(f"    X: [{positions2[:, 0].min():.2f}, {positions2[:, 0].max():.2f}]")
    print(f"    Y: [{positions2[:, 1].min():.2f}, {positions2[:, 1].max():.2f}]")
    print(f"    Z: [{positions2[:, 2].min():.2f}, {positions2[:, 2].max():.2f}]")
    
    print("\n【Traffic Objects 轨迹分析】")
    if len(artifact.traffic_objects) > 0:
        sample_obj1 = list(artifact.traffic_objects.values())[0]
        print(f"\nArtifact 1 第一个 traffic object:")
        print(f"  Track ID: {sample_obj1.track_id}")
        print(f"  Trajectory length: {len(sample_obj1.trajectory)}")
        print(f"  First pose position: {sample_obj1.trajectory.poses[0].vec3}")
        print(f"  Last pose position: {sample_obj1.trajectory.poses[-1].vec3}")
    
    if len(artifact_2.traffic_objects) > 0:
        sample_obj2 = list(artifact_2.traffic_objects.values())[0]
        print(f"\nArtifact 2 第一个 traffic object:")
        print(f"  Track ID: {sample_obj2.track_id}")
        print(f"  Trajectory length: {len(sample_obj2.trajectory)}")
        print(f"  First pose position: {sample_obj2.trajectory.poses[0].vec3}")
        print(f"  Last pose position: {sample_obj2.trajectory.poses[-1].vec3}")
    
    print("\n" + "=" * 80)