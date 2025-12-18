# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
TrajdataDataSource: 直接从trajdata数据加载场景数据的实现

这个类展示了如何创建一个SceneDataSource实现，直接从trajdata转换好的数据加载，
而不需要USDZ格式。这对于使用trajdata数据集的研究者很有用。

使用示例:
    from trajdata import UnifiedDataset
    from alpasim_utils.trajdata_data_source import TrajdataDataSource
    
    # 加载trajdata数据集
    dataset = UnifiedDataset(
        desired_data=["nusc_mini"],
        data_dirs={"/path/to/trajdata/data"},
        ...
    )
    
    # 获取一个场景
    scene = dataset.get_scene("nusc_mini", "scene-0001")
    
    # 创建数据源
    data_source = TrajdataDataSource.from_trajdata_scene(scene)
    
    # 现在可以在Runtime中使用
    # artifacts = {data_source.scene_id: data_source}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

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

logger = logging.getLogger(__name__)


@dataclass
class TrajdataDataSource(SceneDataSource):
    """
    直接从trajdata数据加载场景数据的实现。
    
    这个类实现了SceneDataSource协议，允许直接从trajdata的Scene或AgentBatch对象
    加载数据，而不需要USDZ格式。
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
        base_timestamp_us: Optional[int] = None,
    ) -> TrajdataDataSource:
        """
        从trajdata的Scene对象创建TrajdataDataSource。
        
        Args:
            scene: trajdata的Scene对象
            dataset: UnifiedDataset实例（用于获取scene_cache和地图）
            scene_cache: 可选的EnvCache实例（如果不提供，将从dataset创建）
            scene_id: 可选的场景ID（如果不提供，使用scene.name）
            smooth_trajectories: 是否对轨迹进行平滑处理
            base_timestamp_us: 基础时间戳（微秒），如果为None则从0开始
        
        Returns:
            TrajdataDataSource实例
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
        从trajdata的AgentBatch对象创建TrajdataDataSource。
        
        注意：这个方法需要batch包含完整的场景信息。通常建议使用from_trajdata_scene。
        
        Args:
            batch: trajdata的AgentBatch对象
            scene_id: 场景ID
            smooth_trajectories: 是否对轨迹进行平滑处理
        
        Returns:
            TrajdataDataSource实例
        """
        if AgentBatch is None:
            raise ImportError("trajdata is not installed. Please install it to use TrajdataDataSource.")

        data_source = cls(scene_id=scene_id)
        data_source._smooth_trajectories = smooth_trajectories
        # 从batch中提取数据
        data_source._load_from_batch(batch)
        return data_source

    def _load_from_batch(self, batch: AgentBatch) -> None:
        """从AgentBatch加载数据（内部方法）"""
        # 这里需要根据batch的结构提取数据
        # 具体实现取决于你的trajdata数据格式
        raise NotImplementedError(
            "from_agent_batch需要根据你的trajdata数据格式实现。"
            "建议使用from_trajdata_scene方法。"
        )

    @property
    def scene_id(self) -> str:
        """场景ID"""
        if self._scene_id:
            return self._scene_id
        if self._scene is not None:
            return self._scene.name
        raise ValueError("scene_id未设置且无法从scene获取")

    @scene_id.setter
    def scene_id(self, value: str) -> None:
        self._scene_id = value
    
    def _get_scene_cache(self) -> EnvCache:
        """获取或创建scene_cache"""
        if self._scene_cache is not None:
            return self._scene_cache
        
        if self._scene is None:
            raise ValueError("无法创建scene_cache：scene未设置")
        
        if self._dataset is None:
            raise ValueError("无法创建scene_cache：dataset未设置")
        
        # 创建SceneCache
        self._scene_cache = self._dataset.cache_class(
            self._dataset.cache_path, self._scene, self._dataset.augmentations
        )
        self._scene_cache.set_obs_format(self._dataset.obs_format)
        return self._scene_cache

    def _extract_agent_trajectory(
        self,
        agent: AgentMetadata,
    ) -> tuple[Optional[Trajectory], Optional[VehicleConfig]]:
        """提取agent的完整轨迹（参考trajdata_artifact_converter.py的实现）"""
        if self._scene is None:
            return None, None
        
        scene_cache = self._get_scene_cache()
        dt = self._scene.dt
        base_timestamp_us = getattr(self, "_base_timestamp_us", None)
        
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
        """加载并返回ego车辆的Rig对象"""
        if self._rig is not None:
            return self._rig

        if self._scene is None:
            raise ValueError("无法加载rig：scene未设置")

        # 获取所有agents
        all_agents = self._scene.agents if self._scene.agents else []
        
        # 识别ego agent
        ego_agent = next((a for a in all_agents if a.name == "ego"), None)
        if ego_agent is None and len(all_agents) > 0:
            # 如果没有ego，使用第一个agent
            ego_agent = all_agents[0]
            logger.warning(f"No ego agent found, using first agent: {ego_agent.name}")

        if ego_agent is None:
            raise ValueError("场景中没有找到ego agent")

        # 提取ego轨迹
        ego_trajectory, ego_vehicle_config = self._extract_agent_trajectory(ego_agent)
        
        if ego_trajectory is None:
            raise ValueError("无法提取ego轨迹")

        # 计算 world_to_nre 变换矩阵（将第一个轨迹点作为原点）
        world_to_nre = np.eye(4)
        if len(ego_trajectory) > 0:
            first_pose_position = ego_trajectory.poses[0].vec3
            world_to_nre[:3, 3] = -first_pose_position
            logger.info(
                f"Setting world_to_nre origin at first pose: {first_pose_position}, "
                f"translation: {world_to_nre[:3, 3]}"
            )

        # 将ego轨迹转换为局部坐标（NRE）
        if len(ego_trajectory) > 0:
            translation = world_to_nre[:3, 3]
            local_positions = ego_trajectory.poses.vec3 + translation
            local_quat = ego_trajectory.poses.quat.copy()
            local_poses = QVec(vec3=local_positions, quat=local_quat)
            ego_trajectory = Trajectory(
                timestamps_us=ego_trajectory.timestamps_us.copy(),
                poses=local_poses,
            )

        # 提取相机信息（参考trajdata_artifact_converter.py）
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
        """从scene中提取相机信息（参考trajdata_artifact_converter.py）"""
        camera_ids = []
        camera_calibrations = {}
        
        if self._scene is None:
            return camera_ids, camera_calibrations
        
        # 检查是否存在 sensor_calibration 信息
        if not hasattr(self._scene, 'data_access_info') or not self._scene.data_access_info:
            logger.warning(f"scene.data_access_info 不存在，跳过相机信息提取")
            return camera_ids, camera_calibrations
        
        sensor_calibration = self._scene.data_access_info.get('sensor_calibration')
        if not sensor_calibration or not isinstance(sensor_calibration, dict):
            logger.warning(f"sensor_calibration 不存在或格式不正确，跳过相机信息提取")
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
                logger.warning(f"提取相机 {camera_name} 的信息时出错: {e}")
                continue
        
        if len(camera_ids) == 0:
            # 如果没有相机信息，创建一个默认的
            logger.warning(f"场景 {self.scene_id} 没有相机信息，使用默认相机")
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
        """判断对象是否为静态对象（基于速度）"""
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
        """加载并返回交通对象"""
        if self._traffic_objects is not None:
            return self._traffic_objects

        if self._scene is None:
            raise ValueError("无法加载traffic_objects：scene未设置")

        # 获取所有agents
        all_agents = self._scene.agents if self._scene.agents else []
        
        # 识别ego agent
        ego_agent = next((a for a in all_agents if a.name == "ego"), None)
        if ego_agent is None and len(all_agents) > 0:
            ego_agent = all_agents[0]

        traffic_dict = {}
        for agent in all_agents:
            # 跳过ego agent
            if agent.name == "ego" or agent == ego_agent:
                continue

            # 提取轨迹
            trajectory, _ = self._extract_agent_trajectory(agent)
            
            # 过滤掉空轨迹或只有1个数据点的轨迹
            if trajectory is None or len(trajectory) < 2:
                continue

            # 将轨迹转换为局部坐标（NRE）- 使用rig的world_to_nre
            if self._rig is None:
                # 如果rig还没加载，先加载它
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

            # 如果需要平滑
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

            # 获取AABB
            if hasattr(agent.extent, 'length'):
                aabb = AABB(x=agent.extent.length, y=agent.extent.width, z=agent.extent.height)
            else:
                # 默认AABB
                aabb = AABB(x=4.5, y=1.8, z=1.5)

            # 判断是否为静态对象
            is_static = self._is_static_object(trajectory)

            # 获取类别标签
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
        """加载并返回VectorMap（从 dataset._map_api 获取）"""
        if self._map is not None:
            return self._map

        if VectorMap is None:
            logger.warning("trajdata未安装，无法加载地图")
            return None

        if self._scene is None:
            logger.warning("无法加载地图：scene未设置")
            return None

        if self._dataset is None:
            logger.warning("无法加载地图：dataset未设置")
            return None

        # 从 dataset._map_api 获取地图（参考 trajdata_artifact_converter.py）
        try:
            # 检查 dataset 是否包含地图支持
            if (
                not hasattr(self._dataset, 'incl_vector_map')
                or not self._dataset.incl_vector_map
                or not hasattr(self._dataset, '_map_api')
                or self._dataset._map_api is None
            ):
                logger.debug(f"Dataset 未启用地图支持或 map_api 不可用")
                return None

            # 构建地图名称："{env_name}:{location}"
            if not hasattr(self._scene, 'location') or not self._scene.location:
                logger.debug(f"Scene {self.scene_id} 没有 location 信息，无法加载地图")
                return None

            map_name = f"{self._scene.env_name}:{self._scene.location}"
            
            # 获取 vector_map_params（如果存在）
            vector_map_params = {}
            if hasattr(self._dataset, 'vector_map_params'):
                vector_map_params = self._dataset.vector_map_params

            # 从 map_api 获取地图
            vec_map = self._dataset._map_api.get_map(map_name, **vector_map_params)
            
            if vec_map is None:
                logger.debug(f"场景 {self.scene_id} (map_name: {map_name}) 没有地图数据")
                return None

            # trajdata的VectorMap应该可以直接使用
            self._map = vec_map

            # 如果map需要finalize，调用它
            if hasattr(self._map, "__post_init__"):
                self._map.__post_init__()
            if hasattr(self._map, "compute_search_indices"):
                self._map.compute_search_indices()

            # 修复数据类型（如果需要）
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

            logger.info(f"成功加载地图: {map_name}")
            return self._map
        except Exception as e:
            logger.error(f"加载地图时出错: {e}", exc_info=True)
            return None

    @property
    def metadata(self) -> Metadata:
        """创建并返回Metadata对象"""
        if self._metadata is not None:
            return self._metadata

        # 从scene提取元数据
        scene_id = self.scene_id
        
        # 确保rig已加载
        rig = self.rig
        
        # 从rig中提取相机ID列表
        camera_id_names = []
        if rig and rig.camera_ids:
            camera_id_names = [camera_id.logical_name for camera_id in rig.camera_ids]
        
        # 计算时间范围
        if self._scene is not None:
            dt = self._scene.dt
            length_timesteps = self._scene.length_timesteps
            base_timestamp_us = getattr(self, "_base_timestamp_us", 0)
            time_range_start = float(base_timestamp_us) / 1e6
            time_range_end = float(base_timestamp_us + length_timesteps * dt * 1e6) / 1e6
        else:
            time_range_start = float(rig.trajectory.time_range_us.start) / 1e6
            time_range_end = float(rig.trajectory.time_range_us.stop) / 1e6

        # 创建metadata
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
    base_timestamp_us: Optional[int] = None,
) -> dict[str, TrajdataDataSource]:
    """
    从trajdata UnifiedDataset创建TrajdataDataSource字典。
    
    Args:
        dataset: UnifiedDataset实例
        scene_indices: 要加载的场景索引列表（如果为None，加载所有场景）
        smooth_trajectories: 是否对轨迹进行平滑处理
        base_timestamp_us: 基础时间戳（微秒）
    
    Returns:
        dict[scene_id, TrajdataDataSource]
    """
    if UnifiedDataset is None:
        raise ImportError("trajdata is not installed. Please install it to use this function.")
    
    data_sources = {}
    
    if scene_indices is None:
        # 获取所有场景的索引
        # 使用 dataset._scene_index 来获取场景数量（参考 trajdata_artifact_converter.py）
        # len(dataset) 可能不等于场景数量，所以使用 _scene_index
        if hasattr(dataset, '_scene_index'):
            num_scenes = len(dataset._scene_index)
            logger.info(f"从 dataset._scene_index 获取到 {num_scenes} 个场景")
        elif hasattr(dataset, '__len__'):
            # 备选方案：如果 _scene_index 不存在，尝试使用 __len__
            try:
                num_scenes = len(dataset)
                logger.warning(
                    "使用 len(dataset) 获取场景数量，这可能不准确。"
                    "建议检查 dataset._scene_index 是否可用。"
                )
            except (TypeError, NotImplementedError):
                logger.error("无法确定场景数量：dataset 既没有 _scene_index 也不支持 __len__")
                return data_sources
        else:
            logger.error("无法确定场景数量：dataset 没有 _scene_index 属性且不支持 __len__")
            return data_sources
        
        if num_scenes == 0:
            logger.warning("Dataset 中没有找到场景，请检查 dataset 配置")
            return data_sources
        
        scene_indices = list(range(num_scenes))
        logger.info(f"将加载所有 {num_scenes} 个场景（索引: 0 到 {num_scenes-1}）")
    
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
