# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from collections import defaultdict
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from trajdata.caching import SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType
from trajdata.data_structures.scene import SceneTime, SceneTimeAgent
from trajdata.data_structures.state import StateArray
from trajdata.maps import MapAPI, RasterizedMapPatch, VectorMap
from trajdata.utils.arr_utils import (
    get_close_lanes,
    get_close_road_edges,
    transform_xyh_np,
)
from trajdata.utils.map_utils import LaneSegRelation
from trajdata.utils.state_utils import convert_to_frame_state, transform_from_frame


class AgentBatchElement:
    """A single element of an agent-centric batch."""

    # @profile
    def __init__(
        self,
        cache: SceneCache,
        data_index: int,
        scene_time_agent: SceneTimeAgent,
        history_sec: Tuple[Optional[float], Optional[float]],
        future_sec: Tuple[Optional[float], Optional[float]],
        agent_interaction_distances: Dict[
            Tuple[AgentType, AgentType], float
        ] = defaultdict(lambda: np.inf),
        incl_robot_future: bool = False,
        incl_raster_map: bool = False,
        raster_map_params: Optional[Dict[str, Any]] = None,
        map_api: Optional[MapAPI] = None,
        vector_map_params: Optional[Dict[str, Any]] = None,
        state_format: Optional[str] = None,
        standardize_data: bool = False,
        standardize_derivatives: bool = False,
        max_neighbor_num: Optional[int] = None,
        lane_graph_cache: Optional[dict] = None,
    ) -> None:
        self.cache: SceneCache = cache
        self.data_index: int = data_index
        self.dt: float = scene_time_agent.scene.dt
        self.scene_ts: int = scene_time_agent.ts
        self.history_sec = history_sec
        self.future_sec = future_sec

        agent_info: AgentMetadata = scene_time_agent.agent
        self.agent_name: str = agent_info.name
        self.agent_type: AgentType = agent_info.type
        self.max_neighbor_num = max_neighbor_num

        raw_state: StateArray = cache.get_raw_state(agent_info.name, self.scene_ts)
        if state_format is not None:
            self.curr_agent_state_np = raw_state.as_format(state_format)
        else:
            self.curr_agent_state_np = raw_state

        incl_z = self.curr_agent_state_np.has_attr("position3d")

        self.standardize_data = standardize_data
        if self.standardize_data:
            # Request cache to return observations relative to current agent
            obs_frame: StateArray = convert_to_frame_state(
                raw_state,
                stationary=not standardize_derivatives,
                grounded=True,
            )
            cache.set_obs_frame(obs_frame)

            # Create and store 2d tranformation matrix to agent from world
            agent_pos = self.curr_agent_state_np.position
            agent_heading_vector = self.curr_agent_state_np.heading_vector
            cos_agent, sin_agent = agent_heading_vector[0], agent_heading_vector[1]
            if incl_z:
                world_from_agent_tf: np.ndarray = np.array(
                    [
                        [cos_agent, -sin_agent, 0.0, agent_pos[0]],
                        [sin_agent, cos_agent, 0.0, agent_pos[1]],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            else:
                world_from_agent_tf: np.ndarray = np.array(
                    [
                        [cos_agent, -sin_agent, agent_pos[0]],
                        [sin_agent, cos_agent, agent_pos[1]],
                        [0.0, 0.0, 1.0],
                    ]
                )
            self.agent_from_world_tf: np.ndarray = np.linalg.inv(world_from_agent_tf)
        else:
            self.agent_from_world_tf: np.ndarray = np.eye(4 if incl_z else 3)

        ### AGENT-SPECIFIC DATA ###
        self.agent_history_np, self.agent_history_extent_np = self.get_agent_history(
            agent_info, history_sec
        )
        self.agent_history_len: int = self.agent_history_np.shape[0]

        self.agent_future_np, self.agent_future_extent_np = self.get_agent_future(
            agent_info, future_sec
        )
        self.agent_future_len: int = self.agent_future_np.shape[0]
        self.agent_meta_dict: Dict = get_agent_meta_dict(self.cache, agent_info)

        ### NEIGHBOR-SPECIFIC DATA ###
        def distance_limit(agent_types: np.ndarray, target_type: int) -> np.ndarray:
            return np.array(
                [
                    agent_interaction_distances[(agent_type, target_type)]
                    for agent_type in agent_types
                ]
            )

        nearby_agents, self.neighbor_types_np = self.get_nearby_agents(
            scene_time_agent, agent_info, distance_limit
        )

        self.num_neighbors = len(nearby_agents)
        (
            self.neighbor_histories,
            self.neighbor_history_extents,
            self.neighbor_history_lens_np,
        ) = self.get_neighbor_history(history_sec, nearby_agents)

        (
            self.neighbor_futures,
            self.neighbor_future_extents,
            self.neighbor_future_lens_np,
        ) = self.get_neighbor_future(future_sec, nearby_agents)

        self.neighbor_meta_dicts: Dict = [
            get_agent_meta_dict(self.cache, agent) for agent in nearby_agents
        ]

        ### ROBOT DATA ###
        self.robot_future_np: Optional[StateArray] = None

        if incl_robot_future:
            self.robot_future_np: StateArray = self.get_robot_current_and_future(
                scene_time_agent.robot, future_sec
            )

            # -1 because this is meant to hold the number of future steps
            # (whereas the above returns the current + future, yielding
            # one more timestep).
            self.robot_future_len: int = self.robot_future_np.shape[0] - 1

        ### MAP ###
        self.map_name: Optional[str] = None
        self.map_patch: Optional[RasterizedMapPatch] = None

        map_name: str = (
            f"{scene_time_agent.scene.env_name}:{scene_time_agent.scene.location}"
        )
        if incl_raster_map:
            self.map_name = map_name
            self.map_patch = self.get_agent_map_patch(raster_map_params)

        self.vec_map: Optional[VectorMap] = None
        if map_api is not None:
            self.vec_map = map_api.get_map(
                map_name,
                (
                    self.cache
                    if self.cache.is_traffic_light_data_cached(
                        # Is the original dt cached? If so, we can continue by
                        # interpolating time to get whatever the user desires.
                        self.cache.scene.env_metadata.dt
                    )
                    else None
                ),
                **vector_map_params if vector_map_params is not None else None,
            )
            if vector_map_params.get("calc_lane_graph", False):
                # not tested
                ego_xyh = np.concatenate(
                    [
                        self.curr_agent_state_np.position,
                        self.curr_agent_state_np.heading,
                    ]
                )
                num_pts = vector_map_params.get("num_lane_pts", 30)
                max_num_lanes = vector_map_params.get("max_num_lanes", 20)
                remove_single_successor = vector_map_params.get(
                    "remove_single_successor", False
                )
                radius = vector_map_params.get("radius", 100)
                (
                    self.num_lanes,
                    self.lane_xyh,
                    self.lane_adj,
                    self.lane_ids,
                    self.road_edge_xyzh,
                ) = gen_lane_graph(
                    self.vec_map,
                    ego_xyh,
                    self.agent_from_world_tf,
                    num_pts,
                    max_num_lanes,
                    radius,
                    remove_single_successor=remove_single_successor,
                    get_road_edges=vector_map_params.get("incl_road_edges", False),
                    lane_graph_cache=lane_graph_cache,
                )

            else:
                self.lane_xyh = None
                self.lane_adj = None
                self.lane_ids = list()
                self.num_lanes = 0
                self.road_edge_xyzh = None

        self.scene_id = scene_time_agent.scene.name

        # Will be optionally populated by the user's provided functions.
        self.extras: Dict[str, np.ndarray] = dict()

    def get_agent_history(
        self,
        agent_info: AgentMetadata,
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[StateArray, np.ndarray]:
        agent_history_np, agent_extent_history_np = self.cache.get_agent_history(
            agent_info, self.scene_ts, history_sec
        )
        return agent_history_np, agent_extent_history_np

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[StateArray, np.ndarray]:
        agent_future_np, agent_extent_future_np = self.cache.get_agent_future(
            agent_info, self.scene_ts, future_sec
        )
        return agent_future_np, agent_extent_future_np

    # @profile
    def get_nearby_agents(
        self,
        scene_time: SceneTimeAgent,
        agent_info: AgentMetadata,
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[List[AgentMetadata], np.ndarray]:
        """
        Returns Agent Metadata and Agent types of nearby agents
        """
        # The indices of the returned ndarray match the scene_time agents list
        # (including the index of the central agent, which would have a distance
        # of 0 to itself).
        agent_distances: np.ndarray = scene_time.get_agent_distances_to(agent_info)
        agent_idx: int = scene_time.agents.index(agent_info)

        neighbor_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])
        nearby_mask: np.ndarray = agent_distances <= distance_limit(
            neighbor_types, agent_info.type
        )
        nearby_mask[agent_idx] = False

        nb_idx = agent_distances.argsort()
        nearby_agents: List[AgentMetadata] = [
            scene_time.agents[idx] for idx in nb_idx if nearby_mask[idx]
        ]

        if self.max_neighbor_num is not None:
            # Pruning nearby_agents and re-creating
            # neighbor_types_np with the remaining agents.
            nearby_agents = nearby_agents[: self.max_neighbor_num]

        # Doing this here because the argsort above changes the order of agents.
        neighbor_types_np: np.ndarray = np.array([a.type.value for a in nearby_agents])

        return nearby_agents, neighbor_types_np

    def get_neighbor_history(
        self,
        history_sec: Tuple[Optional[float], Optional[float]],
        nearby_agents: List[AgentMetadata],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        (
            neighbor_data,
            neighbor_extents_data,
            neighbor_data_lens_np,
        ) = self.cache.get_agents_history(self.scene_ts, nearby_agents, history_sec)
        return (
            neighbor_data,
            neighbor_extents_data,
            neighbor_data_lens_np,
        )

    def get_neighbor_future(
        self,
        future_sec: Tuple[Optional[float], Optional[float]],
        nearby_agents: List[AgentMetadata],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        (
            neighbor_data,
            neighbor_extents_data,
            neighbor_data_lens_np,
        ) = self.cache.get_agents_future(self.scene_ts, nearby_agents, future_sec)
        return (
            neighbor_data,
            neighbor_extents_data,
            neighbor_data_lens_np,
        )

    def get_robot_current_and_future(
        self,
        robot_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> StateArray:
        robot_curr_np: StateArray = self.cache.get_state(robot_info.name, self.scene_ts)
        # robot_fut_extents_np,
        (
            robot_fut_np,
            _,
        ) = self.cache.get_agent_future(robot_info, self.scene_ts, future_sec)

        robot_curr_and_fut_np: StateArray = np.concatenate(
            (robot_curr_np[np.newaxis, :], robot_fut_np), axis=0
        ).view(self.cache.obs_type)
        return robot_curr_and_fut_np

    def get_agent_map_patch(self, patch_params: Dict[str, int]) -> RasterizedMapPatch:
        raise NotImplementedError()


class SceneBatchElement:
    """A single batch element."""

    def __init__(
        self,
        cache: SceneCache,
        data_index: int,
        scene_time: SceneTime,
        history_sec: Tuple[Optional[float], Optional[float]],
        future_sec: Tuple[Optional[float], Optional[float]],
        agent_interaction_distances: Dict[
            Tuple[AgentType, AgentType], float
        ] = defaultdict(lambda: np.inf),
        incl_robot_future: bool = False,
        incl_raster_map: bool = False,
        raster_map_params: Optional[Dict[str, Any]] = None,
        map_api: Optional[MapAPI] = None,
        vector_map_params: Optional[Dict[str, Any]] = None,
        state_format: Optional[str] = None,
        standardize_data: bool = False,
        standardize_derivatives: bool = False,
        max_agent_num: Optional[int] = None,
        lane_graph_cache: Optional[dict] = None,
    ) -> None:
        self.cache: SceneCache = cache
        self.data_index = data_index
        self.dt: float = scene_time.scene.dt
        self.scene_ts: int = scene_time.ts

        if max_agent_num is not None:
            scene_time.agents = scene_time.agents[:max_agent_num]

        self.agents: List[AgentMetadata] = scene_time.agents

        robot = [agent for agent in self.agents if agent.name == "ego"]
        if len(robot) > 0:
            self.centered_agent = robot[0]
        else:
            self.centered_agent = self.agents[0]

        raw_state: StateArray = cache.get_raw_state(
            self.centered_agent.name, self.scene_ts
        )

        if state_format is not None:
            self.centered_agent_state_np = raw_state.as_format(state_format)
        else:
            self.centered_agent_state_np = raw_state

        incl_z = self.centered_agent_state_np.has_attr("position3d")

        self.standardize_data = standardize_data

        if self.standardize_data:
            # Request cache to return observations relative to centered agent
            obs_frame: StateArray = convert_to_frame_state(
                raw_state,
                stationary=not standardize_derivatives,
                grounded=True,
            )
            cache.set_obs_frame(obs_frame)

            # Create 2d transformation matrix to and from agent and world
            agent_pos: np.ndarray = self.centered_agent_state_np.position
            agent_heading: float = self.centered_agent_state_np.heading[0]

            cos_agent, sin_agent = np.cos(agent_heading), np.sin(agent_heading)

            if incl_z:
                self.centered_world_from_agent_tf: np.ndarray = np.array(
                    [
                        [cos_agent, -sin_agent, 0.0, agent_pos[0]],
                        [sin_agent, cos_agent, 0.0, agent_pos[1]],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            else:
                self.centered_world_from_agent_tf: np.ndarray = np.array(
                    [
                        [cos_agent, -sin_agent, agent_pos[0]],
                        [sin_agent, cos_agent, agent_pos[1]],
                        [0.0, 0.0, 1.0],
                    ]
                )
            self.centered_agent_from_world_tf: np.ndarray = np.linalg.inv(
                self.centered_world_from_agent_tf
            )
        else:
            self.centered_agent_from_world_tf: np.ndarray = np.eye(4 if incl_z else 3)
            self.centered_world_from_agent_tf: np.ndarray = np.eye(4 if incl_z else 3)

        ### NEIGHBOR-SPECIFIC DATA ###
        def distance_limit(agent_types: np.ndarray, target_type: int) -> np.ndarray:
            return np.array(
                [
                    agent_interaction_distances[(agent_type, target_type)]
                    for agent_type in agent_types
                ]
            )

        nearby_agents, self.agent_types_np = self.get_nearby_agents(
            scene_time, self.centered_agent, distance_limit
        )
        self.agents = nearby_agents
        self.num_agents = len(nearby_agents)
        self.agent_names = [agent.name for agent in nearby_agents]
        (
            self.agent_histories,
            self.agent_history_extents,
            self.agent_history_lens_np,
        ) = self.get_agents_history(history_sec, nearby_agents)
        (
            self.agent_futures,
            self.agent_future_extents,
            self.agent_future_lens_np,
        ) = self.get_agents_future(future_sec, nearby_agents)

        self.agent_meta_dicts = [
            get_agent_meta_dict(self.cache, agent) for agent in nearby_agents
        ]

        ### MAP ###
        self.map_name: Optional[str] = None
        self.map_patches: Optional[RasterizedMapPatch] = None

        map_name: str = f"{scene_time.scene.env_name}:{scene_time.scene.location}"
        if incl_raster_map:
            self.map_name = map_name
            self.map_patches = self.get_agents_map_patch(
                raster_map_params, self.agent_histories
            )

        self.vec_map: Optional[VectorMap] = None
        if map_api is not None:
            self.vec_map = map_api.get_map(
                map_name,
                self.cache if self.cache.is_traffic_light_data_cached() else None,
                **vector_map_params if vector_map_params is not None else None,
            )
            if vector_map_params.get("calc_lane_graph", False):
                # not tested
                ego_xyh = np.concatenate(
                    [
                        self.centered_agent_state_np.position,
                        self.centered_agent_state_np.heading,
                    ]
                )
                num_pts = vector_map_params.get("num_lane_pts", 30)
                max_num_lanes = vector_map_params.get("max_num_lanes", 20)
                remove_single_successor = vector_map_params.get(
                    "remove_single_successor", False
                )
                (
                    self.num_lanes,
                    self.lane_xyh,
                    self.lane_adj,
                    self.lane_ids,
                    self.road_edge_xyzh,
                ) = gen_lane_graph(
                    self.vec_map,
                    ego_xyh,
                    self.centered_agent_from_world_tf,
                    num_pts,
                    max_num_lanes,
                    remove_single_successor=remove_single_successor,
                    get_road_edges=vector_map_params.get("incl_road_edges", False),
                    lane_graph_cache=lane_graph_cache,
                )

            else:
                self.lane_xyh = None
                self.lane_adj = None
                self.lane_ids = list()
                self.num_lanes = 0
                self.road_edge_xyzh = None

        self.scene_id = scene_time.scene.name

        ### ROBOT DATA ###
        self.robot_future_np: Optional[StateArray] = None

        if incl_robot_future:
            self.robot_future_np: StateArray = self.get_robot_current_and_future(
                self.centered_agent, future_sec
            )

            # -1 because this is meant to hold the number of future steps
            # (whereas the above returns the current + future, yielding
            # one more timestep).
            self.robot_future_len: int = self.robot_future_np.shape[0] - 1

        self.scene_id = scene_time.scene.name

        # Will be optionally populated by the user's provided functions.
        self.extras: Dict[str, np.ndarray] = dict()

    def get_nearby_agents(
        self,
        scene_time: SceneTime,
        agent: AgentMetadata,
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[List[AgentMetadata], np.ndarray]:
        agent_distances: np.ndarray = scene_time.get_agent_distances_to(agent)

        agents_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])
        nearby_mask: np.ndarray = agent_distances <= distance_limit(
            agents_types, agent.type
        )
        # sort the agents based on their distance to the centered agent
        idx = np.argsort(agent_distances)
        num_qualified = nearby_mask.sum()
        nearby_agents: List[AgentMetadata] = [
            scene_time.agents[idx[i]] for i in range(num_qualified)
        ]
        agents_types_np = agents_types[idx[:num_qualified]]
        return nearby_agents, agents_types_np

    def get_agents_history(
        self,
        history_sec: Tuple[Optional[float], Optional[float]],
        nearby_agents: List[AgentMetadata],
    ) -> Tuple[List[StateArray], List[np.ndarray], np.ndarray]:
        # The indices of the returned ndarray match the scene_time agents list (including the index of the central agent,
        # which would have a distance of 0 to itself).
        (
            agent_histories,
            agent_history_extents,
            agent_history_lens_np,
        ) = self.cache.get_agents_history(self.scene_ts, nearby_agents, history_sec)

        return (
            agent_histories,
            agent_history_extents,
            agent_history_lens_np,
        )

    def get_agents_future(
        self,
        future_sec: Tuple[Optional[float], Optional[float]],
        nearby_agents: List[AgentMetadata],
    ) -> Tuple[List[StateArray], List[np.ndarray], np.ndarray]:
        (
            agent_futures,
            agent_future_extents,
            agent_future_lens_np,
        ) = self.cache.get_agents_future(self.scene_ts, nearby_agents, future_sec)

        return (
            agent_futures,
            agent_future_extents,
            agent_future_lens_np,
        )

    def get_agents_map_patch(
        self, patch_params: Dict[str, int], agent_histories: List[np.ndarray]
    ) -> List[RasterizedMapPatch]:
        desired_patch_size: int = patch_params["map_size_px"]
        resolution: float = patch_params["px_per_m"]
        offset_xy: Tuple[float, float] = patch_params.get("offset_frac_xy", (0.0, 0.0))
        return_rgb: bool = patch_params.get("return_rgb", True)
        no_map_fill_val: float = patch_params.get("no_map_fill_value", 0.0)

        map_patches = list()

        curr_state = [state[-1] for state in agent_histories]
        curr_state = np.stack(curr_state).view(self.cache.obs_type)

        if self.standardize_data:
            # need to transform back into world frame
            obs_frame: StateArray = convert_to_frame_state(
                self.centered_agent_state_np, stationary=True, grounded=True
            )
            curr_state = transform_from_frame(curr_state, obs_frame)
            heading = curr_state.heading[:, 0]
        else:
            heading = 0.0 * curr_state.heading[:, 0]

        for i in range(curr_state.shape[0]):
            patch_data, raster_from_world_tf, has_data = self.cache.load_map_patch(
                curr_state.get_attr("x")[i],
                curr_state.get_attr("y")[i],
                desired_patch_size,
                resolution,
                offset_xy,
                heading[i],
                return_rgb,
                rot_pad_factor=sqrt(2),
                no_map_val=no_map_fill_val,
            )
            map_patches.append(
                RasterizedMapPatch(
                    data=patch_data,
                    rot_angle=heading[i],
                    crop_size=desired_patch_size,
                    resolution=resolution,
                    raster_from_world_tf=raster_from_world_tf,
                    has_data=has_data,
                )
            )

        return map_patches

    def get_robot_current_and_future(
        self,
        robot_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> StateArray:
        robot_curr_np: StateArray = self.cache.get_state(robot_info.name, self.scene_ts)
        # robot_fut_extents_np,
        (
            robot_fut_np,
            _,
        ) = self.cache.get_agent_future(robot_info, self.scene_ts, future_sec)

        robot_curr_and_fut_np: StateArray = np.concatenate(
            (robot_curr_np[np.newaxis, :], robot_fut_np), axis=0
        ).view(self.cache.obs_type)
        return robot_curr_and_fut_np


def gen_lane_graph(
    vec_map,
    ego_xyh,
    agent_from_world,
    num_pts=20,
    max_num_lanes=15,
    radius=150,
    get_road_edges=False,
    remove_single_successor=False,
    lane_graph_cache=None,
):
    close_lanes, dis = get_close_lanes(radius, ego_xyh, vec_map, num_pts)
    lanes_by_id = {lane.id: lane for lane in close_lanes}
    dis_by_id = {lane.id: dis[i] for i, lane in enumerate(close_lanes)}
    if lane_graph_cache is not None:
        idx = np.argsort(dis)[:max_num_lanes]
        lane_ids = [close_lanes[i].id for i in idx]
        num_lanes = len(lane_ids)
        cache_idx = [lane_graph_cache.lane_ids.index(id) for id in lane_ids]
        lane_xyh = lane_graph_cache.lane_centerlines[cache_idx]
        lane_adj = lane_graph_cache.lane_connectivity[cache_idx][:, cache_idx]
        lane_xyh = transform_xyh_np(
            lane_xyh.reshape(-1, 3), agent_from_world[None]
        ).reshape(num_lanes, -1, 3)
        road_edge_xyzh = None
        return num_lanes, lane_xyh, lane_adj, lane_ids, road_edge_xyzh

    if remove_single_successor:
        for lane in close_lanes:
            while len(lane.next_lanes) == 1:
                # if there are more than one succeeding lanes, then we abort the merging
                next_id = list(lane.next_lanes)[0]

                if next_id in lanes_by_id:
                    next_lane = lanes_by_id[next_id]
                    shared_next = False
                    for id in next_lane.prev_lanes:
                        if id != lane.id and id in lanes_by_id:
                            shared_next = True
                            break
                    if shared_next:
                        # if the next lane shares two prev lanes in the close_lanes, then we abort the merging
                        break
                    lane.combine_next(lanes_by_id[next_id])
                    dis_by_id[lane.id] = min(dis_by_id[lane.id], dis_by_id[next_id])
                    lanes_by_id.pop(next_id)
                else:
                    break
        close_lanes = list(lanes_by_id.values())
        dis = np.array([dis_by_id[lane.id] for lane in close_lanes])
    num_lanes = len(close_lanes)
    if num_lanes > max_num_lanes:
        idx = dis.argsort()[:max_num_lanes]
        close_lanes = [lane for i, lane in enumerate(close_lanes) if i in idx]
        num_lanes = max_num_lanes

    if num_lanes > 0:
        lane_xyh = list()
        lane_adj = np.zeros([len(close_lanes), len(close_lanes)], dtype=np.int32)
        lane_ids = [lane.id for lane in close_lanes]

        for i, lane in enumerate(close_lanes):
            center = lane.center.interpolate(num_pts).points[:, [0, 1, 3]]
            center_local = transform_xyh_np(
                # Add pts dimension and select x, y and homogeneous dimension
                center,
                agent_from_world[None][..., [0, 1, -1]][..., [0, 1, -1], :],
            )
            lane_xyh.append(center_local)
            # construct lane adjacency matrix
            for adj_lane_id in lane.next_lanes:
                if adj_lane_id in lane_ids:
                    lane_adj[i, lane_ids.index(adj_lane_id)] = (
                        LaneSegRelation.NEXT.value
                    )

            for adj_lane_id in lane.prev_lanes:
                if adj_lane_id in lane_ids:
                    lane_adj[i, lane_ids.index(adj_lane_id)] = (
                        LaneSegRelation.PREV.value
                    )

            for adj_lane_id in lane.adj_lanes_left:
                if adj_lane_id in lane_ids:
                    lane_adj[i, lane_ids.index(adj_lane_id)] = (
                        LaneSegRelation.LEFT.value
                    )

            for adj_lane_id in lane.adj_lanes_right:
                if adj_lane_id in lane_ids:
                    lane_adj[i, lane_ids.index(adj_lane_id)] = (
                        LaneSegRelation.RIGHT.value
                    )
        lane_xyh = np.stack(lane_xyh, axis=0)
        lane_xyh = lane_xyh
        lane_adj = lane_adj
    else:
        lane_xyh = np.zeros([0, num_pts, 3])
        lane_adj = np.zeros([0, 0])
        lane_ids = list()

    road_edge_xyzh = None
    if get_road_edges:
        close_road_edges, re_dis = get_close_road_edges(
            radius, ego_xyh, vec_map, num_pts
        )
        num_road_edges = len(close_road_edges)
        if num_road_edges > max_num_lanes:
            idx = re_dis.argsort()[:max_num_lanes]
            close_road_edges = [
                road_edge for i, road_edge in enumerate(close_road_edges) if i in idx
            ]
            num_road_edges = max_num_lanes

        if num_road_edges > 0:
            road_edge_xyzh = list()
            for i, road_edge in enumerate(close_road_edges):
                polyline = road_edge.polyline.interpolate(num_pts).points
                # TODO: What to do when `agent_from_world` doesn't have z coord?
                polyline_local = transform_xyh_np(polyline, agent_from_world[None])
                road_edge_xyzh.append(polyline_local)
            road_edge_xyzh = np.stack(road_edge_xyzh, axis=0)

    return num_lanes, lane_xyh, lane_adj, lane_ids, road_edge_xyzh


def is_agent_stationary(cache: SceneCache, agent_info: AgentMetadata) -> bool:
    # Agent is considered stationary if it moves less than 1m between the first and last valid timestep.
    first_state: StateArray = cache.get_state(
        agent_info.name, agent_info.first_timestep
    )
    last_state: StateArray = cache.get_state(agent_info.name, agent_info.last_timestep)
    is_stationary = np.square(last_state.position - first_state.position).sum(0) < 1.0
    return is_stationary


def get_agent_meta_dict(
    cache: SceneCache, agent_info: AgentMetadata
) -> Dict[str, np.ndarray]:
    return {
        "is_stationary": is_agent_stationary(cache, agent_info),
    }
