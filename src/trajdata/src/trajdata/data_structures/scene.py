# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import List, Optional, Set

import numpy as np
import pandas as pd

from trajdata import filtering
from trajdata.caching import SceneCache
from trajdata.data_structures.agent import Agent, AgentMetadata, AgentType
from trajdata.data_structures.scene_metadata import Scene
from trajdata.data_structures.state import StateArray


class SceneTime:
    """Holds the data for a particular scene at a particular timestep."""

    def __init__(
        self,
        scene: Scene,
        scene_ts: int,
        agents: List[AgentMetadata],
        cache: SceneCache,
    ) -> None:
        self.scene = scene
        self.ts = scene_ts
        self.agents = agents
        self.cache = cache

    @classmethod
    def from_cache(
        cls,
        scene: Scene,
        scene_ts: int,
        cache: SceneCache,
        only_types: Optional[Set[AgentType]] = None,
        no_types: Optional[Set[AgentType]] = None,
    ):
        agents_present: List[AgentMetadata] = scene.agent_presence[scene_ts]
        filtered_agents: List[AgentMetadata] = filtering.agent_types(
            agents_present, no_types, only_types
        )

        return cls(scene, scene_ts, filtered_agents, cache)

    def get_agent_distances_to(self, agent: Agent) -> np.ndarray:
        agent_pos: StateArray = self.cache.get_state(agent.name, self.ts).position
        nb_pos: np.ndarray = np.stack(
            [self.cache.get_state(nb.name, self.ts).position for nb in self.agents]
        )

        return np.linalg.norm(nb_pos - agent_pos, axis=1)


class SceneTimeAgent:
    """Holds the data for a particular agent in a scene at a particular timestep."""

    def __init__(
        self,
        scene: Scene,
        scene_ts: int,
        agents: List[AgentMetadata],
        agent: AgentMetadata,
        cache: SceneCache,
        robot: Optional[AgentMetadata] = None,
    ) -> None:
        self.scene = scene
        self.ts = scene_ts
        self.agents = agents
        self.agent = agent
        self.cache = cache
        self.robot = robot

    @classmethod
    def from_cache(
        cls,
        scene: Scene,
        scene_ts: int,
        agent_id: str,
        cache: SceneCache,
        only_types: Optional[Set[AgentType]] = None,
        no_types: Optional[Set[AgentType]] = None,
        incl_robot_future: bool = False,
    ):
        agents_present: List[AgentMetadata] = scene.agent_presence[scene_ts]
        filtered_agents: List[AgentMetadata] = filtering.agent_types(
            agents_present, no_types, only_types
        )

        agent_metadata = next((a for a in filtered_agents if a.name == agent_id), None)

        if incl_robot_future:
            ego_metadata = next((a for a in filtered_agents if a.name == "ego"), None)

            return cls(
                scene,
                scene_ts,
                agents=filtered_agents,
                agent=agent_metadata,
                cache=cache,
                robot=ego_metadata,
            )
        else:
            return cls(
                scene,
                scene_ts,
                agents=filtered_agents,
                agent=agent_metadata,
                cache=cache,
            )

    # @profile
    def get_agent_distances_to(self, agent_info: AgentMetadata) -> np.ndarray:
        agent_pos: StateArray = self.cache.get_state(agent_info.name, self.ts).position

        curr_poses: StateArray = self.cache.get_states(
            [a.name for a in self.agents], self.ts
        ).position
        return np.linalg.norm(curr_poses - agent_pos, axis=1)
