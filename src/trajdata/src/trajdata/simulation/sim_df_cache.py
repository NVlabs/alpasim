# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from trajdata.augmentation.augmentation import Augmentation
from trajdata.caching.df_cache import DataFrameCache
from trajdata.data_structures.agent import AgentMetadata
from trajdata.data_structures.scene_metadata import Scene
from trajdata.data_structures.state import StateArray
from trajdata.simulation.sim_cache import SimulationCache
from trajdata.simulation.sim_metrics import SimMetric
from trajdata.simulation.sim_stats import SimStatistic


class SimulationDataFrameCache(DataFrameCache, SimulationCache):
    def __init__(
        self,
        cache_path: Path,
        scene: Scene,
        scene_ts: int,
        augmentations: Optional[List[Augmentation]] = None,
    ) -> None:
        super().__init__(cache_path, scene, augmentations)
        self.scene_ts = scene_ts

        agent_names: List[str] = [agent.name for agent in scene.agents]
        in_index: np.ndarray = self.scene_data_df.index.isin(agent_names, level=0)
        self.scene_data_df: pd.DataFrame = self.scene_data_df.iloc[in_index].copy()
        self.index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.index)
        }

        # This will remain untouched through simulation, only present for
        # metrics computation later.
        self.original_scene_df: pd.DataFrame = self.scene_data_df.copy()

        # This will be modified as simulation steps forward.
        self.persistent_data_df: pd.DataFrame = self.scene_data_df.copy()

    def reset(self) -> None:
        self.scene_data_df = self.persistent_data_df.copy()
        self.index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.index)
        }
        self._get_and_reorder_col_idxs()

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if scene_ts >= agent_info.last_timestep:
            # Returning an empty DataFrame with the correct
            # columns. 3 = Extent size.
            return np.zeros((0, self.obs_dim)), np.zeros((0, 3))

        return super().get_agent_future(agent_info, scene_ts, future_sec)

    def get_agents_future(
        self,
        scene_ts: int,
        agents: List[AgentMetadata],
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        last_timesteps = np.array([agent.last_timestep for agent in agents], dtype=int)

        if np.all(np.greater(scene_ts, last_timesteps)):
            return (
                [np.zeros((0, self.obs_dim)) for agent in agents],
                [np.zeros((0, 3)) for agent in agents],  # 3 = Extent size.
                np.zeros_like(last_timesteps),
            )

        return super().get_agents_future(scene_ts, agents, future_sec)

    def append_state(self, xyzh_dict: Dict[str, StateArray]) -> None:
        self.scene_ts += 1

        sim_dict: Dict[str, List[Union[str, float, int]]] = defaultdict(list)
        prev_states: StateArray = self.get_states(
            list(xyzh_dict.keys()), self.scene_ts - 1
        )

        new_xyzh: StateArray
        for idx, (agent, new_xyzh) in enumerate(xyzh_dict.items()):
            prev_state: StateArray = prev_states[idx]

            sim_dict["agent_id"].append(agent)
            sim_dict["scene_ts"].append(self.scene_ts)

            old_x, old_y = prev_state.position
            new_x, new_y = new_xyzh.position

            sim_dict["x"].append(new_x)
            sim_dict["y"].append(new_y)

            vx: float = (new_x - old_x) / self.scene.dt
            vy: float = (new_y - old_y) / self.scene.dt
            sim_dict["vx"].append(vx)
            sim_dict["vy"].append(vy)

            old_vx, old_vy = prev_state.velocity

            ax: float = (vx - old_vx) / self.scene.dt
            ay: float = (vy - old_vy) / self.scene.dt
            sim_dict["ax"].append(ax)
            sim_dict["ay"].append(ay)

            sim_dict["heading"].append(new_xyzh.heading.item())

            if self.extent_cols:
                sim_dict["length"].append(
                    self.get_value(agent, self.scene_ts - 1, "length")
                )
                sim_dict["width"].append(
                    self.get_value(agent, self.scene_ts - 1, "width")
                )
                sim_dict["height"].append(
                    self.get_value(agent, self.scene_ts - 1, "height")
                )

        sim_step_df = pd.DataFrame(sim_dict)
        sim_step_df.set_index(["agent_id", "scene_ts"], inplace=True)
        if (
            self.scene_ts < self.scene.length_timesteps
            and self.scene_ts in self.persistent_data_df.index.get_level_values(1)
        ):
            self.persistent_data_df.drop(index=self.scene_ts, level=1, inplace=True)

        self.persistent_data_df = pd.concat([self.persistent_data_df, sim_step_df])
        self.persistent_data_df.sort_index(inplace=True)
        self.reset()

    def add_agents(self, agent_data: List[Tuple]):
        """Add new agents to the simulation data.

        Args:
            agent_data (List[Tuple]): _description_
        """
        new_state_df = list()
        for data_i in agent_data:
            name, state, ts0, _, extent = data_i

            T = state.shape[0]
            if T == 0:
                vel = np.zeros([1, 2])
                acc = np.zeros([1, 2])
            else:
                vel = (state[1:, :2] - state[:-1, :2]) / self.scene.dt
                vel = np.vstack((vel[0:1], vel))

                acc = (vel[1:] - vel[:-1]) / self.scene.dt
                acc = np.vstack((acc[0:1], acc))

            data = dict(
                agent_id=np.array([name] * T),
                scene_ts=np.arange(ts0, ts0 + T),
                x=state[:, 0],
                y=state[:, 1],
                vx=vel[:, 0],
                vy=vel[:, 1],
                ax=acc[:, 0],
                ay=acc[:, 1],
                heading=state[:, 2],
            )

            if self.extent_cols:
                data["length"] = extent[0]
                data["width"] = extent[1]
                data["height"] = extent[2]

            new_state_df_i = pd.DataFrame(data)
            new_state_df.append(new_state_df_i)

        new_state_df = pd.concat(new_state_df)
        new_state_df.set_index(["agent_id", "scene_ts"], inplace=True)

        self.persistent_data_df = pd.concat([self.persistent_data_df, new_state_df])
        self.persistent_data_df.sort_index(inplace=True)
        self.reset()

    def save_sim_scene(self, sim_scene: Scene) -> None:
        history_idxs = (
            self.persistent_data_df.index.get_level_values("scene_ts") <= self.scene_ts
        )
        DataFrameCache.save_agent_data(
            self.persistent_data_df[history_idxs], self.path, sim_scene
        )

    def calculate_metrics(
        self, metrics: List[SimMetric], ts_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics about the simulated scene.

        Args:
            metrics (List[SimMetric]): The metrics to compute.
            ts_range (Optional[Tuple[int, int]], optional): Optional specification of which timesteps to constrain metric computation within (both inclusive). Defaults to None which means all available timesteps.

        Returns:
            Dict[str, Dict[str, float]]: A mapping from metric names to a dict of present agent names and their associated metric value.
        """
        index_scene_ts: pd.Index = self.original_scene_df.index.get_level_values(1)
        if ts_range is not None:
            from_ts, to_ts = ts_range
        else:
            from_ts, to_ts = 0, index_scene_ts.max()

        ts_range_mask: np.ndarray = (index_scene_ts >= from_ts) & (
            index_scene_ts <= to_ts
        )
        intersected_index: pd.Index = self.original_scene_df.index[
            ts_range_mask
        ].intersection(self.scene_data_df.index, sort=None)
        gt_df: pd.DataFrame = self.original_scene_df.reindex(index=intersected_index)
        sim_df: pd.DataFrame = self.scene_data_df.reindex(index=intersected_index)

        metrics_dict: Dict[str, Dict[str, float]] = dict()
        for metric in metrics:
            metrics_dict[metric.name] = metric(gt_df, sim_df)

        return metrics_dict

    def calculate_stats(
        self, metrics: List[SimStatistic], ts_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Calculate statistics about the simulated scene.

        Args:
            stats (List[SimStatistic]): The statistics to compute.
            ts_range (Optional[Tuple[int, int]], optional): Optional specification of which timesteps to constrain metric computation within (both inclusive). Defaults to None which means all available timesteps.

        Returns:
            Dict[str, np.ndarray]: A mapping from present agent names to their associated statistic.
        """
        og_index_scene_ts: pd.Index = self.original_scene_df.index.get_level_values(1)
        sim_index_scene_ts: pd.Index = self.scene_data_df.index.get_level_values(1)
        if ts_range is not None:
            from_ts, to_ts = ts_range
        else:
            from_ts, to_ts = 0, max(og_index_scene_ts.max(), sim_index_scene_ts.max())

        gt_ts_range_mask: np.ndarray = (og_index_scene_ts >= from_ts) & (
            og_index_scene_ts <= to_ts
        )
        gt_df: pd.DataFrame = self.original_scene_df.iloc[gt_ts_range_mask]

        sim_ts_range_mask: np.ndarray = (sim_index_scene_ts >= from_ts) & (
            sim_index_scene_ts <= to_ts
        )
        sim_df: pd.DataFrame = self.scene_data_df.iloc[sim_ts_range_mask]

        stats_dict: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = dict()
        for metric in metrics:
            stats_dict[metric.name] = {"gt": metric(gt_df), "sim": metric(sim_df)}

        return stats_dict
