# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import dill
import numpy as np
import torch
from torch.utils.data import Sampler

from trajdata import UnifiedDataset
from trajdata.caching.df_cache import DataFrameCache
from trajdata.data_structures import (
    AgentBatch,
    AgentBatchElement,
    AgentDataIndex,
    AgentType,
    SceneBatch,
    SceneBatchElement,
    SceneTimeAgent,
)
from trajdata.data_structures.collation import (
    agent_collate_fn,
    batch_rotate_raster_maps_for_agents_in_scene,
)
from trajdata.maps import RasterizedMapMetadata, RasterizedMapPatch, VectorMap
from trajdata.utils.arr_utils import (
    PadDirection,
    batch_nd_transform_xyvvaahh_pt,
    batch_select,
)
from trajdata.utils.map_utils import load_map_patch


class SceneTimeBatcher(Sampler):
    _agent_data_index: AgentDataIndex
    _agent_idx: int

    def __init__(
        self, agent_centric_dataset: UnifiedDataset, agent_idx_to_follow: int = 0
    ) -> None:
        """
        Returns a sampler (to be used in a torch.utils.data.DataLoader)
        which works with an agent-centric UnifiedDataset, yielding
        batches consisting of whole scenes (AgentBatchElements for all agents
        in a particular scene at a particular time)

        Args:
            agent_centric_dataset (UnifiedDataset)
            agent_idx_to_follow (int): index of agent to return batches for. Defaults to 0,
                meaning we include all scene frames where the ego agent appears, which
                usually covers the entire dataset.
        """
        super().__init__(agent_centric_dataset)
        self._agent_data_index = agent_centric_dataset._data_index
        self._agent_idx = agent_idx_to_follow
        self._cumulative_lengths = np.concatenate(
            [
                [0],
                np.cumsum(
                    [
                        cumulative_scene_length[self._agent_idx + 1]
                        - cumulative_scene_length[self._agent_idx]
                        for cumulative_scene_length in self._agent_data_index._cumulative_scene_lengths
                    ]
                ),
            ]
        )

    def __len__(self):
        return self._cumulative_lengths[-1]

    def __iter__(self) -> Iterator[int]:
        for idx in range(len(self)):
            # TODO(apoorvas) May not need to do this search, since we only support an iterable style access?
            scene_idx: int = (
                np.searchsorted(self._cumulative_lengths, idx, side="right").item() - 1
            )

            # offset into dataset index to reach current scene
            scene_offset = self._agent_data_index._cumulative_lengths[scene_idx].item()

            # how far along we are in the current scene
            scene_elem_index = idx - self._cumulative_lengths[scene_idx].item()

            # convert to scene-timestep for the tracked agent
            scene_ts = (
                scene_elem_index
                + self._agent_data_index._agent_times[scene_idx][self._agent_idx, 0]
            )

            # build a set of indices into the agent-centric dataset for all agents that exist at this scene and timestep
            indices = []
            for agent_idx, agent_times in enumerate(
                self._agent_data_index._agent_times[scene_idx]
            ):
                if scene_ts > agent_times[1]:
                    # we are past the last timestep for this agent (times are inclusive)
                    continue
                agent_offset = scene_ts - agent_times[0]
                if agent_offset < 0:
                    # this agent hasn't entered the scene yet
                    continue

                # compute index into original dataset, first into scene, then into this agent's part in scene, and then the offset
                index_to_add = (
                    scene_offset
                    + self._agent_data_index._cumulative_scene_lengths[scene_idx][
                        agent_idx
                    ]
                    + agent_offset
                )
                indices.append(index_to_add)

            yield indices


def convert_to_agent_batch(
    scene_batch_element: SceneBatchElement,
    only_types: Optional[List[AgentType]] = None,
    no_types: Optional[List[AgentType]] = None,
    agent_interaction_distances: Dict[Tuple[AgentType, AgentType], float] = defaultdict(
        lambda: np.inf
    ),
    incl_map: bool = False,
    map_params: Optional[Dict[str, Any]] = None,
    max_neighbor_num: Optional[int] = None,
    state_format: Optional[str] = None,
    standardize_data: bool = True,
    standardize_derivatives: bool = False,
    pad_format: str = "outside",
) -> AgentBatch:
    """
    Converts a SceneBatchElement into a AgentBatch consisting of
    AgentBatchElements for all agents present at the given scene at the given
    time step.

    Args:
        scene_batch_element (SceneBatchElement): element to process
        only_types (Optional[List[AgentType]], optional): AgentsTypes to consider. Defaults to None.
        no_types (Optional[List[AgentType]], optional): AgentTypes to ignore. Defaults to None.
        agent_interaction_distances (_type_, optional): Distance threshold for interaction. Defaults to defaultdict(lambda: np.inf).
        incl_map (bool, optional): Whether to include map info. Defaults to False.
        map_params (Optional[Dict[str, Any]], optional): Map params. Defaults to None.
        max_neighbor_num (Optional[int], optional): Max number of neighbors to allow. Defaults to None.
        standardize_data (bool): Whether to return data relative to current agent state. Defaults to True.
        standardize_derivatives: Whether to transform relative velocities and accelerations as well. Defaults to False.
        pad_format (str, optional): Pad format when collating agent trajectories. Defaults to "outside".

    Returns:
        AgentBatch: batch of AgentBatchElements corresponding to all agents in the SceneBatchElement
    """
    data_idx = scene_batch_element.data_index
    cache = scene_batch_element.cache
    scene = cache.scene
    dt = scene_batch_element.dt
    ts = scene_batch_element.scene_ts
    state_format = scene_batch_element.centered_agent_state_np._format

    batch_elems: List[AgentBatchElement] = []
    for j, agent_name in enumerate(scene_batch_element.agent_names):
        history_sec = dt * (scene_batch_element.agent_histories[j].shape[0] - 1)
        future_sec = dt * (scene_batch_element.agent_futures[j].shape[0])
        cache.reset_obs_frame()
        scene_time_agent: SceneTimeAgent = SceneTimeAgent.from_cache(
            scene,
            ts,
            agent_name,
            cache,
            only_types=only_types,
            no_types=no_types,
        )

        batch_elems.append(
            AgentBatchElement(
                cache=cache,
                data_index=data_idx,
                scene_time_agent=scene_time_agent,
                history_sec=(history_sec, history_sec),
                future_sec=(future_sec, future_sec),
                agent_interaction_distances=agent_interaction_distances,
                incl_raster_map=incl_map,
                raster_map_params=map_params,
                state_format=state_format,
                standardize_data=standardize_data,
                standardize_derivatives=standardize_derivatives,
                max_neighbor_num=max_neighbor_num,
            )
        )

    return agent_collate_fn(batch_elems, return_dict=False, pad_format=pad_format)


def get_agents_map_patch(
    cache_path: Path,
    map_name: str,
    patch_params: Dict[str, int],
    agent_world_states_xyh: Union[np.ndarray, torch.Tensor],
    allow_nan: float = False,
    vector_map: Optional[VectorMap] = None,
) -> List[RasterizedMapPatch]:

    if isinstance(agent_world_states_xyh, torch.Tensor):
        agent_world_states_xyh = agent_world_states_xyh.cpu().numpy()
    assert agent_world_states_xyh.ndim == 2
    assert agent_world_states_xyh.shape[-1] == 3
    desired_patch_size: int = patch_params["map_size_px"]
    resolution: float = patch_params["px_per_m"]
    offset_xy: Tuple[float, float] = patch_params.get("offset_frac_xy", (0.0, 0.0))
    return_rgb: bool = patch_params.get("return_rgb", True)
    no_map_fill_val: float = patch_params.get("no_map_fill_value", 0.0)

    if (
        vector_map is not None
        and getattr(vector_map, "raster_map_data", None) is not None
    ):
        # Use preloaded raster map data
        raster_map_or_path = vector_map.raster_map_data
        raster_metadata_or_path = vector_map.raster_map_metadata
    else:
        env_name, location_name = map_name.split(
            ":"
        )  # assumes map_name format nusc_mini:boston-seaports
        (
            maps_path,
            _,
            _,
            raster_map_or_path,
            raster_metadata_or_path,
        ) = DataFrameCache.get_map_paths(
            cache_path, env_name, location_name, resolution
        )

    map_patches = list()
    for i in range(agent_world_states_xyh.shape[0]):
        patch_data, raster_from_world_tf, has_data = load_map_patch(
            raster_map_or_path,
            raster_metadata_or_path,
            agent_world_states_xyh[i, 0],
            agent_world_states_xyh[i, 1],
            desired_patch_size,
            resolution,
            offset_xy,
            agent_world_states_xyh[i, 2],
            return_rgb,
            rot_pad_factor=np.sqrt(2),
            no_map_val=no_map_fill_val,
        )
        map_patches.append(
            RasterizedMapPatch(
                data=patch_data,
                rot_angle=agent_world_states_xyh[i, 2],
                crop_size=desired_patch_size,
                resolution=resolution,
                raster_from_world_tf=raster_from_world_tf,
                has_data=has_data,
            )
        )

    return map_patches


def load_raster_map_data(cache_path: Path, map_name: str, patch_params: Dict[str, int]):
    raise NotImplementedError()


def get_raster_maps_for_scene_batch(
    batch: SceneBatch, cache_path: Path, raster_map_params: Dict, ego_only: bool = False
):
    raise NotImplementedError()
