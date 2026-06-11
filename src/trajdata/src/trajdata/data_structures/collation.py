# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from trajdata.augmentation import BatchAugmentation
from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.data_structures.state import TORCH_STATE_TYPES
from trajdata.maps import VectorMap
from trajdata.utils import arr_utils


class CustomCollateData:
    @staticmethod
    def __collate__(elements: list) -> any:
        raise NotImplementedError

    def __to__(self, device, non_blocking=False):
        # Example for moving all elements of a list to a device:
        # return LanesList([[pts.to(device, non_blocking=non_blocking)
        #           for pts in lanelist] for lanelist in self])
        raise NotImplementedError


def _collate_data(elems):
    if hasattr(elems[0], "__collate__"):
        return elems[0].__collate__(elems)
    else:
        return torch.as_tensor(np.stack(elems))


def _collate_lane_graph(elems):
    num_lanes = [elem.num_lanes for elem in elems]
    bs = len(elems)
    M = max(num_lanes)
    lane_xyh = np.zeros([bs, M, *elems[0].lane_xyh.shape[-2:]])
    lane_adj = np.zeros([bs, M, M], dtype=int)
    lane_ids = list()
    lane_mask = np.zeros([bs, M], dtype=int)
    for i, elem in enumerate(elems):
        lane_xyh[i, : num_lanes[i]] = elem.lane_xyh
        lane_adj[i, : num_lanes[i], : num_lanes[i]] = elem.lane_adj
        lane_ids.append(elem.lane_ids)
        lane_mask[i, : num_lanes[i]] = 1

    if elems[0].road_edge_xyzh is not None:
        assert (
            elems[0].road_edge_xyzh.shape[-1] == 4
        ), "Road edge data must have 4 dimensions: x, y, z, heading. "
        num_road_edges = [elem.road_edge_xyzh.shape[0] for elem in elems]
        N = max(num_road_edges)
        road_edge_xyzh = np.zeros([bs, N, *elems[0].road_edge_xyzh.shape[-2:]])
        for i, elem in enumerate(elems):
            road_edge_xyzh[i, : num_road_edges[i]] = elem.road_edge_xyzh
    else:
        road_edge_xyzh = None

    return (
        torch.as_tensor(lane_xyh),
        torch.as_tensor(lane_adj),
        torch.as_tensor(lane_mask),
        lane_ids,
        torch.as_tensor(road_edge_xyzh) if road_edge_xyzh is not None else None,
    )


def raster_map_collate_fn_agent(
    batch_elems: List[AgentBatchElement],
):
    raise NotImplementedError()


def raster_map_collate_fn_scene(
    batch_elems: List[SceneBatchElement],
    max_agent_num: Optional[int] = None,
    pad_value: Any = np.nan,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:

    raise NotImplementedError()


def agent_collate_fn(
    batch_elems: List[AgentBatchElement],
    return_dict: bool,
    pad_format: str,
    batch_augments: Optional[List[BatchAugmentation]] = None,
) -> Union[AgentBatch, Dict[str, Any]]:
    batch_size: int = len(batch_elems)
    history_pad_dir: arr_utils.PadDirection = (
        arr_utils.PadDirection.BEFORE
        if pad_format == "outside"
        else arr_utils.PadDirection.AFTER
    )

    data_index_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    scene_ts_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    dt_t: Tensor = torch.zeros((batch_size,), dtype=torch.float)
    agent_type_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    agent_names: List[str] = list()

    # get agent state and obs format from first item in list
    state_format = batch_elems[0].curr_agent_state_np._format
    obs_format = batch_elems[0].cache.obs_type._format
    AgentStateTensor = TORCH_STATE_TYPES[state_format]
    AgentObsTensor = TORCH_STATE_TYPES[obs_format]

    curr_agent_state: List[AgentStateTensor] = list()

    agent_history: List[AgentObsTensor] = list()
    agent_history_extent: List[Tensor] = list()
    agent_history_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    agent_future: List[AgentObsTensor] = list()
    agent_future_extent: List[Tensor] = list()
    agent_future_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    num_neighbors_t: Tensor = torch.as_tensor(
        [elem.num_neighbors for elem in batch_elems], dtype=torch.long
    )
    max_num_neighbors: int = num_neighbors_t.max().item()

    neighbor_types: List[Tensor] = list()
    neighbor_histories: List[AgentObsTensor] = list()
    neighbor_history_extents: List[Tensor] = list()
    neighbor_futures: List[AgentObsTensor] = list()
    neighbor_future_extents: List[Tensor] = list()

    # Doing this one up here so that I can use it later in the loop.
    if max_num_neighbors > 0:
        neighbor_history_lens_t: Tensor = pad_sequence(
            [
                torch.as_tensor(elem.neighbor_history_lens_np, dtype=torch.long)
                for elem in batch_elems
            ],
            batch_first=True,
            padding_value=0,
        )
        max_neigh_history_len: int = neighbor_history_lens_t.max().item()

        neighbor_future_lens_t: Tensor = pad_sequence(
            [
                torch.as_tensor(elem.neighbor_future_lens_np, dtype=torch.long)
                for elem in batch_elems
            ],
            batch_first=True,
            padding_value=0,
        )
        max_neigh_future_len: int = neighbor_future_lens_t.max().item()
    else:
        neighbor_history_lens_t: Tensor = torch.full((batch_size, 0), np.nan)
        max_neigh_history_len: int = 0

        neighbor_future_lens_t: Tensor = torch.full((batch_size, 0), np.nan)
        max_neigh_future_len: int = 0

    robot_future: List[AgentObsTensor] = list()
    robot_future_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    elem: AgentBatchElement
    for idx, elem in enumerate(batch_elems):
        data_index_t[idx] = elem.data_index
        scene_ts_t[idx] = elem.scene_ts
        dt_t[idx] = elem.dt
        agent_names.append(elem.agent_name)
        agent_type_t[idx] = elem.agent_type.value

        curr_agent_state.append(
            torch.as_tensor(elem.curr_agent_state_np, dtype=torch.float)
        )

        agent_history.append(
            arr_utils.convert_with_dir(
                elem.agent_history_np,
                dtype=torch.float,
                time_dim=-2,
                pad_dir=history_pad_dir,
            )
        )
        agent_history_extent.append(
            arr_utils.convert_with_dir(
                elem.agent_history_extent_np,
                dtype=torch.float,
                time_dim=-2,
                pad_dir=history_pad_dir,
            )
        )
        agent_history_len[idx] = elem.agent_history_len

        agent_future.append(torch.as_tensor(elem.agent_future_np, dtype=torch.float))
        agent_future_extent.append(
            torch.as_tensor(elem.agent_future_extent_np, dtype=torch.float)
        )
        agent_future_len[idx] = elem.agent_future_len

        neighbor_types.append(
            torch.as_tensor(elem.neighbor_types_np, dtype=torch.float)
        )

        if elem.num_neighbors > 0:
            # History
            padded_neighbor_histories = arr_utils.pad_sequences(
                elem.neighbor_histories,
                dtype=torch.float,
                time_dim=-2,
                pad_dir=history_pad_dir,
                batch_first=True,
                padding_value=np.nan,
            )
            padded_neighbor_history_extents = arr_utils.pad_sequences(
                elem.neighbor_history_extents,
                dtype=torch.float,
                time_dim=-2,
                pad_dir=history_pad_dir,
                batch_first=True,
                padding_value=np.nan,
            )
            if padded_neighbor_histories.shape[-2] < max_neigh_history_len:
                to_add = max_neigh_history_len - padded_neighbor_histories.shape[-2]
                padded_neighbor_histories = F.pad(
                    padded_neighbor_histories,
                    pad=(
                        (0, 0, to_add, 0)
                        if history_pad_dir == arr_utils.PadDirection.BEFORE
                        else (0, 0, 0, to_add)
                    ),
                    mode="constant",
                    value=np.nan,
                )
                padded_neighbor_history_extents = F.pad(
                    padded_neighbor_history_extents,
                    pad=(
                        (0, 0, to_add, 0)
                        if history_pad_dir == arr_utils.PadDirection.BEFORE
                        else (0, 0, 0, to_add)
                    ),
                    mode="constant",
                    value=np.nan,
                )

            neighbor_histories.append(
                padded_neighbor_histories.reshape(
                    (-1, padded_neighbor_histories.shape[-1])
                )
            )
            neighbor_history_extents.append(
                padded_neighbor_history_extents.reshape(
                    (-1, padded_neighbor_history_extents.shape[-1])
                )
            )

            # Future
            padded_neighbor_futures = pad_sequence(
                [
                    torch.as_tensor(nh, dtype=torch.float)
                    for nh in elem.neighbor_futures
                ],
                batch_first=True,
                padding_value=np.nan,
            )
            padded_neighbor_future_extents = pad_sequence(
                [
                    torch.as_tensor(nh, dtype=torch.float)
                    for nh in elem.neighbor_future_extents
                ],
                batch_first=True,
                padding_value=np.nan,
            )
            if padded_neighbor_futures.shape[-2] < max_neigh_future_len:
                to_add = max_neigh_future_len - padded_neighbor_futures.shape[-2]
                padded_neighbor_futures = F.pad(
                    padded_neighbor_futures,
                    pad=(0, 0, 0, to_add),
                    mode="constant",
                    value=np.nan,
                )
                padded_neighbor_future_extents = F.pad(
                    padded_neighbor_future_extents,
                    pad=(0, 0, 0, to_add),
                    mode="constant",
                    value=np.nan,
                )

            neighbor_futures.append(
                padded_neighbor_futures.reshape((-1, padded_neighbor_futures.shape[-1]))
            )
            neighbor_future_extents.append(
                padded_neighbor_future_extents.reshape(
                    (-1, padded_neighbor_future_extents.shape[-1])
                )
            )
        else:
            # If there's no neighbors, make the state dimension match the
            # agent history state dimension (presumably they'll be the same
            # since they're obtained from the same cached data source).
            neighbor_histories.append(
                torch.full(
                    (0, elem.agent_history_np.shape[-1]), np.nan, dtype=torch.float
                )
            )
            neighbor_history_extents.append(
                torch.full((0, elem.agent_history_extent_np.shape[-1]), np.nan)
            )

            neighbor_futures.append(
                torch.full(
                    (0, elem.agent_future_np.shape[-1]), np.nan, dtype=torch.float
                )
            )
            neighbor_future_extents.append(
                torch.full((0, elem.agent_future_extent_np.shape[-1]), np.nan)
            )

        if elem.robot_future_np is not None:
            robot_future.append(
                torch.as_tensor(elem.robot_future_np, dtype=torch.float)
            )
            robot_future_len[idx] = elem.robot_future_len

    curr_agent_state_t: AgentStateTensor = torch.stack(curr_agent_state).as_subclass(
        AgentStateTensor
    )

    agent_history_t: AgentObsTensor = arr_utils.pad_with_dir(
        agent_history,
        time_dim=-2,
        pad_dir=history_pad_dir,
        batch_first=True,
        padding_value=np.nan,
    ).as_subclass(AgentObsTensor)
    agent_history_extent_t: Tensor = arr_utils.pad_with_dir(
        agent_history_extent,
        time_dim=-2,
        pad_dir=history_pad_dir,
        batch_first=True,
        padding_value=np.nan,
    )

    agent_future_t: AgentObsTensor = pad_sequence(
        agent_future, batch_first=True, padding_value=np.nan
    ).as_subclass(AgentObsTensor)
    agent_future_extent_t: Tensor = pad_sequence(
        agent_future_extent, batch_first=True, padding_value=np.nan
    )

    # Padding history/future in case the length is less than
    # the minimum desired history/future length.
    if elem.history_sec[0] is not None:
        hist_len = int(elem.history_sec[0] / elem.dt) + 1
        if agent_history_t.shape[-2] < hist_len:
            to_add: int = hist_len - agent_history_t.shape[-2]
            agent_history_t = F.pad(
                agent_history_t,
                (
                    (0, 0, to_add, 0)
                    if history_pad_dir == arr_utils.PadDirection.BEFORE
                    else (0, 0, 0, to_add)
                ),
                value=np.nan,
            ).as_subclass(AgentObsTensor)

        if agent_history_extent_t.shape[-2] < hist_len:
            to_add: int = hist_len - agent_history_extent_t.shape[-2]
            agent_history_extent_t = F.pad(
                agent_history_extent_t,
                (
                    (0, 0, to_add, 0)
                    if history_pad_dir == arr_utils.PadDirection.BEFORE
                    else (0, 0, 0, to_add)
                ),
                value=np.nan,
            )

    if elem.future_sec[0] is not None:
        fut_len = int(elem.future_sec[0] / elem.dt)
        if agent_future_t.shape[-2] < fut_len:
            agent_future_t = F.pad(
                agent_future_t,
                (0, 0, 0, fut_len - agent_future_t.shape[-2]),
                value=np.nan,
            ).as_subclass(AgentObsTensor)

        if agent_future_extent_t.shape[-2] < fut_len:
            agent_future_extent_t = F.pad(
                agent_future_extent_t,
                (0, 0, 0, fut_len - agent_future_extent_t.shape[-2]),
                value=np.nan,
            )

    if max_num_neighbors > 0:
        # This is padding over number of neighbors, so no need
        # to do any padding direction malarkey.
        neighbor_types_t: Tensor = pad_sequence(
            neighbor_types, batch_first=True, padding_value=-1
        )

        neighbor_histories_t: AgentObsTensor = (
            pad_sequence(neighbor_histories, batch_first=True, padding_value=np.nan)
            .reshape(
                (
                    batch_size,
                    max_num_neighbors,
                    max_neigh_history_len,
                    agent_history_t.shape[-1],
                )
            )
            .as_subclass(AgentObsTensor)
        )
        neighbor_history_extents_t: Tensor = pad_sequence(
            neighbor_history_extents, batch_first=True, padding_value=np.nan
        ).reshape(
            (
                batch_size,
                max_num_neighbors,
                max_neigh_history_len,
                agent_history_extent_t.shape[-1],
            )
        )

        neighbor_futures_t: AgentObsTensor = (
            pad_sequence(neighbor_futures, batch_first=True, padding_value=np.nan)
            .reshape(
                (
                    batch_size,
                    max_num_neighbors,
                    max_neigh_future_len,
                    agent_future_t.shape[-1],
                )
            )
            .as_subclass(AgentObsTensor)
        )
        neighbor_future_extents_t: Tensor = pad_sequence(
            neighbor_future_extents, batch_first=True, padding_value=np.nan
        ).reshape(
            (
                batch_size,
                max_num_neighbors,
                max_neigh_future_len,
                agent_future_extent_t.shape[-1],
            )
        )
    else:
        neighbor_types_t: Tensor = torch.full((batch_size, 0), np.nan)

        neighbor_histories_t: AgentObsTensor = torch.full(
            (batch_size, 0, max_neigh_history_len, agent_history_t.shape[-1]),
            np.nan,
            dtype=torch.float,
        ).as_subclass(AgentObsTensor)
        neighbor_history_extents_t: Tensor = torch.full(
            (batch_size, 0, max_neigh_history_len, agent_history_extent_t.shape[-1]),
            np.nan,
        )

        neighbor_futures_t: AgentObsTensor = torch.full(
            (batch_size, 0, max_neigh_future_len, agent_future_t.shape[-1]),
            np.nan,
            dtype=torch.float,
        ).as_subclass(AgentObsTensor)
        neighbor_future_extents_t: Tensor = torch.full(
            (batch_size, 0, max_neigh_future_len, agent_future_extent_t.shape[-1]),
            np.nan,
        )

    robot_future_t: Optional[AgentObsTensor] = (
        pad_sequence(robot_future, batch_first=True, padding_value=np.nan).as_subclass(
            AgentObsTensor
        )
        if robot_future
        else None
    )

    (
        map_names,
        map_patches,
        maps_resolution,
        rasters_from_world_tf,
    ) = raster_map_collate_fn_agent(batch_elems)

    vector_maps: Optional[List[VectorMap]] = None
    if batch_elems[0].vec_map is not None:
        vector_maps = [batch_elem.vec_map for batch_elem in batch_elems]

    lane_xyh, lane_adj, lane_mask, lane_ids, road_edge_xyzh = (
        None,
        None,
        None,
        None,
        None,
    )
    if hasattr(batch_elems[0], "lane_xyh") and batch_elems[0].lane_xyh is not None:
        lane_xyh, lane_adj, lane_mask, lane_ids, road_edge_xyzh = _collate_lane_graph(
            batch_elems
        )

    agents_from_world_tf = torch.as_tensor(
        np.stack([batch_elem.agent_from_world_tf for batch_elem in batch_elems]),
        dtype=torch.float,
    )

    scene_ids = [batch_elem.scene_id for batch_elem in batch_elems]

    extras: Dict[str, Tensor] = {}
    for key in batch_elems[0].extras.keys():
        extras[key] = _collate_data(
            [batch_elem.extras[key] for batch_elem in batch_elems]
        )
    track_ids = (
        _collate_data([batch_elem.track_id for batch_elem in batch_elems])
        if hasattr(batch_elems[0], "track_id")
        else None
    )
    batch = AgentBatch(
        data_idx=data_index_t,
        scene_ts=scene_ts_t,
        dt=dt_t,
        agent_name=agent_names,
        agent_type=agent_type_t,
        curr_agent_state=curr_agent_state_t,
        agent_hist=agent_history_t,
        agent_hist_extent=agent_history_extent_t,
        agent_hist_len=agent_history_len,
        agent_fut=agent_future_t,
        agent_fut_extent=agent_future_extent_t,
        agent_fut_len=agent_future_len,
        num_neigh=num_neighbors_t,
        neigh_types=neighbor_types_t,
        neigh_hist=neighbor_histories_t,
        neigh_hist_extents=neighbor_history_extents_t,
        neigh_hist_len=neighbor_history_lens_t,
        neigh_fut=neighbor_futures_t,
        neigh_fut_extents=neighbor_future_extents_t,
        neigh_fut_len=neighbor_future_lens_t,
        robot_fut=robot_future_t,
        robot_fut_len=robot_future_len,
        map_names=map_names,
        maps=map_patches,
        lane_xyh=lane_xyh,
        lane_adj=lane_adj,
        lane_mask=lane_mask,
        lane_ids=lane_ids,
        road_edge_xyzh=road_edge_xyzh,
        maps_resolution=maps_resolution,
        vector_maps=vector_maps,
        rasters_from_world_tf=rasters_from_world_tf,
        agents_from_world_tf=agents_from_world_tf,
        scene_ids=scene_ids,
        history_pad_dir=history_pad_dir,
        track_ids=track_ids,
        extras=extras,
    )

    if batch_augments:
        for batch_aug in batch_augments:
            batch_aug.apply_agent(batch)

    if return_dict:
        return asdict(batch)

    return batch


def split_pad_crop(
    batch_tensor, sizes, pad_value: float = 0.0, desired_size: Optional[int] = None
) -> Tensor:
    """Split a batched tensor into different sizes and pad them to the same size

    Args:
        batch_tensor: tensor in bach or split tensor list
        sizes (torch.Tensor): sizes of each entry
        pad_value (float, optional): padding value. Defaults to 0.0
        desired_size (int, optional): desired size. Defaults to None.
    """

    if isinstance(batch_tensor, Tensor):
        x = torch.split(batch_tensor, sizes)
        cat_fun = torch.cat
        full_fun = torch.full
    elif isinstance(batch_tensor, np.ndarray):
        x = np.split(batch_tensor, sizes)
        cat_fun = np.concatenate
        full_fun = np.full
    elif isinstance(batch_tensor, List):
        # already splitted in list
        x = batch_tensor
        if isinstance(batch_tensor[0], Tensor):
            cat_fun = torch.cat
            full_fun = torch.full
        elif isinstance(batch_tensor[0], np.ndarray):
            cat_fun = np.concatenate
            full_fun = np.full
    else:
        raise ValueError("wrong data type for batch tensor")

    x: Tensor = pad_sequence(x, batch_first=True, padding_value=pad_value)
    if desired_size is not None:
        if x.shape[1] >= desired_size:
            x = x[:, :desired_size]
        else:
            bs, max_size = x.shape[:2]
            x = cat_fun(
                (x, full_fun([bs, desired_size - max_size, *x.shape[2:]], pad_value)), 1
            )

    return x


def scene_collate_fn(
    batch_elems: List[SceneBatchElement],
    return_dict: bool,
    pad_format: str,
    batch_augments: Optional[List[BatchAugmentation]] = None,
    desired_num_agents=None,
    desired_hist_len=None,
    desired_fut_len=None,
) -> SceneBatch:
    batch_size: int = len(batch_elems)
    history_pad_dir: arr_utils.PadDirection = (
        arr_utils.PadDirection.BEFORE
        if pad_format == "outside"
        else arr_utils.PadDirection.AFTER
    )

    data_index_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    scene_ts_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    dt_t: Tensor = torch.zeros((batch_size,), dtype=torch.float)

    # get agent state and obs format from first item in list
    state_format = batch_elems[0].centered_agent_state_np._format
    obs_format = batch_elems[0].cache.obs_type._format
    AgentStateTensor = TORCH_STATE_TYPES[state_format]
    AgentObsTensor = TORCH_STATE_TYPES[obs_format]

    max_agent_num: int = max(elem.num_agents for elem in batch_elems)
    if desired_num_agents is not None:
        max_agent_num = max(max_agent_num, desired_num_agents)

    centered_agent_state: List[AgentStateTensor] = list()
    agents_types: List[Tensor] = list()
    agents_histories: List[AgentObsTensor] = list()
    agents_history_extents: List[Tensor] = list()
    agents_history_len: Tensor = torch.zeros(
        (batch_size, max_agent_num), dtype=torch.long
    )

    agents_futures: List[Tensor] = list()
    agents_future_extents: List[AgentObsTensor] = list()
    agents_future_len: Tensor = torch.zeros(
        (batch_size, max_agent_num), dtype=torch.long
    )

    num_agents: List[int] = [elem.num_agents for elem in batch_elems]
    num_agents_t: Tensor = torch.as_tensor(num_agents, dtype=torch.long)

    max_history_len: int = max(elem.agent_history_lens_np.max() for elem in batch_elems)
    max_future_len: int = max(elem.agent_future_lens_np.max() for elem in batch_elems)
    if desired_hist_len is not None:
        max_history_len = max(max_history_len, desired_hist_len)
    if desired_fut_len is not None:
        max_future_len = max(max_future_len, desired_fut_len)

    robot_future: List[AgentObsTensor] = list()
    robot_future_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    for idx, elem in enumerate(batch_elems):
        data_index_t[idx] = elem.data_index
        scene_ts_t[idx] = elem.scene_ts
        dt_t[idx] = elem.dt
        centered_agent_state.append(elem.centered_agent_state_np)
        agents_types.append(elem.agent_types_np)
        history_len_i = torch.tensor(
            [rec.shape[0] for rec in elem.agent_histories[:max_agent_num]]
        )
        future_len_i = torch.tensor(
            [rec.shape[0] for rec in elem.agent_futures[:max_agent_num]]
        )
        agents_history_len[idx, : elem.num_agents] = history_len_i
        agents_future_len[idx, : elem.num_agents] = future_len_i

        # History
        padded_agents_histories = arr_utils.pad_sequences(
            elem.agent_histories[:max_agent_num],
            dtype=torch.float,
            time_dim=-2,
            pad_dir=history_pad_dir,
            batch_first=True,
            padding_value=np.nan,
        )
        padded_agents_history_extents = arr_utils.pad_sequences(
            elem.agent_history_extents[:max_agent_num],
            dtype=torch.float,
            time_dim=-2,
            pad_dir=history_pad_dir,
            batch_first=True,
            padding_value=np.nan,
        )
        if padded_agents_histories.shape[-2] < max_history_len:
            to_add = max_history_len - padded_agents_histories.shape[-2]
            padded_agents_histories = F.pad(
                padded_agents_histories,
                pad=(
                    (0, 0, to_add, 0)
                    if history_pad_dir == arr_utils.PadDirection.BEFORE
                    else (0, 0, 0, to_add)
                ),
                mode="constant",
                value=np.nan,
            )
            padded_agents_history_extents = F.pad(
                padded_agents_history_extents,
                pad=(
                    (0, 0, to_add, 0)
                    if history_pad_dir == arr_utils.PadDirection.BEFORE
                    else (0, 0, 0, to_add)
                ),
                mode="constant",
                value=np.nan,
            )

        agents_histories.append(padded_agents_histories)
        agents_history_extents.append(padded_agents_history_extents)

        # Future
        padded_agents_futures = pad_sequence(
            [
                torch.as_tensor(nh, dtype=torch.float)
                for nh in elem.agent_futures[:max_agent_num]
            ],
            batch_first=True,
            padding_value=np.nan,
        )
        padded_agents_future_extents = pad_sequence(
            [
                torch.as_tensor(nh, dtype=torch.float)
                for nh in elem.agent_future_extents
            ],
            batch_first=True,
            padding_value=np.nan,
        )
        if padded_agents_futures.shape[-2] < max_future_len:
            to_add = max_future_len - padded_agents_futures.shape[-2]
            padded_agents_futures = F.pad(
                padded_agents_futures,
                pad=(0, 0, 0, to_add),
                mode="constant",
                value=np.nan,
            )
            padded_agents_future_extents = F.pad(
                padded_agents_future_extents,
                pad=(0, 0, 0, to_add),
                mode="constant",
                value=np.nan,
            )

        agents_futures.append(padded_agents_futures)
        agents_future_extents.append(padded_agents_future_extents)

        if elem.robot_future_np is not None:
            robot_future.append(
                torch.as_tensor(elem.robot_future_np, dtype=torch.float)
            )
            robot_future_len[idx] = elem.robot_future_len

    agents_histories_t = split_pad_crop(
        agents_histories, num_agents, np.nan, max_agent_num
    ).as_subclass(AgentObsTensor)
    agents_history_extents_t = split_pad_crop(
        agents_history_extents, num_agents, np.nan, max_agent_num
    )
    agents_futures_t = split_pad_crop(
        agents_futures, num_agents, np.nan, max_agent_num
    ).as_subclass(AgentObsTensor)
    agents_future_extents_t = split_pad_crop(
        agents_future_extents, num_agents, np.nan, max_agent_num
    )

    centered_agent_state_t = torch.as_tensor(
        np.stack(centered_agent_state), dtype=torch.float
    ).as_subclass(AgentStateTensor)
    agents_types_t = torch.as_tensor(np.concatenate(agents_types))
    agents_types_t = split_pad_crop(
        agents_types_t, num_agents, pad_value=-1, desired_size=max_agent_num
    )

    (
        map_names,
        map_patches,
        maps_resolution,
        rasters_from_world_tf,
    ) = raster_map_collate_fn_scene(batch_elems, max_agent_num)

    vector_maps: Optional[List[VectorMap]] = None
    if batch_elems[0].vec_map is not None:
        vector_maps = [batch_elem.vec_map for batch_elem in batch_elems]

    lane_xyh, lane_adj, lane_mask, lane_ids, road_edge_xyzh = (
        None,
        None,
        None,
        None,
        None,
    )
    if hasattr(batch_elems[0], "lane_xyh") and batch_elems[0].lane_xyh is not None:
        lane_xyh, lane_adj, lane_mask, lane_ids, road_edge_xyzh = _collate_lane_graph(
            batch_elems
        )

    centered_agent_from_world_tf = torch.as_tensor(
        np.stack(
            [batch_elem.centered_agent_from_world_tf for batch_elem in batch_elems]
        ),
        dtype=torch.float,
    )
    centered_world_from_agent_tf = torch.as_tensor(
        np.stack(
            [batch_elem.centered_world_from_agent_tf for batch_elem in batch_elems]
        ),
        dtype=torch.float,
    )

    robot_future_t: Optional[Tensor] = (
        pad_sequence(robot_future, batch_first=True, padding_value=np.nan).as_subclass(
            AgentObsTensor
        )
        if robot_future
        else None
    )

    agent_names = [batch_elem.agent_names for batch_elem in batch_elems]

    scene_ids = [batch_elem.scene_id for batch_elem in batch_elems]

    extras: Dict[str, Tensor] = {}
    for key in batch_elems[0].extras.keys():
        extras[key] = _collate_data(
            [batch_elem.extras[key] for batch_elem in batch_elems]
        )

    batch = SceneBatch(
        data_idx=data_index_t,
        scene_ts=scene_ts_t,
        dt=dt_t,
        num_agents=num_agents_t,
        agent_type=agents_types_t,
        centered_agent_state=centered_agent_state_t,
        agent_names=agent_names,
        track_ids=None,
        agent_hist=agents_histories_t,
        agent_hist_extent=agents_history_extents_t,
        agent_hist_len=agents_history_len,
        agent_fut=agents_futures_t,
        agent_fut_extent=agents_future_extents_t,
        agent_fut_len=agents_future_len,
        robot_fut=robot_future_t,
        robot_fut_len=robot_future_len,
        map_names=map_names,
        maps=map_patches,
        lane_xyh=lane_xyh,
        lane_adj=lane_adj,
        lane_mask=lane_mask,
        lane_ids=lane_ids,
        road_edge_xyzh=road_edge_xyzh,
        maps_resolution=maps_resolution,
        vector_maps=vector_maps,
        rasters_from_world_tf=rasters_from_world_tf,
        centered_agent_from_world_tf=centered_agent_from_world_tf,
        centered_world_from_agent_tf=centered_world_from_agent_tf,
        scene_ids=scene_ids,
        history_pad_dir=history_pad_dir,
        extras=extras,
    )

    if batch_augments:
        for batch_aug in batch_augments:
            batch_aug.apply_scene(batch)

    if return_dict:
        return asdict(batch)

    return batch
