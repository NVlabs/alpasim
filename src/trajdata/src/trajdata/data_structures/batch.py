# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.state import StateTensor
from trajdata.maps import VectorMap
from trajdata.utils.arr_utils import (
    PadDirection,
    batch_nd_transform_xyvvaahh_pt,
    roll_with_tensor,
    transform_xyh_torch,
)


@dataclass
class AgentBatch:
    data_idx: Tensor
    scene_ts: Tensor
    dt: Tensor
    agent_name: List[str]
    agent_type: Tensor
    curr_agent_state: StateTensor
    agent_hist: StateTensor
    agent_hist_extent: Tensor
    agent_hist_len: Tensor
    agent_fut: StateTensor
    agent_fut_extent: Tensor
    agent_fut_len: Tensor
    num_neigh: Tensor
    neigh_types: Tensor
    neigh_hist: StateTensor
    neigh_hist_extents: Tensor
    neigh_hist_len: Tensor
    neigh_fut: StateTensor
    neigh_fut_extents: Tensor
    neigh_fut_len: Tensor
    agents_from_world_tf: Tensor
    history_pad_dir: PadDirection
    extras: Dict[str, Tensor]
    vector_maps: Optional[List[VectorMap]]
    rasters_from_world_tf: Optional[Tensor]
    scene_ids: Optional[List]
    robot_fut: Optional[StateTensor]
    robot_fut_len: Optional[Tensor]
    track_ids: Optional[List[List[str]]]
    map_names: Optional[List[str]]
    maps: Optional[Tensor]
    maps_resolution: Optional[Tensor]
    lane_xyh: Optional[Tensor]
    lane_adj: Optional[Tensor]
    lane_ids: Optional[List[List[str]]]
    lane_mask: Optional[Tensor]
    road_edge_xyzh: Optional[Tensor]
    road_edge_xyzh: Optional[Tensor] = None

    def to(self, device) -> None:
        excl_vals = {
            "data_idx",
            "agent_name",
            "agent_type",
            "agent_hist_len",
            "agent_fut_len",
            "neigh_hist_len",
            "neigh_fut_len",
            "neigh_types",
            "num_neigh",
            "robot_fut_len",
            "map_names",
            "vector_maps",
            "scene_ids",
            "history_pad_dir",
            "extras",
            "lane_ids",
        }
        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                tensor_val: Union[Tensor, StateTensor]
                setattr(self, val, tensor_val.to(device, non_blocking=True))

        for key, val in self.extras.items():
            # Allow for custom .to() method for objects that define a __to__ function.
            if hasattr(val, "__to__"):
                self.extras[key] = val.__to__(device, non_blocking=True)
            else:
                self.extras[key] = val.to(device, non_blocking=True)

    def agent_types(self) -> List[AgentType]:
        unique_types: Tensor = torch.unique(self.agent_type)
        return [AgentType(unique_type.item()) for unique_type in unique_types]

    def for_agent_type(self, agent_type: AgentType) -> AgentBatch:
        match_type = self.agent_type == agent_type
        return self.filter_batch(match_type)

    def filter_batch(self, filter_mask: torch.Tensor) -> AgentBatch:
        """Build a new batch with elements for which filter_mask[i] == True."""

        # Some of the tensors might be on different devices, so we define some convenience functions
        # to make sure the filter_mask is always on the same device as the tensor we are indexing.
        filter_mask_dict = {}
        filter_mask_dict["cpu"] = filter_mask.to("cpu")
        filter_mask_dict[str(self.agent_hist.device)] = filter_mask.to(
            self.agent_hist.device
        )

        _filter = lambda tensor: tensor[filter_mask_dict[str(tensor.device)]]
        _filter_tensor_or_list = lambda tensor_or_list: (
            _filter(tensor_or_list)
            if isinstance(tensor_or_list, torch.Tensor)
            else type(tensor_or_list)(
                [
                    el
                    for idx, el in enumerate(tensor_or_list)
                    if filter_mask_dict["cpu"][idx]
                ]
            )
        )

        return AgentBatch(
            data_idx=_filter(self.data_idx),
            scene_ts=_filter(self.scene_ts),
            dt=_filter(self.dt),
            agent_name=_filter_tensor_or_list(self.agent_name),
            agent_type=_filter(self.agent_type),
            curr_agent_state=_filter(self.curr_agent_state),
            agent_hist=_filter(self.agent_hist),
            agent_hist_extent=_filter(self.agent_hist_extent),
            agent_hist_len=_filter(self.agent_hist_len),
            agent_fut=_filter(self.agent_fut),
            agent_fut_extent=_filter(self.agent_fut_extent),
            agent_fut_len=_filter(self.agent_fut_len),
            num_neigh=_filter(self.num_neigh),
            neigh_types=_filter(self.neigh_types),
            neigh_hist=_filter(self.neigh_hist),
            neigh_hist_extents=_filter(self.neigh_hist_extents),
            neigh_hist_len=_filter(self.neigh_hist_len),
            neigh_fut=_filter(self.neigh_fut),
            neigh_fut_extents=_filter(self.neigh_fut_extents),
            neigh_fut_len=_filter(self.neigh_fut_len),
            robot_fut=_filter(self.robot_fut) if self.robot_fut is not None else None,
            robot_fut_len=(
                _filter(self.robot_fut_len) if self.robot_fut_len is not None else None
            ),
            map_names=(
                _filter_tensor_or_list(self.map_names)
                if self.map_names is not None
                else None
            ),
            maps=_filter(self.maps) if self.maps is not None else None,
            maps_resolution=(
                _filter(self.maps_resolution)
                if self.maps_resolution is not None
                else None
            ),
            vector_maps=(
                _filter_tensor_or_list(self.vector_maps)
                if self.vector_maps is not None
                else None
            ),
            rasters_from_world_tf=(
                _filter(self.rasters_from_world_tf)
                if self.rasters_from_world_tf is not None
                else None
            ),
            lane_xyh=_filter(self.lane_xyh) if self.lane_xyh is not None else None,
            lane_adj=_filter(self.lane_adj) if self.lane_adj is not None else None,
            lane_ids=self.lane_ids,
            lane_mask=_filter(self.lane_mask) if self.lane_mask is not None else None,
            road_edge_xyzh=(
                _filter(self.road_edge_xyzh)
                if self.road_edge_xyzh is not None
                else None
            ),
            agents_from_world_tf=_filter(self.agents_from_world_tf),
            scene_ids=_filter_tensor_or_list(self.scene_ids),
            history_pad_dir=self.history_pad_dir,
            extras={
                key: _filter_tensor_or_list(val) for key, val in self.extras.items()
            },
        )

    def to_scene_batch(self, agent_ind: int) -> SceneBatch:
        """
        Converts AgentBatch to SeceneBatch by combining neighbors and agent.

        The agent of AgentBatch will be treated as if it was the last neighbor.
        self.extras will be simply copied over, any custom conversion must be
        implemented externally.
        """

        batch_size = self.neigh_hist.shape[0]
        num_neigh = self.neigh_hist.shape[1]

        combine = lambda neigh, agent: torch.cat((neigh, agent.unsqueeze(0)), dim=0)
        combine_list = lambda neigh, agent: neigh + [agent]

        return SceneBatch(
            data_idx=self.data_idx,
            scene_ts=self.scene_ts,
            dt=self.dt,
            num_agents=self.num_neigh + 1,
            agent_type=combine(self.neigh_types, self.agent_type),
            centered_agent_state=self.curr_agent_state,  # TODO this is not actually the agent but the `global` coordinate frame
            agent_names=combine_list(
                ["UNKNOWN" for _ in range(num_neigh)], self.agent_name
            ),
            agent_hist=combine(self.neigh_hist, self.agent_hist),
            agent_hist_extent=combine(self.neigh_hist_extents, self.agent_hist_extent),
            agent_hist_len=combine(self.neigh_hist_len, self.agent_hist_len),
            agent_fut=combine(self.neigh_fut, self.agent_fut),
            agent_fut_extent=combine(self.neigh_fut_extents, self.agent_fut_extent),
            agent_fut_len=combine(self.neigh_fut_len, self.agent_fut_len),
            robot_fut=self.robot_fut,
            robot_fut_len=self.robot_fut_len,
            map_names=self.map_names,  # TODO
            maps=self.maps,
            maps_resolution=self.maps_resolution,
            vector_maps=self.vector_maps,
            rasters_from_world_tf=self.rasters_from_world_tf,
            centered_agent_from_world_tf=self.agents_from_world_tf,
            centered_world_from_agent_tf=torch.linalg.inv(self.agents_from_world_tf),
            scene_ids=self.scene_ids,
            history_pad_dir=self.history_pad_dir,
            extras=self.extras,
        )


@dataclass
class SceneBatch:
    data_idx: Tensor
    scene_ts: Tensor
    dt: Tensor
    num_agents: Tensor
    agent_type: Tensor
    centered_agent_state: StateTensor
    agent_names: List[List[str]]
    track_ids: Optional[List[List[str]]]
    agent_hist: StateTensor
    agent_hist_extent: Tensor
    agent_hist_len: Tensor
    agent_fut: StateTensor
    agent_fut_extent: Tensor
    agent_fut_len: Tensor
    centered_agent_from_world_tf: Tensor
    centered_world_from_agent_tf: Tensor
    history_pad_dir: PadDirection
    vector_maps: Optional[List[VectorMap]]
    lane_ids: Optional(List[List[str]])
    robot_fut: Optional[StateTensor]
    robot_fut_len: Optional[Tensor]
    map_names: Optional[Tensor]
    maps: Optional[Tensor]
    maps_resolution: Optional[Tensor]
    lane_xyh: Optional[Tensor]
    lane_adj: Optional[Tensor]
    lane_mask: Optional[Tensor]
    road_edge_xyzh: Optional[Tensor]
    rasters_from_world_tf: Optional[Tensor]
    scene_ids: Optional[List]

    extras: Dict[str, Tensor]

    def to(self, device) -> None:
        excl_vals = {
            "num_agents",
            "agent_names",
            "track_ids",
            "agent_type",
            "agent_hist_len",
            "agent_fut_len",
            "robot_fut_len",
            "map_names",
            "vector_maps",
            "history_pad_dir",
            "scene_ids",
            "extras",
            "lane_ids",
        }

        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                setattr(self, val, tensor_val.to(device))

        for key, val in self.extras.items():
            # Allow for custom .to() method for objects that define a __to__ function.
            if hasattr(val, "__to__"):
                self.extras[key] = val.__to__(device, non_blocking=True)
            else:
                self.extras[key] = val.to(device, non_blocking=True)
        return self

    def astype(self, dtype) -> None:
        new_obj = replace(self)
        excl_vals = {
            "num_agents",
            "agent_names",
            "track_ids",
            "agent_type",
            "agent_hist_len",
            "agent_fut_len",
            "robot_fut_len",
            "map_names",
            "vector_maps",
            "history_pad_dir",
            "scene_ids",
            "extras",
            "lane_ids",
        }

        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                setattr(new_obj, val, tensor_val.type(dtype))
        return new_obj

    def agent_types(self) -> List[AgentType]:
        unique_types: Tensor = torch.unique(self.agent_type)
        return [
            AgentType(unique_type.item())
            for unique_type in unique_types
            if unique_type >= 0
        ]

    def copy(self):
        # Shallow copy
        return replace(self)

    def convert_pad_direction(self, pad_dir: PadDirection) -> SceneBatch:
        if self.history_pad_dir == pad_dir:
            return self
        batch: SceneBatch = self.copy()
        if self.history_pad_dir == PadDirection.BEFORE:
            # n, n, -2 , -1, 0 -->  -2, -1, 0, n, n
            shifts = batch.agent_hist_len
        else:
            #  -2, -1, 0, n, n --> n, n, -2 , -1, 0
            shifts = -batch.agent_hist_len
        batch.agent_hist = roll_with_tensor(batch.agent_hist, shifts, dim=-2)
        batch.agent_hist_extent = roll_with_tensor(
            batch.agent_hist_extent, shifts, dim=-2
        )
        batch.history_pad_dir = pad_dir
        return batch

    def filter_batch(self, filter_mask: torch.Tensor) -> SceneBatch:
        """Build a new batch with elements for which filter_mask[i] == True."""

        if filter_mask.ndim != 1:
            raise ValueError("Expected 1d filter mask.")

        # Some of the tensors might be on different devices, so we define some convenience functions
        # to make sure the filter_mask is always on the same device as the tensor we are indexing.
        filter_mask_dict = {}
        filter_mask_dict["cpu"] = filter_mask.to("cpu")
        filter_mask_dict[str(self.agent_hist.device)] = filter_mask.to(
            self.agent_hist.device
        )

        # Use tensor.__class__ to keep TensorState.
        _filter = lambda tensor: tensor.__class__(
            tensor[filter_mask_dict[str(tensor.device)]]
        )
        _filter_tensor_or_list = lambda tensor_or_list: (
            _filter(tensor_or_list)
            if isinstance(tensor_or_list, torch.Tensor)
            else type(tensor_or_list)(
                [
                    el
                    for idx, el in enumerate(tensor_or_list)
                    if filter_mask_dict["cpu"][idx]
                ]
            )
        )

        return SceneBatch(
            data_idx=_filter(self.data_idx),
            scene_ts=_filter(self.scene_ts),
            dt=_filter(self.dt),
            num_agents=_filter(self.num_agents),
            agent_type=_filter(self.agent_type),
            agent_names=_filter_tensor_or_list(self.agent_names),
            track_ids=_filter_tensor_or_list(self.track_ids),
            centered_agent_state=_filter(self.centered_agent_state),
            agent_hist=_filter(self.agent_hist),
            agent_hist_extent=_filter(self.agent_hist_extent),
            agent_hist_len=_filter(self.agent_hist_len),
            agent_fut=_filter(self.agent_fut),
            agent_fut_extent=_filter(self.agent_fut_extent),
            agent_fut_len=_filter(self.agent_fut_len),
            robot_fut=_filter(self.robot_fut) if self.robot_fut is not None else None,
            robot_fut_len=(
                _filter(self.robot_fut_len) if self.robot_fut_len is not None else None
            ),
            map_names=(
                _filter_tensor_or_list(self.map_names)
                if self.map_names is not None
                else None
            ),
            maps=_filter(self.maps) if self.maps is not None else None,
            maps_resolution=(
                _filter(self.maps_resolution)
                if self.maps_resolution is not None
                else None
            ),
            vector_maps=(
                _filter_tensor_or_list(self.vector_maps)
                if self.vector_maps is not None
                else None
            ),
            lane_xyh=_filter(self.lane_xyh) if self.lane_xyh is not None else None,
            lane_adj=_filter(self.lane_adj) if self.lane_adj is not None else None,
            lane_ids=self.lane_ids,
            lane_mask=_filter(self.lane_mask) if self.lane_mask is not None else None,
            road_edge_xyzh=(
                _filter(self.road_edge_xyzh)
                if self.road_edge_xyzh is not None
                else None
            ),
            rasters_from_world_tf=(
                _filter(self.rasters_from_world_tf)
                if self.rasters_from_world_tf is not None
                else None
            ),
            centered_agent_from_world_tf=_filter(self.centered_agent_from_world_tf),
            centered_world_from_agent_tf=_filter(self.centered_world_from_agent_tf),
            scene_ids=_filter_tensor_or_list(self.scene_ids),
            history_pad_dir=self.history_pad_dir,
            extras={
                key: _filter_tensor_or_list(val) for key, val in self.extras.items()
            },
        )

    def to_agent_batch(self, agent_inds: torch.Tensor) -> AgentBatch:
        """
        Converts SeceneBatch to AgentBatch for agents defined by `agent_inds`.

        self.extras will be simply copied over, any custom conversion must be
        implemented externally.
        """

        batch_size = self.agent_hist.shape[0]
        num_agents = self.agent_hist.shape[1]

        if agent_inds.ndim != 1 or agent_inds.shape[0] != batch_size:
            raise ValueError("Wrong shape for agent_inds, expected [batch_size].")

        if (agent_inds < 0).any() or (agent_inds >= num_agents).any():
            raise ValueError("Invalid agent index")

        batch_inds = torch.arange(batch_size)
        others_mask = torch.ones((batch_size, num_agents), dtype=torch.bool)
        others_mask[batch_inds, agent_inds] = False
        index_agent = lambda x: x[batch_inds, agent_inds] if x is not None else None
        index_agent_list = lambda xlist: (
            [x[ind] for x, ind in zip(xlist, agent_inds)] if xlist is not None else None
        )
        index_neighbors = lambda x: x[others_mask].reshape(
            [
                batch_size,
                num_agents - 1,
            ]
            + list(x.shape[2:])
        )

        return AgentBatch(
            data_idx=self.data_idx,
            scene_ts=self.scene_ts,
            dt=self.dt,
            agent_name=index_agent_list(self.agent_names),
            track_ids=index_agent_list(self.track_ids),
            agent_type=index_agent(self.agent_type),
            curr_agent_state=self.centered_agent_state,  # TODO this is not actually the agent but the `global` coordinate frame
            agent_hist=StateTensor.from_array(
                index_agent(self.agent_hist), self.agent_hist._format
            ),
            agent_hist_extent=index_agent(self.agent_hist_extent),
            agent_hist_len=index_agent(self.agent_hist_len),
            agent_fut=StateTensor.from_array(
                index_agent(self.agent_fut), self.agent_fut._format
            ),
            agent_fut_extent=index_agent(self.agent_fut_extent),
            agent_fut_len=index_agent(self.agent_fut_len),
            num_neigh=self.num_agents - 1,
            neigh_types=index_neighbors(self.agent_type),
            neigh_hist=StateTensor.from_array(
                index_neighbors(self.agent_hist), self.agent_hist._format
            ),
            neigh_hist_extents=index_neighbors(self.agent_hist_extent),
            neigh_hist_len=index_neighbors(self.agent_hist_len),
            neigh_fut=StateTensor.from_array(
                index_neighbors(self.agent_fut), self.agent_fut._format
            ),
            neigh_fut_extents=index_neighbors(self.agent_fut_extent),
            neigh_fut_len=index_neighbors(self.agent_fut_len),
            robot_fut=self.robot_fut,
            robot_fut_len=self.robot_fut_len,
            map_names=self.map_names,
            maps=self.maps,
            vector_maps=self.vector_maps,
            lane_xyh=self.lane_xyh,
            lane_adj=self.lane_adj,
            lane_ids=self.lane_ids,
            lane_mask=self.lane_mask,
            road_edge_xyzh=self.road_edge_xyzh,
            maps_resolution=self.maps_resolution,
            rasters_from_world_tf=self.rasters_from_world_tf,
            agents_from_world_tf=self.centered_agent_from_world_tf,
            scene_ids=self.scene_ids,
            history_pad_dir=self.history_pad_dir,
            extras=self.extras,
        )

    def apply_transform(
        self, tf: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> SceneBatch:
        """
        Applies a transformation matrix to all coordinates stored in the SceneBatch.

        Returns a shallow copy, only coordinate fields are replaced.
        self.extras will be simply copied over (shallow copy), any custom conversion must be
        implemented externally.
        """
        assert tf.ndim == 3  # b, 3, 3
        assert tf.shape[-1] == 3 and tf.shape[-1] == 3
        assert (
            tf.dtype == torch.double
        )  # tf should be double precision, otherwise we have large numerical errors
        if dtype is None:
            dtype = self.agent_hist.dtype

        # Shallow copy
        batch: SceneBatch = replace(self)

        # TODO support generic format
        assert batch.agent_hist._format == "x,y,xd,yd,xdd,ydd,s,c"
        assert batch.agent_fut._format == "x,y,xd,yd,xdd,ydd,s,c"
        state_class = batch.agent_hist.__class__

        # Transforms
        batch.agent_hist = state_class(
            batch_nd_transform_xyvvaahh_pt(batch.agent_hist.double(), tf).type(dtype)
        )
        batch.agent_fut = state_class(
            batch_nd_transform_xyvvaahh_pt(batch.agent_fut.double(), tf).type(dtype)
        )
        batch.rasters_from_world_tf = (
            tf.unsqueeze(1) @ batch.rasters_from_world_tf
            if batch.rasters_from_world_tf is not None
            else None
        )
        batch.centered_agent_from_world_tf = tf @ batch.centered_agent_from_world_tf
        centered_world_from_agent_tf = torch.linalg.inv(
            batch.centered_agent_from_world_tf
        )
        if batch.lane_xyh is not None:
            batch.lane_xyh = transform_xyh_torch(batch.lane_xyh.double(), tf).type(
                dtype
            )
        if batch.road_edge_xyzh is not None:
            batch.road_edge_xyzh = transform_xyh_torch(
                batch.road_edge_xyzh.double(), tf
            ).type(dtype)
        # sanity check
        assert torch.isclose(
            batch.centered_world_from_agent_tf @ torch.linalg.inv(tf),
            centered_world_from_agent_tf,
            atol=1e-5,
        ).all()
        batch.centered_world_from_agent_tf = centered_world_from_agent_tf

        return batch
