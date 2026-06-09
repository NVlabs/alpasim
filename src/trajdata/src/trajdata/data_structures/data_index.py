# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm


class DataIndex:
    """The data index is effectively a big list of tuples taking the form:

    [(scene_path, total_index_len, valid_scene_ts)] for scene-centric data, or
    [(scene_path, total_index_len, [(agent_name, valid_agent_ts)])] for agent-centric data
    """

    def __init__(
        self,
        data_index: Union[
            List[Tuple[str, int, np.ndarray]],
            List[Tuple[str, int, List[Tuple[str, np.ndarray]]]],
        ],
        verbose: bool = False,
    ) -> None:
        scene_paths, full_index_len, _ = zip(*data_index)

        self._cumulative_lengths: np.ndarray = np.concatenate(
            ([0], np.cumsum(full_index_len))
        )
        self._len: int = self._cumulative_lengths[-1].item()

        self._scene_paths: np.ndarray = np.array(scene_paths).astype(np.bytes_)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Tuple[str, int, int]:
        scene_idx: int = (
            np.searchsorted(self._cumulative_lengths, index, side="right").item() - 1
        )

        scene_path: str = str(self._scene_paths[scene_idx], encoding="utf-8")
        scene_elem_index: int = index - self._cumulative_lengths[scene_idx].item()
        return (scene_path, scene_idx, scene_elem_index)


class AgentDataIndex(DataIndex):
    def __init__(
        self,
        data_index: List[Tuple[str, int, List[Tuple[str, np.ndarray]]]],
        verbose: bool = False,
    ) -> None:
        super().__init__(data_index)

        agent_timesteps: List[List[Tuple[str, np.ndarray]]] = [
            agent_ts_index for _, _, agent_ts_index in data_index
        ]

        self._agent_ids: List[np.ndarray] = list()
        self._agent_times: List[np.ndarray] = list()
        self._cumulative_scene_lengths: List[np.ndarray] = list()
        for scene_data_index in tqdm(
            agent_timesteps, desc="Structuring Agent Data Index", disable=not verbose
        ):
            agent_ids, agent_times = zip(*scene_data_index)

            self._agent_ids.append(np.array(agent_ids).astype(np.bytes_))

            agent_ts: np.ndarray = np.stack(agent_times)
            self._agent_times.append(agent_ts)
            self._cumulative_scene_lengths.append(
                np.concatenate(([0], np.cumsum(agent_ts[:, 1] - agent_ts[:, 0] + 1)))
            )

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        scene_path, scene_idx, scene_elem_index = super().__getitem__(index)

        agent_idx: int = (
            np.searchsorted(
                self._cumulative_scene_lengths[scene_idx],
                scene_elem_index,
                side="right",
            ).item()
            - 1
        )

        agent_id: str = str(self._agent_ids[scene_idx][agent_idx], encoding="utf-8")

        agent_timestep: int = (
            scene_elem_index
            - self._cumulative_scene_lengths[scene_idx][agent_idx].item()
            + self._agent_times[scene_idx][agent_idx, 0]
        ).item()

        assert (
            self._agent_times[scene_idx][agent_idx, 0]
            <= agent_timestep
            <= self._agent_times[scene_idx][agent_idx, 1]
        )

        return scene_path, agent_id, agent_timestep


class SceneDataIndex(DataIndex):
    def __init__(
        self, data_index: List[Tuple[str, int, np.ndarray]], verbose: bool = False
    ) -> None:
        super().__init__(data_index)

        self.scene_ts: List[np.ndarray] = [
            valid_ts
            for _, _, valid_ts in tqdm(
                data_index, desc="Structuring Scene Data Index", disable=not verbose
            )
        ]

    def __getitem__(self, index: int) -> Tuple[str, int]:
        scene_path, scene_idx, scene_elem_index = super().__getitem__(index)

        scene_ts: int = self.scene_ts[scene_idx][scene_elem_index].item()

        return scene_path, scene_ts
