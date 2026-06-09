# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import torch

from trajdata.augmentation.augmentation import BatchAugmentation
from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.utils.arr_utils import PadDirection, mask_up_to


class NoiseHistories(BatchAugmentation):
    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 0.1,
    ) -> None:
        self.mean = mean
        self.stddev = stddev

    def apply_agent(self, agent_batch: AgentBatch) -> None:
        agent_hist_noise = torch.normal(
            self.mean, self.stddev, size=agent_batch.agent_hist.shape
        )
        neigh_hist_noise = torch.normal(
            self.mean, self.stddev, size=agent_batch.neigh_hist.shape
        )

        if agent_batch.history_pad_dir == PadDirection.BEFORE:
            agent_hist_noise[..., -1, :] = 0
            neigh_hist_noise[..., -1, :] = 0
        else:
            len_mask = ~mask_up_to(
                agent_batch.agent_hist_len,
                delta=-1,
                max_len=agent_batch.agent_hist.shape[1],
            ).unsqueeze(-1)
            agent_hist_noise[len_mask.expand(-1, -1, agent_hist_noise.shape[-1])] = 0

            len_mask = ~mask_up_to(
                agent_batch.neigh_hist_len,
                delta=-1,
                max_len=agent_batch.neigh_hist.shape[2],
            ).unsqueeze(-1)
            neigh_hist_noise[
                len_mask.expand(-1, -1, -1, neigh_hist_noise.shape[-1])
            ] = 0

        agent_batch.agent_hist += agent_hist_noise
        agent_batch.neigh_hist += neigh_hist_noise

    def apply_scene(self, scene_batch: SceneBatch) -> None:
        scene_batch.agent_hist[..., :-1, :] += torch.normal(
            self.mean, self.stddev, size=scene_batch.agent_hist[..., :-1, :].shape
        )
