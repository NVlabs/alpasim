# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from abc import ABC

import torch


class RenderState(dict):
    CAMERAS = "cameras"
    LIDAR = "lidar"
    AGENT_STATE = "agent_state"
    TIMESTAMP = "timestamp"


class BaseRenderer(ABC):

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.__from_scratch__()

    def __from_scratch__(self):
        self.sensors = None

    def _check_for_reliance(self):
        if self.background_asset is None:
            raise ValueError("No background asset set for renderer.")

    def reset(self):
        self.__from_scratch__()

    @property
    def background_asset(self):
        return None

    def set_asset(self, asset):
        pass

    def render(self, render_state: RenderState):
        pass

    def physical_world(self, agent_state):
        raise NotImplementedError
