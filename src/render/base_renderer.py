from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Type, Union, Any
from typing_extensions import Literal

import torch
from torch.nn import Module, Parameter


class RenderState(dict):
    CAMERAS = "cameras"
    LIDAR = "lidar"
    AGENT_STATE = "agent_state"   # {object_id: np.ndarray([x, y, heading])}
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
        # [x, y, heading]
        raise NotImplementedError