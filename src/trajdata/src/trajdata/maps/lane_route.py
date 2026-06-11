# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from dataclasses import dataclass
from typing import Set


@dataclass
class LaneRoute:
    lane_idxs: Set[int]
