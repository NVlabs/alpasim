# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from types import MappingProxyType
from typing import Final

OBSTACLE_CLASS_NAME_TO_ID: Final = MappingProxyType(
    {
        "car": 0,
        "truck": 1,
        "pedestrian": 2,
        "cyclist": 3,
        "others": 4,
    }
)
OBSTACLE_CLASS_ID_TO_NAME: Final = MappingProxyType(
    {class_id: name for name, class_id in OBSTACLE_CLASS_NAME_TO_ID.items()}
)


def obstacle_class_metadata() -> dict:
    return {
        "obstacle_class_name_2_id": dict(OBSTACLE_CLASS_NAME_TO_ID),
        "obstacle_class_id_2_name": dict(OBSTACLE_CLASS_ID_TO_NAME),
    }
