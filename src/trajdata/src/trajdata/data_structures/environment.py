# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trajdata.data_structures.scene_tag import SceneTag


class EnvMetadata:
    def __init__(
        self,
        name: str,
        data_dir: str,
        dt: float,
        parts: List[Tuple[str]],
        scene_split_map: Dict[str, str],
        map_locations: Optional[Tuple[str]] = None,
    ) -> None:
        self.name = name
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.dt = dt
        self.map_locations = map_locations
        self.parts = parts
        self.scene_tags: List[SceneTag] = [
            SceneTag(tag_tuple)
            # Cartesian product of the given list of tuples
            for tag_tuple in itertools.product(*([(name,)] + parts))
        ]
        self.scene_split_map = scene_split_map
