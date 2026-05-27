# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass, field

import dataclasses_json


@dataclass(kw_only=True)
class Metadata(dataclasses_json.DataClassJsonMixin):
    scene_id: str
    version_string: str
    training_date: str
    dataset_hash: str
    uuid: str
    is_resumable: bool

    @dataclass
    class Sensors(dataclasses_json.DataClassJsonMixin):
        camera_ids: list[str] = field(default_factory=list)
        lidar_ids: list[str] = field(default_factory=list)

    sensors: Sensors

    @dataclass
    class Logger(dataclasses_json.DataClassJsonMixin):
        name: str | None = None
        run_id: str | None = None
        run_url: str | None = None

    logger: Logger

    @dataclass
    class TimeRange(dataclasses_json.DataClassJsonMixin):
        start: float
        end: float

    time_range: TimeRange

    training_step_outputs: dict[str, float] = field(default_factory=dict)
