# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from dataclasses import dataclass
from typing import Literal

import pytest
from alpasim_utils.types import (
    T0_AUTO,
    ClipDataSource,
    Scene,
    parse_scene_key,
    scene_from_proto,
    scene_to_proto,
)


def test_clip_data_source_supplies_scene_identity() -> None:
    source = ClipDataSource(clip_id="clipgt-a", usdz_uuid="uuid-a")
    scene = Scene(source=source, policy_start_us=T0_AUTO)

    assert source.source_id == "clipgt-a"
    assert scene.scene_key == "clipgt-a@t0=AUTO"
    assert parse_scene_key(scene.scene_key) == ("clipgt-a", T0_AUTO)


@dataclass(frozen=True)
class _TestDataSource:
    source_id: str
    revision: str


def test_scene_accepts_data_source_protocol_implementations() -> None:
    scene = Scene(
        source=_TestDataSource(source_id="dataset-item", revision="v1"),
        policy_start_us=12,
    )

    assert scene.scene_key == "dataset-item@t0=12"


@pytest.mark.parametrize("source_id", ["", ".", "..", "a/b", "a\\b", "a@t0=1"])
def test_scene_rejects_unsafe_source_ids(source_id: str) -> None:
    with pytest.raises(ValueError, match="source_id"):
        Scene(
            source=_TestDataSource(source_id=source_id, revision="v1"),
            policy_start_us=0,
        )


@pytest.mark.parametrize(
    ("policy_start_us", "has_policy_start"),
    [(T0_AUTO, False), (0, True), (1_250_000, True)],
)
def test_scene_proto_round_trip_preserves_policy_start_presence(
    policy_start_us: int | Literal["AUTO"],
    has_policy_start: bool,
) -> None:
    scene = Scene(
        source=ClipDataSource(clip_id="clipgt-a", usdz_uuid="artifact-a"),
        policy_start_us=policy_start_us,
    )

    info = scene_to_proto(scene)

    assert info.source.WhichOneof("kind") == "clip"
    assert info.source.clip.clip_id == "clipgt-a"
    assert info.source.clip.usdz_uuid == "artifact-a"
    assert info.HasField("policy_start_us") is has_policy_start
    if policy_start_us != T0_AUTO:
        assert info.policy_start_us == policy_start_us
    assert scene_from_proto(info) == scene
