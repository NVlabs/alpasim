# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest
from alpasim_grpc.v0 import runtime_pb2
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata


@pytest.mark.parametrize(
    ("policy_start_us", "has_policy_start"),
    [(None, False), (0, True), (1_250_000, True)],
)
def test_rollout_spec_preserves_policy_start_presence(
    policy_start_us: int | None,
    has_policy_start: bool,
) -> None:
    spec = runtime_pb2.RolloutSpec(
        nr_rollouts=1,
        scene=runtime_pb2.SceneInfo(
            source=runtime_pb2.DataSource(
                clip=runtime_pb2.ClipDataSource(
                    clip_id="clipgt-a", usdz_uuid="artifact-a"
                )
            )
        ),
    )
    if policy_start_us is not None:
        spec.scene.policy_start_us = policy_start_us

    assert spec.scene.source.clip.clip_id == "clipgt-a"
    assert spec.scene.HasField("policy_start_us") is has_policy_start
    if policy_start_us is not None:
        assert spec.scene.policy_start_us == policy_start_us


def test_scene_info_embeds_the_pinned_data_source() -> None:
    scene = runtime_pb2.SceneInfo(
        source=runtime_pb2.DataSource(
            clip=runtime_pb2.ClipDataSource(
                clip_id="clipgt-a",
                usdz_uuid="artifact-a",
            )
        ),
        policy_start_us=0,
    )

    assert scene.source.WhichOneof("kind") == "clip"
    assert scene.source.clip.clip_id == "clipgt-a"
    assert scene.source.clip.usdz_uuid == "artifact-a"
    assert scene.HasField("policy_start_us")
    assert scene.policy_start_us == 0


def test_scene_identity_schema_uses_structured_scene() -> None:
    spec_fields = runtime_pb2.RolloutSpec.DESCRIPTOR.fields_by_name
    session_fields = RolloutMetadata.SessionMetadata.DESCRIPTOR.fields_by_name
    return_fields = runtime_pb2.SimulationReturn.RolloutReturn.DESCRIPTOR.fields_by_name

    assert "scenario_id" not in spec_fields
    assert "start_time_offset_us" not in spec_fields
    assert spec_fields["random_seed"].number == 5
    assert spec_fields["scene"].number == 6
    assert spec_fields["ego_history"].number == 7
    assert "source_id" not in spec_fields
    assert "policy_start_us" not in spec_fields
    assert "scene_id" not in session_fields
    assert "batch_size" not in session_fields
    assert session_fields["scene"].number == 11
    assert "effective_policy_start_us" not in session_fields
    assert "scene_key" not in session_fields
    assert "scene" not in return_fields
    assert "effective_policy_start_us" not in return_fields


def test_runtime_scene_schema_uses_source_for_identity() -> None:
    scene_fields = runtime_pb2.SceneInfo.DESCRIPTOR.fields_by_name
    metadata_fields = runtime_pb2.SceneMetadata.DESCRIPTOR.fields_by_name
    runtime_fields = runtime_pb2.RuntimeInfo.DESCRIPTOR.fields_by_name

    assert "scene_id" not in scene_fields
    assert "provider_kind" not in scene_fields
    assert scene_fields["metadata"].number == 3
    assert scene_fields["source"].number == 4
    assert scene_fields["policy_start_us"].number == 5
    assert "uuid" not in metadata_fields
    assert runtime_fields["scenes"].number == 4
    assert "data_sources" not in runtime_fields
