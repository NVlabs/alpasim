# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Dynamic actor conditioning tests for the built-in video-model renderer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from alpasim_grpc.v0.video_model_pb2 import ActorClassId
from alpasim_runtime.events.base import EventQueue
from alpasim_runtime.events.video_model.prefetch import VideoModelPrefetchEvent
from alpasim_runtime.services.video_model_service import ChunkResult
from alpasim_runtime.video_model.utils import build_dynamic_world_state_for_video_model
from alpasim_utils import geometry
from alpasim_utils.scenario import AABB, TrafficObject, TrafficObjects


def _traffic_pose_at(x: float) -> geometry.Pose:
    return geometry.Pose(
        np.array([x, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    )


def _traffic_object(
    *,
    track_id: str,
    timestamps_us: list[int],
    xs: list[float],
    label_class: str = "automobile",
    is_static: bool = False,
    aabb: AABB | None = None,
) -> TrafficObject:
    return TrafficObject(
        track_id=track_id,
        aabb=aabb or AABB(4.0, 1.8, 1.5),
        trajectory=geometry.Trajectory.from_poses(
            np.array(timestamps_us, dtype=np.uint64),
            [_traffic_pose_at(x) for x in xs],
        ),
        is_static=is_static,
        label_class=label_class,
    )


def test_dynamic_world_state_samples_only_current_chunk_non_static_actors() -> None:
    traffic_objects = TrafficObjects(
        {
            "car-1": _traffic_object(
                track_id="car-1",
                timestamps_us=[0, 100_000, 200_000, 300_000],
                xs=[0.0, 1.0, 2.0, 3.0],
                label_class="automobile",
                aabb=AABB(4.5, 2.0, 1.6),
            ),
            "parked": _traffic_object(
                track_id="parked",
                timestamps_us=[0, 100_000, 200_000],
                xs=[10.0, 10.0, 10.0],
                label_class="automobile",
                is_static=True,
            ),
            "late-ped": _traffic_object(
                track_id="late-ped",
                timestamps_us=[200_000, 300_000],
                xs=[5.0, 6.0],
                label_class="pedestrian",
                aabb=AABB(0.8, 0.8, 1.8),
            ),
        }
    )

    state = build_dynamic_world_state_for_video_model(
        traffic_objects,
        start_us=100_000,
        num_frames=3,
        frame_interval_us=100_000,
    )

    assert len(state.actors) == 2
    car, pedestrian = state.actors
    assert car.class_id == ActorClassId.CAR
    assert car.bbox_dims.size_x == pytest.approx(4.5)
    assert [p.timestamp_us for p in car.trajectory.poses] == [
        100_000,
        200_000,
        300_000,
    ]
    assert [p.pose.vec.x for p in car.trajectory.poses] == pytest.approx(
        [1.0, 2.0, 3.0]
    )

    assert pedestrian.class_id == ActorClassId.PEDESTRIAN
    assert [p.timestamp_us for p in pedestrian.trajectory.poses] == [
        200_000,
        300_000,
    ]


def test_dynamic_world_state_orders_actors_by_track_id() -> None:
    traffic_objects = TrafficObjects(
        {
            "z-car": _traffic_object(
                track_id="z-car",
                timestamps_us=[0, 100_000],
                xs=[0.0, 1.0],
                label_class="automobile",
            ),
            "a-ped": _traffic_object(
                track_id="a-ped",
                timestamps_us=[0, 100_000],
                xs=[10.0, 11.0],
                label_class="pedestrian",
            ),
        }
    )

    state = build_dynamic_world_state_for_video_model(
        traffic_objects,
        start_us=0,
        num_frames=2,
        frame_interval_us=100_000,
    )

    assert [actor.class_id for actor in state.actors] == [
        ActorClassId.PEDESTRIAN,
        ActorClassId.CAR,
    ]


@pytest.mark.parametrize(
    ("label_class", "expected_class_id"),
    [
        ("animal", ActorClassId.OTHER),
        ("auto", ActorClassId.CAR),
        ("automobile", ActorClassId.CAR),
        ("bus", ActorClassId.TRUCK),
        ("car", ActorClassId.CAR),
        ("cyclist", ActorClassId.CYCLIST),
        ("heavy_truck", ActorClassId.TRUCK),
        ("other_vehicle", ActorClassId.OTHER),
        ("passenger-car", ActorClassId.CAR),
        ("pedestrian", ActorClassId.PEDESTRIAN),
        ("person", ActorClassId.PEDESTRIAN),
        ("protruding_object", ActorClassId.OTHER),
        ("rider", ActorClassId.CYCLIST),
        ("stroller", ActorClassId.PEDESTRIAN),
        ("trailer", ActorClassId.TRUCK),
        ("train_or_tram_car", ActorClassId.OTHER),
        ("truck", ActorClassId.TRUCK),
        ("VEHICLE", ActorClassId.OTHER),
        ("bicycle", ActorClassId.CYCLIST),
        ("bike", ActorClassId.CYCLIST),
        ("trolley_bus", ActorClassId.TRUCK),
    ],
)
def test_dynamic_world_state_maps_clipgt_labels_to_actor_classes(
    label_class: str, expected_class_id: int
) -> None:
    traffic_objects = TrafficObjects(
        {
            "actor": _traffic_object(
                track_id="actor",
                timestamps_us=[0, 100_000],
                xs=[0.0, 1.0],
                label_class=label_class,
            )
        }
    )

    state = build_dynamic_world_state_for_video_model(
        traffic_objects,
        start_us=0,
        num_frames=2,
        frame_interval_us=100_000,
    )

    assert len(state.actors) == 1
    assert state.actors[0].class_id == expected_class_id


@pytest.mark.asyncio
async def test_video_model_prefetch_passes_traffic_objects_as_dynamic_state() -> None:
    """Prefetch builds dynamic conditioning from current rollout traffic state."""

    class FakeVideoModel:
        frame_interval_us = 100_000
        chunk_size = 2
        config = SimpleNamespace(
            frame_forwarding_mode="all",
            forward_hdmap_to_driver=False,
        )

        def __init__(self) -> None:
            self.dynamic_state = None

        async def render_chunk(self, *, trajectory_local_to_rig, dynamic_state):
            del trajectory_local_to_rig
            self.dynamic_state = dynamic_state
            return ChunkResult()

    pose_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    ego_traj = geometry.Trajectory.from_poses(
        np.array([100_000, 200_000, 300_000], dtype=np.uint64),
        [
            geometry.Pose(np.array([1.0, 0.0, 0.0]), pose_quat),
            geometry.Pose(np.array([2.0, 0.0, 0.0]), pose_quat),
            geometry.Pose(np.array([3.0, 0.0, 0.0]), pose_quat),
        ],
    )
    traffic_traj = geometry.Trajectory.from_poses(
        np.array([200_000, 300_000], dtype=np.uint64),
        [
            geometry.Pose(np.array([20.0, 0.0, 0.0]), pose_quat),
            geometry.Pose(np.array([30.0, 0.0, 0.0]), pose_quat),
        ],
    )
    state = SimpleNamespace(
        ego_trajectory=ego_traj,
        unbound=SimpleNamespace(hidden_traffic_objs=None),
        traffic_objs=TrafficObjects(
            {
                "actor-1": TrafficObject(
                    track_id="actor-1",
                    aabb=AABB(4.0, 1.8, 1.5),
                    trajectory=traffic_traj,
                    is_static=False,
                    label_class="automobile",
                )
            }
        ),
    )
    video_model = FakeVideoModel()
    event = VideoModelPrefetchEvent(
        timestamp_us=200_000,
        chunk_start_us=200_000,
        chunk_size=2,
        is_first_chunk=False,
        video_model=video_model,  # type: ignore[arg-type]
        broadcaster=SimpleNamespace(),
        runtime_cameras=[],
        driver=SimpleNamespace(),
    )

    await event.handle(state, EventQueue())  # type: ignore[arg-type]

    assert video_model.dynamic_state is not None
    assert len(video_model.dynamic_state.actors) == 1
    actor = video_model.dynamic_state.actors[0]
    assert [p.timestamp_us for p in actor.trajectory.poses] == [200_000, 300_000]
    assert [p.pose.vec.x for p in actor.trajectory.poses] == pytest.approx([20.0, 30.0])
