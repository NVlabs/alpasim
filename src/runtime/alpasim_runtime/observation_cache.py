# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared recent-frame observation cache for runtime-side driver orchestration."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from alpasim_utils import geometry
from alpasim_utils.scenario import TrafficObject, TrafficObjects
from alpasim_utils.types import ImageWithMetadata


def _clone_image(image: ImageWithMetadata) -> ImageWithMetadata:
    return ImageWithMetadata(
        start_timestamp_us=image.start_timestamp_us,
        end_timestamp_us=image.end_timestamp_us,
        image_bytes=image.image_bytes,
        camera_logical_id=image.camera_logical_id,
    )


def _clone_traffic_objects(traffic_objs: TrafficObjects) -> TrafficObjects:
    cloned = {}
    for object_id, traffic_obj in traffic_objs.items():
        time_range = traffic_obj.trajectory.time_range_us
        cloned[object_id] = TrafficObject(
            track_id=traffic_obj.track_id,
            aabb=traffic_obj.aabb,
            trajectory=traffic_obj.trajectory.clip(time_range.start, time_range.stop),
            is_static=traffic_obj.is_static,
            label_class=traffic_obj.label_class,
        )
    return TrafficObjects(cloned)


def _normalize_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_cache_value(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        try:
            return _normalize_cache_value(value.tolist())
        except Exception:
            pass
    return repr(value)


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    step_id: int
    input_snapshot_id: str
    time_now_us: int
    time_query_us: int
    camera_frame_timestamps_us: dict[str, int]
    rendered_images: dict[str, ImageWithMetadata]
    renderer_data: bytes | None
    ego_trajectory: geometry.DynamicTrajectory
    ego_trajectory_estimate: geometry.DynamicTrajectory | None
    traffic_objs: TrafficObjects
    ego_pose_history_timestamps_us: list[int]
    route_waypoints_in_rig: list[list[float]]
    planner_context: dict[str, Any] | None
    active_backend_ids: list[str]

    def clone(self) -> ObservationFrame:
        return ObservationFrame(
            step_id=self.step_id,
            input_snapshot_id=self.input_snapshot_id,
            time_now_us=self.time_now_us,
            time_query_us=self.time_query_us,
            camera_frame_timestamps_us=dict(self.camera_frame_timestamps_us),
            rendered_images={
                camera_id: _clone_image(image)
                for camera_id, image in self.rendered_images.items()
            },
            renderer_data=self.renderer_data,
            ego_trajectory=self.ego_trajectory.clone(),
            ego_trajectory_estimate=(
                self.ego_trajectory_estimate.clone()
                if self.ego_trajectory_estimate is not None
                else None
            ),
            traffic_objs=_clone_traffic_objects(self.traffic_objs),
            ego_pose_history_timestamps_us=list(self.ego_pose_history_timestamps_us),
            route_waypoints_in_rig=[list(waypoint) for waypoint in self.route_waypoints_in_rig],
            planner_context=_normalize_cache_value(self.planner_context),
            active_backend_ids=list(self.active_backend_ids),
        )


@dataclass(frozen=True, slots=True)
class ObservationWindow:
    anchor_snapshot_id: str
    frames: list[ObservationFrame]


@dataclass(frozen=True, slots=True)
class ObservationCacheCheckpoint:
    latest_snapshot_id: str | None
    frames_by_snapshot_id: dict[str, ObservationFrame]
    snapshot_order: list[str]


class ObservationCache:
    """Bounded in-memory cache of recent shared observation frames."""

    def __init__(self, max_frames: int = 32) -> None:
        self._max_frames = max(1, max_frames)
        self._frames_by_snapshot_id: OrderedDict[str, ObservationFrame] = OrderedDict()
        self._latest_snapshot_id: str | None = None

    def append(self, frame: ObservationFrame) -> None:
        self._frames_by_snapshot_id[frame.input_snapshot_id] = frame.clone()
        self._frames_by_snapshot_id.move_to_end(frame.input_snapshot_id)
        self._latest_snapshot_id = frame.input_snapshot_id
        while len(self._frames_by_snapshot_id) > self._max_frames:
            self._frames_by_snapshot_id.popitem(last=False)
        if self._latest_snapshot_id not in self._frames_by_snapshot_id:
            self._latest_snapshot_id = (
                next(reversed(self._frames_by_snapshot_id))
                if self._frames_by_snapshot_id
                else None
            )

    def get(self, input_snapshot_id: str) -> ObservationFrame:
        return self._frames_by_snapshot_id[input_snapshot_id].clone()

    def latest(self) -> ObservationFrame | None:
        if self._latest_snapshot_id is None:
            return None
        return self.get(self._latest_snapshot_id)

    def get_window(
        self,
        input_snapshot_id: str,
        window_size: int,
    ) -> ObservationWindow:
        snapshot_ids = list(self._frames_by_snapshot_id.keys())
        try:
            end_idx = snapshot_ids.index(input_snapshot_id) + 1
        except ValueError as exc:
            raise KeyError(f"Unknown input_snapshot_id: {input_snapshot_id}") from exc
        start_idx = max(0, end_idx - max(1, window_size))
        selected_ids = snapshot_ids[start_idx:end_idx]
        return ObservationWindow(
            anchor_snapshot_id=input_snapshot_id,
            frames=[self._frames_by_snapshot_id[snapshot_id].clone() for snapshot_id in selected_ids],
        )

    def list_snapshot_ids(self) -> list[str]:
        return list(self._frames_by_snapshot_id.keys())

    def checkpoint(self) -> ObservationCacheCheckpoint:
        return ObservationCacheCheckpoint(
            latest_snapshot_id=self._latest_snapshot_id,
            frames_by_snapshot_id={
                snapshot_id: frame.clone()
                for snapshot_id, frame in self._frames_by_snapshot_id.items()
            },
            snapshot_order=list(self._frames_by_snapshot_id.keys()),
        )

    def restore(self, checkpoint: ObservationCacheCheckpoint) -> None:
        self._frames_by_snapshot_id = OrderedDict(
            (snapshot_id, checkpoint.frames_by_snapshot_id[snapshot_id].clone())
            for snapshot_id in checkpoint.snapshot_order
        )
        self._latest_snapshot_id = checkpoint.latest_snapshot_id


class ObservationCacheReader:
    """Read-only helper for backend adapters to access shared observation windows."""

    def __init__(self, cache: ObservationCache) -> None:
        self._cache = cache

    def get_frame(self, input_snapshot_id: str) -> ObservationFrame:
        return self._cache.get(input_snapshot_id)

    def latest(self) -> ObservationFrame | None:
        return self._cache.latest()

    def get_window(
        self,
        input_snapshot_id: str,
        window_size: int,
    ) -> ObservationWindow:
        return self._cache.get_window(input_snapshot_id, window_size)

    def build_window_summary(
        self,
        input_snapshot_id: str,
        window_size: int,
    ) -> dict[str, Any]:
        window = self.get_window(input_snapshot_id, window_size)
        return {
            "anchor_snapshot_id": window.anchor_snapshot_id,
            "window_size": window_size,
            "available_frames": len(window.frames),
            "frames": [
                {
                    "input_snapshot_id": frame.input_snapshot_id,
                    "step_id": frame.step_id,
                    "time_now_us": frame.time_now_us,
                    "time_query_us": frame.time_query_us,
                    "camera_ids": sorted(frame.rendered_images.keys()),
                    "camera_frame_timestamps_us": dict(frame.camera_frame_timestamps_us),
                    "active_backend_ids": list(frame.active_backend_ids),
                }
                for frame in window.frames
            ],
        }
