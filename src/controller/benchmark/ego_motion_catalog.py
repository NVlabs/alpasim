# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Ego-motion trajectory catalog and k-disk benchmark selection."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .trajectories import ReferenceTrajectory, TrajectoryGenerator, TrajectoryPoint

_EPSILON = 1e-9
_DEFAULT_FEATURE_NAMES = (
    "path_length_m",
    "lateral_displacement_m",
    "mean_speed_mps",
    "max_speed_mps",
    "delta_speed_mps",
    "mean_abs_accel_mps2",
    "max_abs_accel_mps2",
    "yaw_delta_rad",
    "max_abs_yaw_rate_radps",
    "mean_abs_curvature_1pm",
    "max_abs_curvature_1pm",
)


@dataclass
class EgoMotionRecord:
    """One ego-motion clip in a local frame."""

    name: str
    description: str
    timestamps_us: list[int]
    x: list[float]
    y: list[float]
    yaw: list[float]
    vx: list[float]
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate array lengths and time ordering."""
        lengths = {
            len(self.timestamps_us),
            len(self.x),
            len(self.y),
            len(self.yaw),
            len(self.vx),
        }
        if lengths != {len(self.timestamps_us)}:
            raise ValueError(
                f"{self.name}: timestamps, pose, yaw, and vx lengths must match."
            )
        if len(self.timestamps_us) < 2:
            raise ValueError(
                f"{self.name}: ego-motion records need at least two samples."
            )
        if any(t1 <= t0 for t0, t1 in zip(self.timestamps_us, self.timestamps_us[1:])):
            raise ValueError(f"{self.name}: timestamps_us must be strictly increasing.")


@dataclass
class EgoMotionCatalog:
    """Serializable ego-motion benchmark catalog."""

    records: list[EgoMotionRecord]
    selected_names: list[str] = field(default_factory=list)
    selection_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate records and selected names."""
        names = set()
        for record in self.records:
            record.validate()
            if record.name in names:
                raise ValueError(f"Duplicate ego-motion record name: {record.name}")
            names.add(record.name)
        missing_names = sorted(set(self.selected_names) - names)
        if missing_names:
            raise ValueError(
                f"Selected records are missing from the catalog: {missing_names}"
            )

    def with_selected_cover(self, k: int) -> "EgoMotionCatalog":
        """Return a copy with selected names set by greedy k-disk coverage."""
        indices, metadata = select_k_disk_cover(self.records, k)
        selected_names = [self.records[index].name for index in indices]
        return EgoMotionCatalog(
            records=self.records,
            selected_names=selected_names,
            selection_metadata=metadata,
        )

    def selected_records(self) -> list[EgoMotionRecord]:
        """Return selected records, or all records when no selected set is stored."""
        if not self.selected_names:
            return list(self.records)
        selected_names = set(self.selected_names)
        return [record for record in self.records if record.name in selected_names]

    def to_reference_trajectories(self) -> list[ReferenceTrajectory]:
        """Convert selected ego-motion records into benchmark reference trajectories."""
        return [
            record_to_reference_trajectory(record) for record in self.selected_records()
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert the catalog to a JSON-serializable dictionary."""
        return {
            "schema_version": 1,
            "records": [
                {
                    "name": record.name,
                    "description": record.description,
                    "source_id": record.source_id,
                    "timestamps_us": record.timestamps_us,
                    "x": record.x,
                    "y": record.y,
                    "yaw": record.yaw,
                    "vx": record.vx,
                    "features": record.features,
                    "metadata": record.metadata,
                }
                for record in self.records
            ],
            "selected_names": self.selected_names,
            "selection_metadata": self.selection_metadata,
        }

    def save(self, path: Path) -> None:
        """Write the catalog to JSON."""
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EgoMotionCatalog":
        """Build a catalog from decoded JSON data."""
        records = [
            EgoMotionRecord(
                name=str(record["name"]),
                description=str(record.get("description", "")),
                source_id=str(record.get("source_id", "")),
                timestamps_us=[int(value) for value in record["timestamps_us"]],
                x=[float(value) for value in record["x"]],
                y=[float(value) for value in record["y"]],
                yaw=[float(value) for value in record["yaw"]],
                vx=[float(value) for value in record["vx"]],
                features={
                    key: float(value)
                    for key, value in record.get("features", {}).items()
                },
                metadata=dict(record.get("metadata", {})),
            )
            for record in data["records"]
        ]
        catalog = cls(
            records=records,
            selected_names=[str(name) for name in data.get("selected_names", [])],
            selection_metadata=dict(data.get("selection_metadata", {})),
        )
        catalog.validate()
        return catalog

    @classmethod
    def load(cls, path: Path) -> "EgoMotionCatalog":
        """Load a catalog from JSON."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def make_ego_motion_record(
    name: str,
    description: str,
    timestamps_us: list[int],
    x: list[float],
    y: list[float],
    yaw: list[float] | None = None,
    source_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> EgoMotionRecord:
    """Create an ego-motion record and derive missing yaw and speed."""
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if yaw is None:
        yaw_array = _yaw_from_xy(timestamps, x_array, y_array)
    else:
        yaw_array = np.asarray(yaw, dtype=float)
    vx_array = _vx_from_pose(timestamps, x_array, y_array, yaw_array)
    record = EgoMotionRecord(
        name=name,
        description=description,
        timestamps_us=timestamps.astype(int).tolist(),
        x=x_array.astype(float).tolist(),
        y=y_array.astype(float).tolist(),
        yaw=yaw_array.astype(float).tolist(),
        vx=vx_array.astype(float).tolist(),
        source_id=source_id,
        metadata=metadata or {},
    )
    record.validate()
    record.features = compute_ego_motion_features(record)
    return record


def record_to_reference_trajectory(record: EgoMotionRecord) -> ReferenceTrajectory:
    """Convert one ego-motion record into a benchmark reference trajectory."""
    record.validate()
    resampled_record = resample_record(record, TrajectoryGenerator.DT_US)
    timestamps = np.asarray(resampled_record.timestamps_us, dtype=np.int64)
    rebased_timestamps = timestamps - int(timestamps[0])
    points = [
        TrajectoryPoint(
            timestamp_us=int(timestamp_us),
            x=float(x),
            y=float(y),
            yaw=float(yaw),
            vx=float(vx),
        )
        for timestamp_us, x, y, yaw, vx in zip(
            rebased_timestamps,
            resampled_record.x,
            resampled_record.y,
            resampled_record.yaw,
            resampled_record.vx,
        )
    ]
    trajectory_duration_us = int(rebased_timestamps[-1])
    lookahead_duration_us = (
        TrajectoryGenerator.HORIZON_POINTS - 1
    ) * TrajectoryGenerator.DT_US
    if trajectory_duration_us > lookahead_duration_us:
        simulation_duration_us = trajectory_duration_us - lookahead_duration_us
    else:
        simulation_duration_us = trajectory_duration_us
    if simulation_duration_us <= 0:
        raise ValueError(
            f"{record.name}: record is too short for a benchmark simulation."
        )
    return ReferenceTrajectory(
        name=record.name,
        description=record.description,
        points=points,
        simulation_duration_us=simulation_duration_us,
        trajectory_duration_us=trajectory_duration_us,
    )


def resample_record(record: EgoMotionRecord, dt_us: int) -> EgoMotionRecord:
    """Resample a record to a uniform timestamp grid."""
    record.validate()
    timestamps = np.asarray(record.timestamps_us, dtype=np.int64)
    start_us = int(timestamps[0])
    end_us = int(timestamps[-1])
    resampled_timestamps = np.arange(start_us, end_us + 1, dt_us, dtype=np.int64)
    if int(resampled_timestamps[-1]) != end_us:
        resampled_timestamps = np.append(resampled_timestamps, end_us)

    yaw = np.unwrap(np.asarray(record.yaw, dtype=float))
    resampled_yaw = np.interp(resampled_timestamps, timestamps, yaw)
    return make_ego_motion_record(
        name=record.name,
        description=record.description,
        source_id=record.source_id,
        timestamps_us=resampled_timestamps.astype(int).tolist(),
        x=np.interp(
            resampled_timestamps, timestamps, np.asarray(record.x, dtype=float)
        ).tolist(),
        y=np.interp(
            resampled_timestamps, timestamps, np.asarray(record.y, dtype=float)
        ).tolist(),
        yaw=resampled_yaw.tolist(),
        metadata=record.metadata,
    )


def compute_ego_motion_features(record: EgoMotionRecord) -> dict[str, float]:
    """Compute k-disk features from one ego-motion record."""
    record.validate()
    timestamps = np.asarray(record.timestamps_us, dtype=float) / 1e6
    dt = np.gradient(timestamps)
    x = np.asarray(record.x, dtype=float)
    y = np.asarray(record.y, dtype=float)
    yaw = np.unwrap(np.asarray(record.yaw, dtype=float))
    vx = np.asarray(record.vx, dtype=float)

    dx = np.gradient(x)
    dy = np.gradient(y)
    segment_lengths = np.hypot(np.diff(x), np.diff(y))
    speed = np.hypot(dx / np.maximum(dt, _EPSILON), dy / np.maximum(dt, _EPSILON))
    accel = np.gradient(vx) / np.maximum(dt, _EPSILON)
    yaw_rate = np.gradient(yaw) / np.maximum(dt, _EPSILON)
    curvature = yaw_rate / np.maximum(np.abs(vx), 0.5)

    return {
        "path_length_m": float(np.sum(segment_lengths)),
        "lateral_displacement_m": float(y[-1] - y[0]),
        "mean_speed_mps": float(np.mean(speed)),
        "max_speed_mps": float(np.max(speed)),
        "delta_speed_mps": float(vx[-1] - vx[0]),
        "mean_abs_accel_mps2": float(np.mean(np.abs(accel))),
        "max_abs_accel_mps2": float(np.max(np.abs(accel))),
        "yaw_delta_rad": float(yaw[-1] - yaw[0]),
        "max_abs_yaw_rate_radps": float(np.max(np.abs(yaw_rate))),
        "mean_abs_curvature_1pm": float(np.mean(np.abs(curvature))),
        "max_abs_curvature_1pm": float(np.max(np.abs(curvature))),
    }


def select_k_disk_cover(
    records: list[EgoMotionRecord], k: int
) -> tuple[list[int], dict[str, Any]]:
    """Select k records with greedy farthest-point coverage in normalized feature space."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not records:
        raise ValueError("Cannot select from an empty ego-motion catalog.")
    for record in records:
        record.validate()
        if not record.features:
            record.features = compute_ego_motion_features(record)

    feature_names = list(_DEFAULT_FEATURE_NAMES)
    feature_matrix = np.asarray(
        [[record.features[name] for name in feature_names] for record in records],
        dtype=float,
    )
    feature_mean = np.mean(feature_matrix, axis=0)
    feature_std = np.std(feature_matrix, axis=0)
    normalized_features = (feature_matrix - feature_mean) / np.maximum(
        feature_std, _EPSILON
    )

    selected = [int(np.argmin(np.linalg.norm(normalized_features, axis=1)))]
    min_distance = np.linalg.norm(
        normalized_features - normalized_features[selected[0]], axis=1
    )
    while len(selected) < min(k, len(records)):
        candidate_scores = min_distance.copy()
        candidate_scores[selected] = -np.inf
        next_index = int(np.argmax(candidate_scores))
        selected.append(next_index)
        distance_to_next = np.linalg.norm(
            normalized_features - normalized_features[next_index],
            axis=1,
        )
        min_distance = np.minimum(min_distance, distance_to_next)

    metadata = {
        "method": "greedy_k_disk",
        "k": k,
        "feature_names": feature_names,
        "feature_mean": feature_mean.astype(float).tolist(),
        "feature_std": feature_std.astype(float).tolist(),
        "covering_radius": float(np.max(min_distance)),
        "mean_covering_distance": float(np.mean(min_distance)),
        "max_distance_record": records[int(np.argmax(min_distance))].name,
    }
    return selected, metadata


def _yaw_from_xy(timestamps_us: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    timestamps_s = timestamps_us.astype(float) / 1e6
    dt = np.gradient(timestamps_s)
    dx_dt = np.gradient(x) / np.maximum(dt, _EPSILON)
    dy_dt = np.gradient(y) / np.maximum(dt, _EPSILON)
    yaw = np.arctan2(dy_dt, dx_dt)
    return np.unwrap(yaw)


def _vx_from_pose(
    timestamps_us: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    timestamps_s = timestamps_us.astype(float) / 1e6
    dt = np.gradient(timestamps_s)
    dx_dt = np.gradient(x) / np.maximum(dt, _EPSILON)
    dy_dt = np.gradient(y) / np.maximum(dt, _EPSILON)
    return dx_dt * np.cos(yaw) + dy_dt * np.sin(yaw)
