# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensorDescriptorModel:
    sensor_id: str
    logical_id: str
    nominal_width: int
    nominal_height: int
    nominal_frame_interval_us: int
    rig_to_sensor: object
    frame_encoding: str


@dataclass(frozen=True)
class FrameDataModel:
    sensor_id: str
    frame_start_us: int
    frame_end_us: int
    frame_encoding: str
    content_type: str
    image_bytes: bytes


@dataclass(frozen=True)
class FrameRefModel:
    sensor_id: str
    tick_id: int
    frame_start_us: int
    frame_end_us: int
    frame_encoding: str


@dataclass(frozen=True)
class EgoStateModel:
    pose: object
    dynamics: object
    front_steering_angle_rad: float = 0.0


@dataclass(frozen=True)
class ActorStateModel:
    actor_id: str
    pose: object


@dataclass(frozen=True)
class PolylinePointModel:
    x: float
    y: float


@dataclass(frozen=True)
class CandidatePlanModel:
    candidate_id: str
    backend_id: str
    selected: bool
    points: list[PolylinePointModel]


@dataclass(frozen=True)
class CandidateSummaryModel:
    candidate_id: str
    backend_id: str
    status: str
    selected: bool
    error: str = ""


@dataclass(frozen=True)
class DecisionSummaryModel:
    step_id: int
    input_snapshot_id: str
    selected_candidate_id: str | None
    candidates: list[CandidateSummaryModel]
    arbitration_reason: str = ""


@dataclass(frozen=True)
class CheckpointSummaryModel:
    checkpoint_id: str
    tick_id: int
    sim_time_us: int
    status: str = "PAUSED"
    restore_supported: bool = True
    unsupported_backend_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SessionSnapshotModel:
    interactive_session_id: str
    tick_id: int
    sim_time_us: int
    ego: EgoStateModel
    actors: list[ActorStateModel]
    frame_refs: list[FrameRefModel]
    latest_decision: DecisionSummaryModel | None = None
    ego_history: list[PolylinePointModel] = field(default_factory=list)
    selected_plan: list[PolylinePointModel] = field(default_factory=list)
    candidate_plans: list[CandidatePlanModel] = field(default_factory=list)


@dataclass(frozen=True)
class SessionStateModel:
    interactive_session_id: str
    rollout_uuid: str
    scene_id: str
    status: str
    current_tick_id: int
    current_sim_time_us: int
    latest_snapshot: SessionSnapshotModel | None
    latest_decision: DecisionSummaryModel | None = None
    active_backend_ids: list[str] = field(default_factory=list)
    available_backend_ids: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class SessionEventModel:
    state: SessionStateModel | None = None
    snapshot: SessionSnapshotModel | None = None


@dataclass
class SessionUpdateModel:
    state: SessionStateModel
    committed_snapshots: list[SessionSnapshotModel] = field(default_factory=list)
