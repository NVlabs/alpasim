# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import argparse
import json
import math
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import grpc
import yaml
from alpasim_grpc.v1 import interactive_runtime_pb2, interactive_runtime_pb2_grpc

from .map_provider import SceneMapProvider, default_usdz_glob


@dataclass(frozen=True)
class SensorView:
    sensor_id: str
    logical_id: str
    nominal_width: int
    nominal_height: int
    frame_encoding: str = "svg"


@dataclass
class ActorView:
    actor_id: str
    x: float
    y: float
    heading_deg: float = 0.0
    speed_mps: float = 0.0


@dataclass(frozen=True)
class CandidateView:
    candidate_id: str
    backend_id: str
    status: str
    selected: bool
    error: str = ""


@dataclass(frozen=True)
class DecisionView:
    step_id: int
    input_snapshot_id: str
    selected_candidate_id: str | None
    candidates: list[CandidateView]
    arbitration_reason: str = ""


@dataclass(frozen=True)
class CheckpointView:
    checkpoint_id: str
    tick_id: int
    sim_time_us: int
    status: str = "PAUSED"
    restore_supported: bool = True
    unsupported_backend_ids: list[str] = field(default_factory=list)


@dataclass
class SessionSnapshotView:
    interactive_session_id: str
    tick_id: int
    sim_time_us: int
    ego: ActorView
    actors: list[ActorView]
    frame_refs: list[dict[str, object]]
    latest_decision: DecisionView | None = None


@dataclass
class SessionStateView:
    interactive_session_id: str
    rollout_uuid: str
    scene_id: str
    status: str
    current_tick_id: int
    current_sim_time_us: int
    latest_snapshot: SessionSnapshotView | None
    latest_decision: DecisionView | None = None
    active_backend_ids: list[str] = field(default_factory=list)
    error: str = ""


class InteractiveApiAdapter:
    """HTTP gateway abstraction over the future interactive runtime API."""

    def create_session(self, scene_id: str) -> SessionStateView:
        raise NotImplementedError

    def list_sessions(self) -> list[SessionStateView]:
        raise NotImplementedError

    def get_state(self, session_id: str) -> SessionStateView:
        raise NotImplementedError

    def list_sensors(self, session_id: str) -> list[SensorView]:
        raise NotImplementedError

    def list_candidates(self, session_id: str) -> list[CandidateView]:
        raise NotImplementedError

    def list_checkpoints(self, session_id: str) -> list[CheckpointView]:
        raise NotImplementedError

    def list_all_sessions(self) -> list[str]:
        raise NotImplementedError

    def step_session(self, session_id: str, num_steps: int) -> SessionStateView:
        raise NotImplementedError

    def start_session(self, session_id: str) -> SessionStateView:
        raise NotImplementedError

    def pause_session(self, session_id: str) -> SessionStateView:
        raise NotImplementedError

    def resume_session(self, session_id: str) -> SessionStateView:
        raise NotImplementedError

    def set_active_backends(self, session_id: str, backend_ids: list[str]) -> SessionStateView:
        raise NotImplementedError

    def recompute_candidate(self, session_id: str, backend_id: str) -> SessionStateView:
        raise NotImplementedError

    def select_candidate(self, session_id: str, candidate_id: str) -> SessionStateView:
        raise NotImplementedError

    def restore_checkpoint(self, session_id: str, checkpoint_id: str) -> SessionStateView:
        raise NotImplementedError

    def get_frame_payload(
        self,
        session_id: str,
        sensor_id: str,
        tick_id: int,
    ) -> tuple[bytes, str]:
        raise NotImplementedError

    def close(self) -> None:
        return None


def _yaw_deg_from_pose(pose) -> float:
    quat = pose.quat
    siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _speed_mps_from_dynamics(dynamics) -> float:
    velocity = dynamics.linear_velocity
    return math.sqrt(
        float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
    )


@dataclass
class _MockSession:
    scene_id: str
    interactive_session_id: str
    rollout_uuid: str
    status: str = "CREATED"
    current_tick_id: int = 0
    current_sim_time_us: int = 0
    sensors: list[SensorView] = field(default_factory=list)
    ego_x: float = 0.0
    ego_y: float = 0.0
    latest_snapshot: SessionSnapshotView | None = None
    active_backend_ids: list[str] = field(default_factory=lambda: ["vla_default"])
    available_backend_ids: list[str] = field(
        default_factory=lambda: ["vla_default", "vla_shadow"]
    )
    checkpoints: list[CheckpointView] = field(default_factory=list)


class MockInteractiveApiAdapter(InteractiveApiAdapter):
    """In-memory adapter used until the real interactive runtime is wired in."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _MockSession] = {}

    def create_session(self, scene_id: str) -> SessionStateView:
        with self._lock:
            session_id = uuid.uuid4().hex[:12]
            sensors = [
                SensorView(
                    sensor_id="camera_front_wide_120fov",
                    logical_id="camera_front_wide_120fov",
                    nominal_width=1280,
                    nominal_height=720,
                ),
                SensorView(
                    sensor_id="camera_front_left_120fov",
                    logical_id="camera_front_left_120fov",
                    nominal_width=1280,
                    nominal_height=720,
                ),
                SensorView(
                    sensor_id="camera_front_right_120fov",
                    logical_id="camera_front_right_120fov",
                    nominal_width=1280,
                    nominal_height=720,
                ),
            ]
            session = _MockSession(
                scene_id=scene_id,
                interactive_session_id=session_id,
                rollout_uuid=f"mock-{session_id}",
                sensors=sensors,
            )
            session.latest_snapshot = self._build_snapshot(session)
            self._record_checkpoint(session)
            self._sessions[session_id] = session
            return self._build_state(session)

    def list_sessions(self) -> list[SessionStateView]:
        with self._lock:
            return [self._build_state(session) for session in self._sessions.values()]

    def get_state(self, session_id: str) -> SessionStateView:
        with self._lock:
            return self._build_state(self._require_session(session_id))

    def list_sensors(self, session_id: str) -> list[SensorView]:
        with self._lock:
            return list(self._require_session(session_id).sensors)

    def list_candidates(self, session_id: str) -> list[CandidateView]:
        with self._lock:
            session = self._require_session(session_id)
            decision = self._current_decision(session)
            return list(decision.candidates if decision is not None else [])

    def list_checkpoints(self, session_id: str) -> list[CheckpointView]:
        with self._lock:
            session = self._require_session(session_id)
            return list(session.checkpoints)

    def list_all_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def step_session(self, session_id: str, num_steps: int) -> SessionStateView:
        with self._lock:
            session = self._require_session(session_id)
            steps = max(num_steps, 1)
            session.status = "PAUSED"
            for _ in range(steps):
                session.current_tick_id += 1
                session.current_sim_time_us += 100_000
                session.ego_x += 2.5
                session.ego_y += 0.45
            session.latest_snapshot = self._build_snapshot(session)
            self._record_checkpoint(session)
            return self._build_state(session)

    def start_session(self, session_id: str) -> SessionStateView:
        with self._lock:
            session = self._require_session(session_id)
            session.status = "RUNNING"
            return self._build_state(session)

    def pause_session(self, session_id: str) -> SessionStateView:
        with self._lock:
            session = self._require_session(session_id)
            session.status = "PAUSED"
            return self._build_state(session)

    def resume_session(self, session_id: str) -> SessionStateView:
        with self._lock:
            session = self._require_session(session_id)
            session.status = "RUNNING"
            session.current_tick_id += 1
            session.current_sim_time_us += 100_000
            session.ego_x += 2.0
            session.ego_y += 0.25
            session.latest_snapshot = self._build_snapshot(session)
            self._record_checkpoint(session)
            return self._build_state(session)

    def set_active_backends(self, session_id: str, backend_ids: list[str]) -> SessionStateView:
        with self._lock:
            session = self._require_session(session_id)
            selected = list(backend_ids) if backend_ids else list(session.available_backend_ids)
            unknown = [backend_id for backend_id in selected if backend_id not in session.available_backend_ids]
            if unknown:
                raise KeyError(f"Unknown backend ids: {unknown}")
            session.active_backend_ids = selected
            session.latest_snapshot = self._build_snapshot(session)
            self._record_checkpoint(session)
            return self._build_state(session)

    def recompute_candidate(self, session_id: str, backend_id: str) -> SessionStateView:
        with self._lock:
            session = self._require_session(session_id)
            if backend_id not in session.available_backend_ids:
                raise KeyError(f"Unknown backend_id: {backend_id}")
            session.status = "PAUSED"
            decision = self._current_decision(session)
            if decision is None:
                raise RuntimeError("No committed decision bundle available for recompute")
            updated = []
            recompute_suffix = 1
            for candidate in decision.candidates:
                if candidate.backend_id == backend_id and not candidate.selected:
                    updated.append(CandidateView(
                        candidate_id=candidate.candidate_id,
                        backend_id=candidate.backend_id,
                        status="STALE",
                        selected=False,
                        error=candidate.error,
                    ))
                    recompute_suffix += 1
                else:
                    updated.append(candidate)
            updated.append(
                CandidateView(
                    candidate_id=f"{backend_id}:step{decision.step_id}:r{recompute_suffix}",
                    backend_id=backend_id,
                    status="READY",
                    selected=False,
                )
            )
            new_decision = DecisionView(
                step_id=decision.step_id,
                input_snapshot_id=decision.input_snapshot_id,
                selected_candidate_id=decision.selected_candidate_id,
                candidates=updated,
                arbitration_reason="recomputed",
            )
            session.latest_snapshot = self._replace_snapshot_decision(session, new_decision)
            self._record_checkpoint(session)
            return self._build_state(session)

    def select_candidate(self, session_id: str, candidate_id: str) -> SessionStateView:
        with self._lock:
            session = self._require_session(session_id)
            session.status = "PAUSED"
            decision = self._current_decision(session)
            if decision is None:
                raise RuntimeError("No committed decision bundle available for selection")
            found = False
            updated = []
            for candidate in decision.candidates:
                if candidate.candidate_id == candidate_id:
                    found = True
                    updated.append(CandidateView(
                        candidate_id=candidate.candidate_id,
                        backend_id=candidate.backend_id,
                        status="SELECTED",
                        selected=True,
                        error=candidate.error,
                    ))
                elif candidate.selected:
                    updated.append(CandidateView(
                        candidate_id=candidate.candidate_id,
                        backend_id=candidate.backend_id,
                        status="REJECTED",
                        selected=False,
                        error=candidate.error,
                    ))
                else:
                    updated.append(candidate)
            if not found:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")
            new_decision = DecisionView(
                step_id=decision.step_id,
                input_snapshot_id=decision.input_snapshot_id,
                selected_candidate_id=candidate_id,
                candidates=updated,
                arbitration_reason="manual_selection",
            )
            session.latest_snapshot = self._replace_snapshot_decision(session, new_decision)
            self._record_checkpoint(session)
            return self._build_state(session)

    def restore_checkpoint(self, session_id: str, checkpoint_id: str) -> SessionStateView:
        with self._lock:
            session = self._require_session(session_id)
            checkpoint = next(
                (item for item in session.checkpoints if item.checkpoint_id == checkpoint_id),
                None,
            )
            if checkpoint is None:
                raise KeyError(f"Unknown checkpoint_id: {checkpoint_id}")
            session.status = "PAUSED"
            session.current_tick_id = checkpoint.tick_id
            session.current_sim_time_us = checkpoint.sim_time_us
            session.ego_x = checkpoint.tick_id * 2.5
            session.ego_y = checkpoint.tick_id * 0.45
            session.latest_snapshot = self._build_snapshot(session)
            return self._build_state(session)

    def get_frame_payload(
        self,
        session_id: str,
        sensor_id: str,
        tick_id: int,
    ) -> tuple[bytes, str]:
        with self._lock:
            session = self._require_session(session_id)
            self._require_sensor(session.sensors, sensor_id)
            status = session.status
            sim_seconds = tick_id * 0.1
            color = {
                "RUNNING": "#2d8f5b",
                "PAUSED": "#b17b17",
                "CREATED": "#4d6aa5",
                "FAILED": "#9f2f2f",
            }.get(status, "#4d6aa5")
            svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1280 720\">
  <defs>
    <linearGradient id=\"bg\" x1=\"0\" x2=\"1\" y1=\"0\" y2=\"1\">
      <stop offset=\"0%\" stop-color=\"#0f1724\"/>
      <stop offset=\"100%\" stop-color=\"#1c3148\"/>
    </linearGradient>
  </defs>
  <rect width=\"1280\" height=\"720\" fill=\"url(#bg)\"/>
  <circle cx=\"{220 + (tick_id % 20) * 36}\" cy=\"455\" r=\"48\" fill=\"#4ec0e9\" opacity=\"0.9\"/>
  <rect x=\"0\" y=\"500\" width=\"1280\" height=\"220\" fill=\"#22313f\"/>
  <rect x=\"120\" y=\"540\" width=\"1040\" height=\"16\" rx=\"8\" fill=\"#6f7f8e\"/>
  <rect x=\"420\" y=\"280\" width=\"420\" height=\"70\" rx=\"16\" fill=\"#0c1118\" opacity=\"0.7\"/>
  <text x=\"640\" y=\"325\" text-anchor=\"middle\" fill=\"#f2f5f7\" font-size=\"34\" font-family=\"Verdana\">{sensor_id}</text>
  <text x=\"92\" y=\"96\" fill=\"#f2f5f7\" font-size=\"48\" font-family=\"Verdana\">Tick {tick_id}</text>
  <text x=\"92\" y=\"146\" fill=\"#c9d5df\" font-size=\"28\" font-family=\"Verdana\">Session {session_id}</text>
  <text x=\"92\" y=\"190\" fill=\"#c9d5df\" font-size=\"26\" font-family=\"Verdana\">Scene {session.scene_id}</text>
  <text x=\"92\" y=\"234\" fill=\"{color}\" font-size=\"26\" font-family=\"Verdana\">Status {status}</text>
  <text x=\"92\" y=\"278\" fill=\"#c9d5df\" font-size=\"26\" font-family=\"Verdana\">Sim time {sim_seconds:.1f}s</text>
</svg>"""
            return (svg.encode("utf-8"), "image/svg+xml; charset=utf-8")

    def _build_snapshot(self, session: _MockSession) -> SessionSnapshotView:
        ego = ActorView(
            actor_id="EGO",
            x=session.ego_x,
            y=session.ego_y,
            heading_deg=10.0,
            speed_mps=5.6,
        )
        actors = [
            ActorView(
                actor_id="veh_a",
                x=session.ego_x + 16.0,
                y=session.ego_y + 2.5,
                heading_deg=0.0,
                speed_mps=4.2,
            ),
            ActorView(
                actor_id="veh_b",
                x=session.ego_x - 12.5,
                y=session.ego_y - 3.0,
                heading_deg=180.0,
                speed_mps=1.5,
            ),
            ActorView(
                actor_id="ped_a",
                x=session.ego_x + 5.0,
                y=session.ego_y + 9.0,
                heading_deg=-90.0,
                speed_mps=0.9,
            ),
        ]
        frame_refs = [
            {
                "sensor_id": sensor.sensor_id,
                "tick_id": session.current_tick_id,
                "frame_start_us": max(session.current_sim_time_us - 33_000, 0),
                "frame_end_us": session.current_sim_time_us,
                "frame_encoding": sensor.frame_encoding,
            }
            for sensor in session.sensors
        ]
        return SessionSnapshotView(
            interactive_session_id=session.interactive_session_id,
            tick_id=session.current_tick_id,
            sim_time_us=session.current_sim_time_us,
            ego=ego,
            actors=actors,
            frame_refs=frame_refs,
            latest_decision=self._build_decision(session),
        )

    def _build_state(self, session: _MockSession) -> SessionStateView:
        latest_decision = self._current_decision(session)
        return SessionStateView(
            interactive_session_id=session.interactive_session_id,
            rollout_uuid=session.rollout_uuid,
            scene_id=session.scene_id,
            status=session.status,
            current_tick_id=session.current_tick_id,
            current_sim_time_us=session.current_sim_time_us,
            latest_snapshot=session.latest_snapshot,
            latest_decision=latest_decision,
            active_backend_ids=list(session.active_backend_ids),
        )

    def _build_decision(self, session: _MockSession) -> DecisionView:
        selected_backend_id = session.active_backend_ids[0] if session.active_backend_ids else session.available_backend_ids[0]
        candidates = []
        for index, backend_id in enumerate(session.active_backend_ids or session.available_backend_ids):
            is_selected = backend_id == selected_backend_id
            status = "SELECTED" if is_selected else "READY"
            candidates.append(
                CandidateView(
                    candidate_id=f"{backend_id}:step{session.current_tick_id}:r0",
                    backend_id=backend_id,
                    status=status,
                    selected=is_selected,
                )
            )
        if not candidates:
            candidates = [
                CandidateView(
                    candidate_id=f"vla_default:step{session.current_tick_id}:r0",
                    backend_id="vla_default",
                    status="SELECTED",
                    selected=True,
                )
            ]
            selected_backend_id = "vla_default"
        return DecisionView(
            step_id=session.current_tick_id,
            input_snapshot_id=f"mock-step-{session.current_tick_id}",
            selected_candidate_id=f"{selected_backend_id}:step{session.current_tick_id}:r0",
            candidates=candidates,
            arbitration_reason="priority_default",
        )

    def _current_decision(self, session: _MockSession) -> DecisionView | None:
        return session.latest_snapshot.latest_decision if session.latest_snapshot is not None else None

    def _replace_snapshot_decision(self, session: _MockSession, decision: DecisionView) -> SessionSnapshotView:
        snapshot = session.latest_snapshot or self._build_snapshot(session)
        return SessionSnapshotView(
            interactive_session_id=snapshot.interactive_session_id,
            tick_id=snapshot.tick_id,
            sim_time_us=snapshot.sim_time_us,
            ego=snapshot.ego,
            actors=snapshot.actors,
            frame_refs=snapshot.frame_refs,
            latest_decision=decision,
        )

    def _record_checkpoint(self, session: _MockSession) -> None:
        checkpoint = CheckpointView(
            checkpoint_id=f"ckpt-{session.current_tick_id:04d}-{len(session.checkpoints)}",
            tick_id=session.current_tick_id,
            sim_time_us=session.current_sim_time_us,
            status="PAUSED" if session.status != "FAILED" else "FAILED",
            restore_supported=True,
            unsupported_backend_ids=[],
        )
        session.checkpoints = [item for item in session.checkpoints if item.tick_id != checkpoint.tick_id]
        session.checkpoints.append(checkpoint)
        session.checkpoints = session.checkpoints[-16:]

    def _require_session(self, session_id: str) -> _MockSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        return self._sessions[session_id]

    @staticmethod
    def _require_sensor(sensors: Iterable[SensorView], sensor_id: str) -> SensorView:
        for sensor in sensors:
            if sensor.sensor_id == sensor_id:
                return sensor
        raise KeyError(f"Unknown sensor_id: {sensor_id}")


class DebuggerRequestHandler(BaseHTTPRequestHandler):
    server: "InteractiveDebuggerServer"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if parsed.path == "/assets/styles.css":
            return self._serve_static("styles.css", "text/css; charset=utf-8")
        if parsed.path == "/assets/app.js":
            return self._serve_static("app.js", "application/javascript; charset=utf-8")
        if parsed.path == "/api/scenes":
            return self._handle_list_scenes()
        if parsed.path == "/api/sessions":
            return self._handle_list_sessions()
        if parsed.path == "/api/session/state":
            return self._handle_get_state(parsed)
        if parsed.path == "/api/sensors":
            return self._handle_list_sensors(parsed)
        if parsed.path == "/api/candidates":
            return self._handle_list_candidates(parsed)
        if parsed.path == "/api/checkpoints":
            return self._handle_list_checkpoints(parsed)
        if parsed.path == "/api/sessions":
            return self._handle_list_all_sessions()
        if parsed.path == "/api/map":
            return self._handle_get_map(parsed)
        if parsed.path == "/api/frame":
            return self._handle_get_frame(parsed)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/session/create":
            return self._handle_create_session()
        if parsed.path == "/api/session/start":
            return self._handle_mutation(parsed.path)
        if parsed.path == "/api/session/pause":
            return self._handle_mutation(parsed.path)
        if parsed.path == "/api/session/resume":
            return self._handle_mutation(parsed.path)
        if parsed.path == "/api/session/step":
            return self._handle_mutation(parsed.path)
        if parsed.path == "/api/backends/active":
            return self._handle_mutation(parsed.path)
        if parsed.path == "/api/candidates/recompute":
            return self._handle_mutation(parsed.path)
        if parsed.path == "/api/candidates/select":
            return self._handle_mutation(parsed.path)
        if parsed.path == "/api/checkpoints/restore":
            return self._handle_mutation(parsed.path)
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        del format, args

    def _handle_create_session(self) -> None:
        try:
            payload = self._read_json_body()
            scene_id = str(payload.get("scene_id") or "clipgt-demo")
            state = self.server.adapter.create_session(scene_id=scene_id)
            self._write_json(self._state_to_dict(state), status=HTTPStatus.CREATED)
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_get_state(self, parsed) -> None:
        try:
            params = parse_qs(parsed.query)
            session_id = self._require_query_param(params, "session_id")
            state = self.server.adapter.get_state(session_id)
            self._write_json(self._state_to_dict(state))
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_list_sessions(self) -> None:
        try:
            sessions = self.server.adapter.list_sessions()
            self._write_json(
                {"sessions": [self._session_summary_to_dict(item) for item in sessions]}
            )
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_list_scenes(self) -> None:
        try:
            scene_ids = list(self.server.scene_ids)
            self._write_json(
                {
                    "scenes": [
                        {
                            "scene_id": scene_id,
                            "label": scene_id,
                        }
                        for scene_id in scene_ids
                    ]
                }
            )
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_list_sensors(self, parsed) -> None:
        try:
            params = parse_qs(parsed.query)
            session_id = self._require_query_param(params, "session_id")
            sensors = self.server.adapter.list_sensors(session_id)
            self._write_json({"sensors": [sensor.__dict__ for sensor in sensors]})
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_list_candidates(self, parsed) -> None:
        try:
            params = parse_qs(parsed.query)
            session_id = self._require_query_param(params, "session_id")
            candidates = self.server.adapter.list_candidates(session_id)
            self._write_json({"candidates": [self._candidate_to_dict(item) for item in candidates]})
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_list_checkpoints(self, parsed) -> None:
        try:
            params = parse_qs(parsed.query)
            session_id = self._require_query_param(params, "session_id")
            checkpoints = self.server.adapter.list_checkpoints(session_id)
            self._write_json({"checkpoints": [self._checkpoint_to_dict(item) for item in checkpoints]})
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_list_all_sessions(self) -> None:
        try:
            session_ids = self.server.adapter.list_all_sessions()
            self._write_json({"session_ids": session_ids})
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_get_map(self, parsed) -> None:
        try:
            params = parse_qs(parsed.query)
            scene_id = self._require_query_param(params, "scene_id")
            self._write_json(self.server.map_provider.get_scene_map(scene_id).as_dict())
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_get_frame(self, parsed) -> None:
        try:
            params = parse_qs(parsed.query)
            session_id = self._require_query_param(params, "session_id")
            sensor_id = self._require_query_param(params, "sensor_id")
            tick_id = int(self._require_query_param(params, "tick_id"))
            body, content_type = self.server.adapter.get_frame_payload(
                session_id, sensor_id, tick_id
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _handle_mutation(self, path: str) -> None:
        try:
            payload = self._read_json_body()
            session_id = str(payload["session_id"])
            if path.endswith("/start"):
                state = self.server.adapter.start_session(session_id)
            elif path.endswith("/pause"):
                state = self.server.adapter.pause_session(session_id)
            elif path.endswith("/resume"):
                state = self.server.adapter.resume_session(session_id)
            elif path.endswith("/step"):
                state = self.server.adapter.step_session(
                    session_id=session_id,
                    num_steps=int(payload.get("num_steps", 1)),
                )
            elif path.endswith("/active"):
                backend_ids = payload.get("backend_ids") or []
                state = self.server.adapter.set_active_backends(
                    session_id=session_id,
                    backend_ids=[str(item) for item in backend_ids],
                )
            elif path.endswith("/recompute"):
                state = self.server.adapter.recompute_candidate(
                    session_id=session_id,
                    backend_id=str(payload["backend_id"]),
                )
            elif path.endswith("/select"):
                state = self.server.adapter.select_candidate(
                    session_id=session_id,
                    candidate_id=str(payload["candidate_id"]),
                )
            elif path.endswith("/restore"):
                state = self.server.adapter.restore_checkpoint(
                    session_id=session_id,
                    checkpoint_id=str(payload["checkpoint_id"]),
                )
            else:
                raise AssertionError(f"Unhandled mutation path: {path}")
            self._write_json(self._state_to_dict(state))
        except Exception as exc:  # noqa: BLE001
            self._write_exception(exc)

    def _serve_static(self, filename: str, content_type: str) -> None:
        static_root = Path(__file__).with_name("static")
        path = static_root / filename
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_exception(self, exc: Exception) -> None:
        status = HTTPStatus.INTERNAL_SERVER_ERROR
        payload: dict[str, object] = {"error": str(exc)}
        if isinstance(exc, grpc.RpcError):
            payload["grpc_status"] = exc.code().name if exc.code() is not None else None
            payload["grpc_details"] = exc.details()
            if exc.code() == grpc.StatusCode.NOT_FOUND:
                status = HTTPStatus.NOT_FOUND
            elif exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                status = HTTPStatus.TOO_MANY_REQUESTS
            elif exc.code() == grpc.StatusCode.INVALID_ARGUMENT:
                status = HTTPStatus.BAD_REQUEST
            elif exc.code() == grpc.StatusCode.FAILED_PRECONDITION:
                status = HTTPStatus.PRECONDITION_FAILED
        elif isinstance(exc, KeyError):
            status = HTTPStatus.BAD_REQUEST
        elif isinstance(exc, RuntimeError):
            status = HTTPStatus.PRECONDITION_FAILED
        self._write_json(payload, status=status)

    @staticmethod
    def _require_query_param(params: dict[str, list[str]], key: str) -> str:
        values = params.get(key)
        if not values:
            raise KeyError(f"Missing query parameter: {key}")
        return values[0]

    def _state_to_dict(self, state: SessionStateView) -> dict[str, object]:
        latest_snapshot = self._snapshot_to_dict(state.latest_snapshot)
        return {
            "interactive_session_id": state.interactive_session_id,
            "rollout_uuid": state.rollout_uuid,
            "scene_id": state.scene_id,
            "status": state.status,
            "current_tick_id": state.current_tick_id,
            "current_sim_time_us": state.current_sim_time_us,
            "latest_snapshot": latest_snapshot,
            "latest_decision": self._decision_to_dict(state.latest_decision),
            "active_backend_ids": state.active_backend_ids,
            "error": state.error,
        }

    def _snapshot_to_dict(self, snapshot: SessionSnapshotView | None) -> dict[str, object] | None:
        if snapshot is None:
            return None
        return {
            "interactive_session_id": snapshot.interactive_session_id,
            "tick_id": snapshot.tick_id,
            "sim_time_us": snapshot.sim_time_us,
            "ego": snapshot.ego.__dict__,
            "actors": [actor.__dict__ for actor in snapshot.actors],
            "frame_refs": snapshot.frame_refs,
            "latest_decision": self._decision_to_dict(snapshot.latest_decision),
        }

    @staticmethod
    def _candidate_to_dict(candidate: CandidateView) -> dict[str, object]:
        return {
            "candidate_id": candidate.candidate_id,
            "backend_id": candidate.backend_id,
            "status": candidate.status,
            "selected": candidate.selected,
            "error": candidate.error,
        }

    def _decision_to_dict(self, decision: DecisionView | None) -> dict[str, object] | None:
        if decision is None:
            return None
        return {
            "step_id": decision.step_id,
            "input_snapshot_id": decision.input_snapshot_id,
            "selected_candidate_id": decision.selected_candidate_id,
            "arbitration_reason": decision.arbitration_reason,
            "candidates": [self._candidate_to_dict(item) for item in decision.candidates],
        }

    @staticmethod
    def _checkpoint_to_dict(checkpoint: CheckpointView) -> dict[str, object]:
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "tick_id": checkpoint.tick_id,
            "sim_time_us": checkpoint.sim_time_us,
            "status": checkpoint.status,
            "restore_supported": checkpoint.restore_supported,
            "unsupported_backend_ids": checkpoint.unsupported_backend_ids,
        }

    @staticmethod
    def _session_summary_to_dict(state: SessionStateView) -> dict[str, object]:
        return {
            "interactive_session_id": state.interactive_session_id,
            "rollout_uuid": state.rollout_uuid,
            "scene_id": state.scene_id,
            "status": state.status,
            "current_tick_id": state.current_tick_id,
            "current_sim_time_us": state.current_sim_time_us,
        }


class InteractiveDebuggerServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        adapter: InteractiveApiAdapter,
        map_provider: SceneMapProvider,
        scene_ids: list[str] | None = None,
    ):
        super().__init__(server_address, DebuggerRequestHandler)
        self.adapter = adapter
        self.map_provider = map_provider
        self.scene_ids = scene_ids or []


class GrpcInteractiveApiAdapter(InteractiveApiAdapter):
    """HTTP gateway adapter backed by the runtime interactive gRPC service."""

    def __init__(self, runtime_address: str) -> None:
        self._channel = grpc.insecure_channel(runtime_address)
        self._stub = interactive_runtime_pb2_grpc.InteractiveRuntimeServiceStub(
            self._channel
        )

    def create_session(self, scene_id: str) -> SessionStateView:
        response = self._stub.CreateSession(
            interactive_runtime_pb2.CreateSessionRequest(
                scene_id=scene_id,
                start_paused=True,
                max_retained_ticks=64,
            )
        )
        return self._state_from_proto(response.initial_state)

    def list_sessions(self) -> list[SessionStateView]:
        response = self._stub.ListSessions(interactive_runtime_pb2.ListSessionsRequest())
        return [self._state_from_proto(item) for item in response.sessions]

    def get_state(self, session_id: str) -> SessionStateView:
        response = self._stub.GetSessionState(
            interactive_runtime_pb2.GetSessionStateRequest(
                interactive_session_id=session_id
            )
        )
        return self._state_from_proto(response)

    def list_sensors(self, session_id: str) -> list[SensorView]:
        response = self._stub.ListSensors(
            interactive_runtime_pb2.ListSensorsRequest(
                interactive_session_id=session_id
            )
        )
        return [self._sensor_from_proto(sensor) for sensor in response.sensors]

    def list_candidates(self, session_id: str) -> list[CandidateView]:
        response = self._stub.ListCandidates(
            interactive_runtime_pb2.ListCandidatesRequest(
                interactive_session_id=session_id
            )
        )
        return [self._candidate_from_proto(item) for item in response.candidates]

    def list_checkpoints(self, session_id: str) -> list[CheckpointView]:
        response = self._stub.ListCheckpoints(
            interactive_runtime_pb2.ListCheckpointsRequest(
                interactive_session_id=session_id
            )
        )
        return [self._checkpoint_from_proto(item) for item in response.checkpoints]

    def list_all_sessions(self) -> list[str]:
        # gRPC backend may not support listing all sessions yet
        return []

    def step_session(self, session_id: str, num_steps: int) -> SessionStateView:
        response = self._stub.StepSession(
            interactive_runtime_pb2.StepSessionRequest(
                interactive_session_id=session_id,
                num_steps=max(int(num_steps), 1),
            )
        )
        return self._state_from_proto(response.state)

    def start_session(self, session_id: str) -> SessionStateView:
        response = self._stub.StartSession(
            interactive_runtime_pb2.StartSessionRequest(
                interactive_session_id=session_id
            )
        )
        return self._state_from_proto(response)

    def pause_session(self, session_id: str) -> SessionStateView:
        response = self._stub.PauseSession(
            interactive_runtime_pb2.PauseSessionRequest(
                interactive_session_id=session_id
            )
        )
        return self._state_from_proto(response)

    def resume_session(self, session_id: str) -> SessionStateView:
        response = self._stub.ResumeSession(
            interactive_runtime_pb2.ResumeSessionRequest(
                interactive_session_id=session_id
            )
        )
        return self._state_from_proto(response)

    def set_active_backends(self, session_id: str, backend_ids: list[str]) -> SessionStateView:
        response = self._stub.SetActiveBackends(
            interactive_runtime_pb2.SetActiveBackendsRequest(
                interactive_session_id=session_id,
                backend_ids=backend_ids,
            )
        )
        return self._state_from_proto(response)

    def recompute_candidate(self, session_id: str, backend_id: str) -> SessionStateView:
        response = self._stub.RecomputeCandidate(
            interactive_runtime_pb2.RecomputeCandidateRequest(
                interactive_session_id=session_id,
                backend_id=backend_id,
            )
        )
        return self._state_from_proto(response)

    def select_candidate(self, session_id: str, candidate_id: str) -> SessionStateView:
        response = self._stub.SelectCandidate(
            interactive_runtime_pb2.SelectCandidateRequest(
                interactive_session_id=session_id,
                candidate_id=candidate_id,
            )
        )
        return self._state_from_proto(response)

    def restore_checkpoint(self, session_id: str, checkpoint_id: str) -> SessionStateView:
        response = self._stub.RestoreCheckpoint(
            interactive_runtime_pb2.RestoreCheckpointRequest(
                interactive_session_id=session_id,
                checkpoint_id=checkpoint_id,
            )
        )
        return self._state_from_proto(response)

    def get_frame_payload(
        self,
        session_id: str,
        sensor_id: str,
        tick_id: int,
    ) -> tuple[bytes, str]:
        response = self._stub.GetFrame(
            interactive_runtime_pb2.GetFrameRequest(
                interactive_session_id=session_id,
                sensor_id=sensor_id,
                tick_id=tick_id,
            )
        )
        content_type = "image/jpeg"
        if (
            response.frame_ref.frame_encoding
            == interactive_runtime_pb2.FRAME_ENCODING_PNG
        ):
            content_type = "image/png"
        return (bytes(response.image_bytes), content_type)

    def close(self) -> None:
        self._channel.close()

    @staticmethod
    def _sensor_from_proto(sensor) -> SensorView:
        return SensorView(
            sensor_id=sensor.sensor_id,
            logical_id=sensor.logical_id,
            nominal_width=sensor.nominal_width,
            nominal_height=sensor.nominal_height,
            frame_encoding=interactive_runtime_pb2.FrameEncoding.Name(
                sensor.frame_encoding
            ).removeprefix("FRAME_ENCODING_").lower(),
        )

    @staticmethod
    def _candidate_from_proto(candidate) -> CandidateView:
        return CandidateView(
            candidate_id=candidate.candidate_id,
            backend_id=candidate.backend_id,
            status=candidate.status,
            selected=bool(candidate.selected),
            error=candidate.error,
        )

    @classmethod
    def _decision_from_proto(cls, decision) -> DecisionView:
        return DecisionView(
            step_id=int(decision.step_id),
            input_snapshot_id=decision.input_snapshot_id,
            selected_candidate_id=decision.selected_candidate_id or None,
            candidates=[cls._candidate_from_proto(item) for item in decision.candidates],
            arbitration_reason=decision.arbitration_reason,
        )

    @staticmethod
    def _checkpoint_from_proto(checkpoint) -> CheckpointView:
        return CheckpointView(
            checkpoint_id=checkpoint.checkpoint_id,
            tick_id=int(checkpoint.tick_id),
            sim_time_us=int(checkpoint.sim_time_us),
            status=interactive_runtime_pb2.SessionStatus.Name(checkpoint.status).removeprefix(
                "SESSION_STATUS_"
            ),
            restore_supported=bool(checkpoint.restore_supported),
            unsupported_backend_ids=list(checkpoint.unsupported_backend_ids),
        )

    @classmethod
    def _state_from_proto(cls, state) -> SessionStateView:
        latest_snapshot = (
            cls._snapshot_from_proto(state.latest_snapshot)
            if state.HasField("latest_snapshot")
            else None
        )
        latest_decision = (
            cls._decision_from_proto(state.latest_decision)
            if hasattr(state, "HasField") and state.HasField("latest_decision")
            else None
        )
        return SessionStateView(
            interactive_session_id=state.interactive_session_id,
            rollout_uuid=state.rollout_uuid,
            scene_id=state.scene_id,
            status=interactive_runtime_pb2.SessionStatus.Name(state.status).removeprefix(
                "SESSION_STATUS_"
            ),
            current_tick_id=int(state.current_tick_id),
            current_sim_time_us=int(state.current_sim_time_us),
            latest_snapshot=latest_snapshot,
            latest_decision=latest_decision,
            active_backend_ids=list(getattr(state, "active_backend_ids", [])),
            error=state.error,
        )

    @classmethod
    def _snapshot_from_proto(cls, snapshot) -> SessionSnapshotView:
        latest_decision = (
            cls._decision_from_proto(snapshot.latest_decision)
            if hasattr(snapshot, "HasField") and snapshot.HasField("latest_decision")
            else None
        )
        return SessionSnapshotView(
            interactive_session_id=snapshot.interactive_session_id,
            tick_id=int(snapshot.tick_id),
            sim_time_us=int(snapshot.sim_time_us),
            ego=ActorView(
                actor_id="EGO",
                x=float(snapshot.ego.pose.vec.x),
                y=float(snapshot.ego.pose.vec.y),
                heading_deg=_yaw_deg_from_pose(snapshot.ego.pose),
                speed_mps=_speed_mps_from_dynamics(snapshot.ego.dynamics),
            ),
            actors=[
                ActorView(
                    actor_id=actor.actor_id,
                    x=float(actor.pose.vec.x),
                    y=float(actor.pose.vec.y),
                    heading_deg=_yaw_deg_from_pose(actor.pose),
                )
                for actor in snapshot.actors
            ],
            frame_refs=[
                {
                    "sensor_id": frame_ref.sensor_id,
                    "tick_id": int(frame_ref.tick_id),
                    "frame_start_us": int(frame_ref.frame_start_us),
                    "frame_end_us": int(frame_ref.frame_end_us),
                    "frame_encoding": interactive_runtime_pb2.FrameEncoding.Name(
                        frame_ref.frame_encoding
                    ).removeprefix("FRAME_ENCODING_"),
                }
                for frame_ref in snapshot.frame_refs
            ],
            latest_decision=latest_decision,
        )


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the interactive web debugger")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--runtime-address",
        default=None,
        help="Interactive runtime gRPC address, for example 127.0.0.1:50051",
    )
    parser.add_argument(
        "--user-config",
        default=None,
        help="Optional runtime user config YAML used to populate the scene dropdown",
    )
    parser.add_argument(
        "--usdz-glob",
        default=default_usdz_glob(),
        help="Glob used to resolve scene artifacts for map rendering",
    )
    return parser


def _load_scene_ids_from_user_config(user_config_path: str | None) -> list[str]:
    if not user_config_path:
        return []
    with open(user_config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    scenes = payload.get("scenes") or []
    scene_ids: list[str] = []
    for item in scenes:
        if not isinstance(item, dict):
            continue
        scene_id = item.get("scene_id")
        if scene_id:
            scene_ids.append(str(scene_id))
    return sorted(set(scene_ids))


def main() -> None:
    args = create_arg_parser().parse_args()
    adapter: InteractiveApiAdapter
    if args.runtime_address:
        adapter = GrpcInteractiveApiAdapter(args.runtime_address)
    else:
        adapter = MockInteractiveApiAdapter()
    map_provider = SceneMapProvider(args.usdz_glob)
    scene_ids = _load_scene_ids_from_user_config(args.user_config)
    if not scene_ids and not args.runtime_address:
        scene_ids = map_provider.list_scene_ids()
    server = InteractiveDebuggerServer(
        (args.host, args.port),
        adapter,
        map_provider,
        scene_ids=scene_ids,
    )
    print(f"Interactive debugger listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        adapter.close()


if __name__ == "__main__":
    main()
