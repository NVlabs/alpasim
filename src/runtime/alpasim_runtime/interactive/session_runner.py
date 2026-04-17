# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import copy
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

import numpy as np
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_runtime.decision import (
    CandidateDecision,
    DecisionBundle,
    DriverServiceBackendAdapter,
    SingleBackendDriverOrchestrator,
    select_candidate_in_bundle,
)
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.config import UserSimulatorConfig
from alpasim_runtime.event_loop import EventBasedRollout, RuntimeCheckpoint
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_runtime.unbound_rollout import UnboundRollout
from alpasim_runtime.worker.artifact_cache import make_artifact_loader
from alpasim_runtime.worker.ipc import ServiceEndpoints
from alpasim_utils import geometry
from alpasim_utils.geometry import pose_to_grpc
from alpasim_utils.types import ImageWithMetadata
from alpasim_grpc.v0.common_pb2 import DynamicState, Vec3
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata

from eval.schema import EvalConfig

from .frame_store import FrameStore
from .models import (
    ActorStateModel,
    CandidatePlanModel,
    CandidateSummaryModel,
    CheckpointSummaryModel,
    DecisionSummaryModel,
    EgoStateModel,
    FrameDataModel,
    PolylinePointModel,
    SensorDescriptorModel,
    SessionEventModel,
    SessionSnapshotModel,
    SessionStateModel,
    SessionUpdateModel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _InteractiveCheckpoint:
    checkpoint_id: str
    tick_id: int
    sim_time_us: int
    runtime_checkpoint: RuntimeCheckpoint
    frame_store_snapshot: OrderedDict[int, dict[str, FrameDataModel]]
    latest_snapshot: SessionSnapshotModel | None
    pending_frames: list[FrameDataModel]
    status: str


def _frame_encoding_name(image_format_enum: int) -> tuple[str, str]:
    if sensorsim_pb2.ImageFormat.Name(image_format_enum) == "PNG":
        return ("PNG", "image/png")
    return ("JPEG", "image/jpeg")


def _dynamic_state_from_row(row: np.ndarray) -> DynamicState:
    return DynamicState(
        linear_velocity=Vec3(x=float(row[0]), y=float(row[1]), z=float(row[2])),
        angular_velocity=Vec3(x=float(row[3]), y=float(row[4]), z=float(row[5])),
        linear_acceleration=Vec3(x=float(row[6]), y=float(row[7]), z=float(row[8])),
        angular_acceleration=Vec3(
            x=float(row[9]),
            y=float(row[10]),
            z=float(row[11]),
        ),
    )


class InteractiveSessionRunner:
    """Owns one live interactive simulation session."""

    def __init__(
        self,
        *,
        interactive_session_id: str,
        scene_id: str,
        artifact_path: str,
        endpoints: ServiceEndpoints,
        user_config: UserSimulatorConfig,
        eval_config: EvalConfig,
        version_ids: RolloutMetadata.VersionIds,
        rollouts_dir: str,
        max_retained_ticks: int,
        on_released: Callable[[], Awaitable[None]],
    ) -> None:
        self._interactive_session_id = interactive_session_id
        self._scene_id = scene_id
        self._artifact_path = artifact_path
        self._endpoints = endpoints
        self._user_config = user_config
        self._eval_config = eval_config
        self._version_ids = version_ids
        self._rollouts_dir = rollouts_dir
        self._on_released = on_released

        self._lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue[SessionEventModel]] = set()
        self._run_continuously = False
        self._background_task: asyncio.Task[None] | None = None
        self._released = False
        self._closed = False
        self._tick_id = -1
        self._latest_snapshot: SessionSnapshotModel | None = None
        self._status = "CREATED"
        self._error = ""
        self._camera_catalog = CameraCatalog(user_config.extra_cameras)
        self._frame_store = FrameStore(max_retained_ticks=max_retained_ticks)
        self._max_retained_ticks = max(1, max_retained_ticks)
        self._pending_frames: list[FrameDataModel] = []
        self._artifact_loader = make_artifact_loader(
            smooth_trajectories=user_config.smooth_trajectories,
            max_cache_size=user_config.artifact_cache_size,
        )
        self._rollout: EventBasedRollout | None = None
        self._sensors: list[SensorDescriptorModel] = []
        self._checkpoints: OrderedDict[str, _InteractiveCheckpoint] = OrderedDict()

    async def initialize(self) -> SessionStateModel:
        async with self._lock:
            if self._rollout is None:
                artifact = self._artifact_loader(self._scene_id, self._artifact_path)
                rollout = EventBasedRollout(
                    unbound=UnboundRollout.create(
                        simulation_config=self._user_config.simulation_config,
                        scene_id=self._scene_id,
                        version_ids=self._version_ids,
                        available_artifacts={self._scene_id: artifact},
                        rollouts_dir=self._rollouts_dir,
                    ),
                    driver=DriverService(
                        self._endpoints.driver.address,
                        skip=self._endpoints.driver.skip,
                    ),
                    sensorsim=SensorsimService(
                        self._endpoints.sensorsim.address,
                        skip=self._endpoints.sensorsim.skip,
                        camera_catalog=self._camera_catalog,
                    ),
                    physics=PhysicsService(
                        self._endpoints.physics.address,
                        skip=self._endpoints.physics.skip,
                    ),
                    trafficsim=TrafficService(
                        self._endpoints.trafficsim.address,
                        skip=self._endpoints.trafficsim.skip,
                    ),
                    controller=ControllerService(
                        self._endpoints.controller.address,
                        skip=self._endpoints.controller.skip,
                    ),
                    camera_catalog=self._camera_catalog,
                    eval_config=self._eval_config,
                )
                await rollout.initialize()
                rollout.current_state.rendered_images_handler = self._capture_images
                self._rollout = rollout
                self._sensors = self._build_sensors()

            if self._latest_snapshot is None:
                await self._advance_once_locked()

            self._status = "PAUSED"
            state = self._build_state()
        await self._publish(SessionEventModel(state=state))
        return state

    async def start(self) -> SessionStateModel:
        return await self.resume()

    async def resume(self) -> SessionStateModel:
        async with self._lock:
            if self._closed:
                return self._build_state()
            if self._rollout is None:
                raise RuntimeError("session not initialized")
            if self._status == "COMPLETED":
                return self._build_state()
            self._run_continuously = True
            self._status = "RUNNING"
            if self._background_task is None or self._background_task.done():
                self._background_task = asyncio.create_task(self._run_forever())
            state = self._build_state()
        await self._publish(SessionEventModel(state=state))
        return state

    async def pause(self) -> SessionStateModel:
        task: asyncio.Task[None] | None
        async with self._lock:
            self._run_continuously = False
            task = self._background_task
        if task is not None:
            await task
        async with self._lock:
            if self._status not in {"COMPLETED", "FAILED"}:
                self._status = "PAUSED"
            state = self._build_state()
        await self._publish(SessionEventModel(state=state))
        return state

    async def step(self, num_steps: int) -> SessionUpdateModel:
        committed: list[SessionSnapshotModel] = []
        await self.pause()
        async with self._lock:
            for _ in range(max(1, num_steps)):
                if self._status in {"COMPLETED", "FAILED"}:
                    break
                snapshot = await self._advance_once_locked()
                if snapshot is not None:
                    committed.append(snapshot)
                    if self._status == "COMPLETED":
                        break
            state = self._build_state()
        await self._publish(SessionEventModel(state=state))
        return SessionUpdateModel(state=state, committed_snapshots=committed)

    async def close(self) -> None:
        task: asyncio.Task[None] | None
        async with self._lock:
            if self._closed:
                return
            self._run_continuously = False
            task = self._background_task
        if task is not None:
            await task
        async with self._lock:
            completed = self._status == "COMPLETED"
            await self._finalize_locked(
                mark_complete=completed,
                run_evaluation=completed,
            )
            self._closed = True

    async def get_state(self) -> SessionStateModel:
        async with self._lock:
            return self._build_state()

    async def set_active_backends(self, backend_ids: list[str]) -> SessionStateModel:
        async with self._lock:
            if self._rollout is None:
                raise RuntimeError("session not initialized")
            available = set(self._rollout.current_state.available_driver_backend_ids)
            selected = list(backend_ids)
            if selected:
                unknown = [backend_id for backend_id in selected if backend_id not in available]
                if unknown:
                    raise KeyError(f"Unknown backend ids: {unknown}")
                self._rollout.current_state.active_driver_backend_ids = selected
            else:
                self._rollout.current_state.active_driver_backend_ids = list(available)
            state = self._build_state()
        await self._publish(SessionEventModel(state=state))
        return state

    async def list_candidates(self) -> list[CandidateSummaryModel]:
        async with self._lock:
            decision = self._latest_decision_summary()
            if decision is None:
                return []
            return list(decision.candidates)

    async def list_checkpoints(self) -> list[CheckpointSummaryModel]:
        async with self._lock:
            return [
                CheckpointSummaryModel(
                    checkpoint_id=checkpoint.checkpoint_id,
                    tick_id=checkpoint.tick_id,
                    sim_time_us=checkpoint.sim_time_us,
                    status=checkpoint.status,
                    restore_supported=not bool(
                        checkpoint.runtime_checkpoint.unsupported_backend_ids
                    ),
                    unsupported_backend_ids=list(
                        checkpoint.runtime_checkpoint.unsupported_backend_ids
                    ),
                )
                for checkpoint in self._checkpoints.values()
            ]

    async def recompute_candidate(self, backend_id: str) -> SessionStateModel:
        await self.pause()
        async with self._lock:
            if self._rollout is None:
                raise RuntimeError("session not initialized")
            bundle = self._rollout.current_state.last_committed_decision_bundle
            if bundle is None:
                raise RuntimeError("No committed decision bundle available for recompute")
            available = set(self._rollout.current_state.available_driver_backend_ids)
            if backend_id not in available:
                raise KeyError(f"Unknown backend_id: {backend_id}")
            orchestrator = self._driver_orchestrator()
            updated_bundle = await orchestrator.recompute_candidate(bundle, backend_id)
            self._rollout.current_state.last_committed_decision_bundle = updated_bundle
            if self._latest_snapshot is not None:
                self._latest_snapshot = replace(
                    self._latest_snapshot,
                    latest_decision=_decision_summary_from_bundle(updated_bundle),
                )
            if self._latest_snapshot is not None:
                self._record_checkpoint_locked(self._latest_snapshot)
            state = self._build_state()
        await self._publish(SessionEventModel(state=state))
        return state

    async def select_candidate(self, candidate_id: str) -> SessionStateModel:
        await self.pause()
        async with self._lock:
            if self._rollout is None:
                raise RuntimeError("session not initialized")
            bundle = self._rollout.current_state.last_committed_decision_bundle
            if bundle is None:
                raise RuntimeError("No committed decision bundle available for selection")
            updated_bundle = select_candidate_in_bundle(bundle, candidate_id)
            self._rollout.current_state.last_committed_decision_bundle = updated_bundle
            if self._latest_snapshot is not None:
                self._latest_snapshot = replace(
                    self._latest_snapshot,
                    latest_decision=_decision_summary_from_bundle(updated_bundle),
                )
                self._record_checkpoint_locked(self._latest_snapshot)
            state = self._build_state()
        await self._publish(SessionEventModel(state=state))
        return state

    async def restore_checkpoint(self, checkpoint_id: str) -> SessionStateModel:
        await self.pause()
        async with self._lock:
            if self._rollout is None:
                raise RuntimeError("session not initialized")
            checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint is None:
                raise KeyError(f"Unknown checkpoint_id: {checkpoint_id}")
            if checkpoint.runtime_checkpoint.unsupported_backend_ids:
                raise RuntimeError(
                    "Checkpoint restore is not supported for backends: "
                    + ", ".join(checkpoint.runtime_checkpoint.unsupported_backend_ids)
                )
            await self._rollout.restore_runtime_checkpoint(checkpoint.runtime_checkpoint)
            self._rollout.current_state.rendered_images_handler = self._capture_images
            self._frame_store.restore(checkpoint.frame_store_snapshot)
            self._tick_id = checkpoint.tick_id
            self._latest_snapshot = copy.deepcopy(checkpoint.latest_snapshot)
            self._pending_frames = copy.deepcopy(checkpoint.pending_frames)
            self._status = "PAUSED"
            self._error = ""
            state = self._build_state()
        await self._publish(SessionEventModel(state=state))
        return state

    async def list_sensors(self) -> list[SensorDescriptorModel]:
        async with self._lock:
            return list(self._sensors)

    async def get_frame(self, sensor_id: str, tick_id: int) -> FrameDataModel:
        async with self._lock:
            return self._frame_store.get_frame(sensor_id=sensor_id, tick_id=tick_id)

    def subscribe(self) -> asyncio.Queue[SessionEventModel]:
        queue: asyncio.Queue[SessionEventModel] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[SessionEventModel]) -> None:
        self._subscribers.discard(queue)

    async def _run_forever(self) -> None:
        try:
            while True:
                async with self._lock:
                    if not self._run_continuously or self._status in {
                        "COMPLETED",
                        "FAILED",
                    }:
                        break
                    await self._advance_once_locked()
                    state = self._build_state()
                await self._publish(SessionEventModel(state=state))
        except Exception as exc:  # noqa: BLE001
            logger.exception("interactive session failed: %s", self._interactive_session_id)
            async with self._lock:
                self._status = "FAILED"
                self._error = str(exc)
                await self._finalize_locked(mark_complete=False, run_evaluation=False)
                state = self._build_state()
            await self._publish(SessionEventModel(state=state))
        finally:
            async with self._lock:
                if self._status not in {"COMPLETED", "FAILED"}:
                    self._status = "PAUSED"
                self._background_task = None

    async def _advance_once_locked(self) -> SessionSnapshotModel | None:
        assert self._rollout is not None
        result = await self._rollout.run_until_step_commit()
        snapshot = None
        if result.step_committed:
            self._tick_id += 1
            frame_refs = self._frame_store.store_tick_frames(
                tick_id=self._tick_id,
                frames=self._pending_frames,
            )
            self._pending_frames = []
            snapshot = self._build_snapshot(frame_refs)
            self._latest_snapshot = snapshot
            self._record_checkpoint_locked(snapshot)
            await self._publish(SessionEventModel(snapshot=snapshot))
        if result.simulation_finished:
            self._status = "COMPLETED"
        return snapshot

    async def _capture_images(self, images: list[ImageWithMetadata]) -> None:
        if self._rollout is None:
            return
        frame_encoding, content_type = _frame_encoding_name(
            self._rollout.unbound.image_format
        )
        for image in images:
            self._pending_frames.append(
                FrameDataModel(
                    sensor_id=image.camera_logical_id,
                    frame_start_us=image.start_timestamp_us,
                    frame_end_us=image.end_timestamp_us,
                    frame_encoding=frame_encoding,
                    content_type=content_type,
                    image_bytes=image.image_bytes,
                )
            )

    def _build_sensors(self) -> list[SensorDescriptorModel]:
        assert self._rollout is not None
        sensors: list[SensorDescriptorModel] = []
        frame_encoding, _ = _frame_encoding_name(self._rollout.unbound.image_format)
        for camera in self._rollout.runtime_cameras:
            definition = self._camera_catalog.get_camera_definition(
                self._scene_id, camera.logical_id
            )
            sensors.append(
                SensorDescriptorModel(
                    sensor_id=camera.logical_id,
                    logical_id=camera.logical_id,
                    nominal_width=camera.render_resolution_hw[1],
                    nominal_height=camera.render_resolution_hw[0],
                    nominal_frame_interval_us=camera.clock.interval_us,
                    rig_to_sensor=pose_to_grpc(definition.rig_to_camera),
                    frame_encoding=frame_encoding,
                )
            )
        return sensors

    def _build_snapshot(
        self,
        frame_refs: list[object],
    ) -> SessionSnapshotModel:
        assert self._rollout is not None
        state = self._rollout.current_state
        sim_time_us = int(state.ego_trajectory.timestamps_us[-1])
        ego_pose = pose_to_grpc(state.ego_trajectory.get_pose(-1))
        ego_dynamics = _dynamic_state_from_row(state.ego_trajectory.dynamics[-1])

        actors: list[ActorStateModel] = []
        for actor_id, traffic_obj in state.traffic_objs.items():
            if sim_time_us not in traffic_obj.trajectory.time_range_us:
                continue
            actors.append(
                ActorStateModel(
                    actor_id=actor_id,
                    pose=geometry.pose_to_grpc(
                        traffic_obj.trajectory.interpolate_pose(sim_time_us)
                    ),
                )
            )

        ego_history = _polyline_from_positions(state.ego_trajectory.positions)
        candidate_plans = _candidate_plans_from_bundle(state.last_committed_decision_bundle)
        selected_plan = next(
            (candidate.points for candidate in candidate_plans if candidate.selected),
            [],
        )

        return SessionSnapshotModel(
            interactive_session_id=self._interactive_session_id,
            tick_id=self._tick_id,
            sim_time_us=sim_time_us,
            ego=EgoStateModel(
                pose=ego_pose,
                dynamics=ego_dynamics,
                front_steering_angle_rad=state.current_front_steering_angle_rad,
            ),
            actors=actors,
            frame_refs=list(frame_refs),
            latest_decision=self._latest_decision_summary(),
            ego_history=ego_history,
            selected_plan=selected_plan,
            candidate_plans=candidate_plans,
        )

    def _build_state(self) -> SessionStateModel:
        sim_time_us = self._latest_snapshot.sim_time_us if self._latest_snapshot else 0
        return SessionStateModel(
            interactive_session_id=self._interactive_session_id,
            rollout_uuid=(
                self._rollout.unbound.rollout_uuid if self._rollout is not None else ""
            ),
            scene_id=self._scene_id,
            status=self._status,
            current_tick_id=max(self._tick_id, 0) if self._latest_snapshot else 0,
            current_sim_time_us=sim_time_us,
            latest_snapshot=self._latest_snapshot,
            latest_decision=self._latest_decision_summary(),
            active_backend_ids=self._active_backend_ids(),
            available_backend_ids=self._available_backend_ids(),
            error=self._error,
        )

    def _latest_decision_summary(self) -> DecisionSummaryModel | None:
        if self._rollout is None:
            return self._latest_snapshot.latest_decision if self._latest_snapshot else None
        bundle = self._rollout.current_state.last_committed_decision_bundle
        if bundle is None:
            return self._latest_snapshot.latest_decision if self._latest_snapshot else None
        return _decision_summary_from_bundle(bundle)

    def _active_backend_ids(self) -> list[str]:
        if self._rollout is None:
            return []
        active = self._rollout.current_state.active_driver_backend_ids
        return list(active) if active is not None else []

    def _available_backend_ids(self) -> list[str]:
        if self._rollout is None:
            return []
        return list(self._rollout.current_state.available_driver_backend_ids)

    def _driver_orchestrator(self):
        assert self._rollout is not None
        orchestrator = self._rollout._build_default_driver_orchestrator()
        if orchestrator is not None:
            return orchestrator
        return SingleBackendDriverOrchestrator(
            DriverServiceBackendAdapter(self._rollout.driver)
        )

    def _record_checkpoint_locked(self, snapshot: SessionSnapshotModel) -> None:
        assert self._rollout is not None
        checkpoint_id = f"tick-{self._tick_id}"
        checkpoint = _InteractiveCheckpoint(
            checkpoint_id=checkpoint_id,
            tick_id=self._tick_id,
            sim_time_us=snapshot.sim_time_us,
            runtime_checkpoint=self._rollout.capture_runtime_checkpoint(),
            frame_store_snapshot=self._frame_store.snapshot(),
            latest_snapshot=copy.deepcopy(snapshot),
            pending_frames=copy.deepcopy(self._pending_frames),
            status="PAUSED",
        )
        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoints.move_to_end(checkpoint_id)
        while len(self._checkpoints) > self._max_retained_ticks:
            self._checkpoints.popitem(last=False)

    async def _publish(self, event: SessionEventModel) -> None:
        dead: list[asyncio.Queue[SessionEventModel]] = []
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(queue)
        for queue in dead:
            self._subscribers.discard(queue)

    async def _finalize_locked(
        self,
        *,
        mark_complete: bool,
        run_evaluation: bool,
    ) -> None:
        if self._rollout is not None:
            await self._rollout.aclose(
                mark_complete=mark_complete,
                run_evaluation=run_evaluation,
            )
            self._rollout = None
        if not self._released:
            await self._on_released()
            self._released = True


def _candidate_summary_from_decision(
    candidate: CandidateDecision,
    *,
    selected_candidate_id: str | None,
) -> CandidateSummaryModel:
    return CandidateSummaryModel(
        candidate_id=candidate.candidate_id,
        backend_id=candidate.backend_id,
        status=candidate.status.value,
        selected=candidate.candidate_id == selected_candidate_id,
        error=candidate.error or "",
    )


def _decision_summary_from_bundle(bundle: DecisionBundle) -> DecisionSummaryModel:
    return DecisionSummaryModel(
        step_id=bundle.snapshot.step_id,
        input_snapshot_id=bundle.snapshot.input_snapshot_id,
        selected_candidate_id=bundle.selected_candidate_id,
        candidates=[
            _candidate_summary_from_decision(
                candidate,
                selected_candidate_id=bundle.selected_candidate_id,
            )
            for candidate in bundle.candidates
        ],
        arbitration_reason=bundle.arbitration_reason or "",
    )


def _polyline_from_positions(positions: np.ndarray | None) -> list[PolylinePointModel]:
    if positions is None or len(positions) == 0:
        return []
    return [
        PolylinePointModel(x=float(position[0]), y=float(position[1]))
        for position in positions
    ]


def _selected_plan_from_bundle(
    bundle: DecisionBundle | None,
) -> list[PolylinePointModel]:
    if bundle is None:
        return []

    selected_candidate: CandidateDecision | None = None
    if bundle.selected_candidate_id is not None:
        selected_candidate = next(
            (
                candidate
                for candidate in bundle.candidates
                if candidate.candidate_id == bundle.selected_candidate_id
            ),
            None,
        )
    if selected_candidate is None:
        selected_candidate = next(
            (candidate for candidate in bundle.candidates if candidate.status.value == "SELECTED"),
            None,
        )
    if selected_candidate is None or selected_candidate.trajectory is None:
        return []
    return _polyline_from_positions(selected_candidate.trajectory.positions)


def _candidate_plans_from_bundle(
    bundle: DecisionBundle | None,
) -> list[CandidatePlanModel]:
    if bundle is None:
        return []
    plans: list[CandidatePlanModel] = []
    for candidate in bundle.candidates:
        if candidate.trajectory is None:
            continue
        plans.append(
            CandidatePlanModel(
                candidate_id=candidate.candidate_id,
                backend_id=candidate.backend_id,
                selected=candidate.candidate_id == bundle.selected_candidate_id,
                points=_polyline_from_positions(candidate.trajectory.positions),
            )
        )
    return plans
