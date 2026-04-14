# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import logging

from alpasim_grpc.v1 import interactive_runtime_pb2, interactive_runtime_pb2_grpc
from alpasim_runtime.interactive.models import (
    ActorStateModel,
    CandidateSummaryModel,
    CheckpointSummaryModel,
    DecisionSummaryModel,
    EgoStateModel,
    FrameRefModel,
    SensorDescriptorModel,
    SessionSnapshotModel,
    SessionStateModel,
    SessionUpdateModel,
)
from alpasim_runtime.interactive.session_manager import InteractiveSessionManager

import grpc

logger = logging.getLogger(__name__)


def _status_to_proto(status: str) -> interactive_runtime_pb2.SessionStatus.ValueType:
    mapping = {
        "CREATED": interactive_runtime_pb2.SESSION_STATUS_CREATED,
        "RUNNING": interactive_runtime_pb2.SESSION_STATUS_RUNNING,
        "PAUSED": interactive_runtime_pb2.SESSION_STATUS_PAUSED,
        "COMPLETED": interactive_runtime_pb2.SESSION_STATUS_COMPLETED,
        "FAILED": interactive_runtime_pb2.SESSION_STATUS_FAILED,
    }
    return mapping.get(status, interactive_runtime_pb2.SESSION_STATUS_UNSPECIFIED)


def _frame_encoding_to_proto(encoding: str) -> interactive_runtime_pb2.FrameEncoding.ValueType:
    if encoding == "PNG":
        return interactive_runtime_pb2.FRAME_ENCODING_PNG
    if encoding == "JPEG":
        return interactive_runtime_pb2.FRAME_ENCODING_JPEG
    return interactive_runtime_pb2.FRAME_ENCODING_UNSPECIFIED


def _sensor_to_proto(sensor: SensorDescriptorModel) -> interactive_runtime_pb2.SensorDescriptor:
    return interactive_runtime_pb2.SensorDescriptor(
        sensor_id=sensor.sensor_id,
        logical_id=sensor.logical_id,
        sensor_type=interactive_runtime_pb2.SENSOR_TYPE_CAMERA,
        nominal_width=sensor.nominal_width,
        nominal_height=sensor.nominal_height,
        nominal_frame_interval_us=sensor.nominal_frame_interval_us,
        rig_to_sensor=sensor.rig_to_sensor,
        frame_encoding=_frame_encoding_to_proto(sensor.frame_encoding),
    )


def _frame_ref_to_proto(frame_ref: FrameRefModel) -> interactive_runtime_pb2.FrameRef:
    return interactive_runtime_pb2.FrameRef(
        sensor_id=frame_ref.sensor_id,
        tick_id=frame_ref.tick_id,
        frame_start_us=frame_ref.frame_start_us,
        frame_end_us=frame_ref.frame_end_us,
        frame_encoding=_frame_encoding_to_proto(frame_ref.frame_encoding),
    )


def _ego_to_proto(ego: EgoStateModel) -> interactive_runtime_pb2.EgoState:
    return interactive_runtime_pb2.EgoState(
        pose=ego.pose,
        dynamics=ego.dynamics,
    )


def _actor_to_proto(actor: ActorStateModel) -> interactive_runtime_pb2.ActorState:
    return interactive_runtime_pb2.ActorState(actor_id=actor.actor_id, pose=actor.pose)


def _snapshot_to_proto(snapshot: SessionSnapshotModel) -> interactive_runtime_pb2.SessionSnapshot:
    message = interactive_runtime_pb2.SessionSnapshot(
        interactive_session_id=snapshot.interactive_session_id,
        tick_id=snapshot.tick_id,
        sim_time_us=snapshot.sim_time_us,
        ego=_ego_to_proto(snapshot.ego),
        actors=[_actor_to_proto(actor) for actor in snapshot.actors],
        frame_refs=[_frame_ref_to_proto(frame_ref) for frame_ref in snapshot.frame_refs],
    )
    if snapshot.latest_decision is not None and hasattr(message, "latest_decision"):
        message.latest_decision.CopyFrom(_decision_to_proto(snapshot.latest_decision))
    return message


def _state_to_proto(state: SessionStateModel) -> interactive_runtime_pb2.SessionState:
    message = interactive_runtime_pb2.SessionState(
        interactive_session_id=state.interactive_session_id,
        rollout_uuid=state.rollout_uuid,
        scene_id=state.scene_id,
        status=_status_to_proto(state.status),
        current_tick_id=state.current_tick_id,
        current_sim_time_us=state.current_sim_time_us,
        error=state.error,
        active_backend_ids=state.active_backend_ids,
    )
    if state.latest_snapshot is not None:
        message.latest_snapshot.CopyFrom(_snapshot_to_proto(state.latest_snapshot))
    if state.latest_decision is not None and hasattr(message, "latest_decision"):
        message.latest_decision.CopyFrom(_decision_to_proto(state.latest_decision))
    return message


def _candidate_to_proto(
    candidate: CandidateSummaryModel,
) -> interactive_runtime_pb2.CandidateSummary:
    return interactive_runtime_pb2.CandidateSummary(
        candidate_id=candidate.candidate_id,
        backend_id=candidate.backend_id,
        status=candidate.status,
        selected=candidate.selected,
        error=candidate.error,
    )


def _decision_to_proto(
    decision: DecisionSummaryModel,
) -> interactive_runtime_pb2.DecisionSummary:
    return interactive_runtime_pb2.DecisionSummary(
        step_id=decision.step_id,
        input_snapshot_id=decision.input_snapshot_id,
        selected_candidate_id=decision.selected_candidate_id or "",
        candidates=[_candidate_to_proto(candidate) for candidate in decision.candidates],
        arbitration_reason=decision.arbitration_reason,
    )


def _checkpoint_to_proto(
    checkpoint: CheckpointSummaryModel,
) -> interactive_runtime_pb2.CheckpointSummary:
    return interactive_runtime_pb2.CheckpointSummary(
        checkpoint_id=checkpoint.checkpoint_id,
        tick_id=checkpoint.tick_id,
        sim_time_us=checkpoint.sim_time_us,
        status=_status_to_proto(checkpoint.status),
        restore_supported=checkpoint.restore_supported,
        unsupported_backend_ids=checkpoint.unsupported_backend_ids,
    )


def _update_to_proto(
    update: SessionUpdateModel,
) -> interactive_runtime_pb2.StepSessionResponse:
    return interactive_runtime_pb2.StepSessionResponse(
        state=_state_to_proto(update.state),
        committed_snapshots=[
            _snapshot_to_proto(snapshot) for snapshot in update.committed_snapshots
        ],
    )


class InteractiveRuntimeServicer(
    interactive_runtime_pb2_grpc.InteractiveRuntimeServiceServicer
):
    def __init__(self, manager: InteractiveSessionManager):
        self._manager = manager

    async def CreateSession(self, request, context):
        try:
            state = await self._manager.create_session(
                scene_id=request.scene_id,
                start_paused=request.start_paused,
                max_retained_ticks=request.max_retained_ticks or 64,
            )
            return interactive_runtime_pb2.CreateSessionResponse(
                interactive_session_id=state.interactive_session_id,
                initial_state=_state_to_proto(state),
            )
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except RuntimeError as exc:
            if "No service capacity available" in str(exc):
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
            logger.exception("interactive CreateSession failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("interactive CreateSession failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def ListSessions(self, request, context):
        del request
        try:
            sessions = await self._manager.list_sessions()
            return interactive_runtime_pb2.ListSessionsResponse(
                sessions=[_state_to_proto(state) for state in sessions]
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("interactive ListSessions failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def StartSession(self, request, context):
        return await self._with_state(
            context, self._manager.start_session(request.interactive_session_id)
        )

    async def PauseSession(self, request, context):
        return await self._with_state(
            context, self._manager.pause_session(request.interactive_session_id)
        )

    async def ResumeSession(self, request, context):
        return await self._with_state(
            context, self._manager.resume_session(request.interactive_session_id)
        )

    async def StepSession(self, request, context):
        try:
            update = await self._manager.step_session(
                request.interactive_session_id,
                max(int(request.num_steps), 1),
            )
            return _update_to_proto(update)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def GetSessionState(self, request, context):
        return await self._with_state(
            context, self._manager.get_state(request.interactive_session_id)
        )

    async def SetActiveBackends(self, request, context):
        try:
            state = await self._manager.set_active_backends(
                request.interactive_session_id,
                list(request.backend_ids),
            )
            return _state_to_proto(state)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def ListCandidates(self, request, context):
        try:
            candidates = await self._manager.list_candidates(
                request.interactive_session_id
            )
            return interactive_runtime_pb2.ListCandidatesResponse(
                candidates=[_candidate_to_proto(candidate) for candidate in candidates]
            )
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def RecomputeCandidate(self, request, context):
        try:
            state = await self._manager.recompute_candidate(
                request.interactive_session_id,
                request.backend_id,
            )
            return _state_to_proto(state)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except RuntimeError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def SelectCandidate(self, request, context):
        try:
            state = await self._manager.select_candidate(
                request.interactive_session_id,
                request.candidate_id,
            )
            return _state_to_proto(state)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except RuntimeError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def ListCheckpoints(self, request, context):
        try:
            checkpoints = await self._manager.list_checkpoints(
                request.interactive_session_id
            )
            return interactive_runtime_pb2.ListCheckpointsResponse(
                checkpoints=[
                    _checkpoint_to_proto(checkpoint) for checkpoint in checkpoints
                ]
            )
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def RestoreCheckpoint(self, request, context):
        try:
            state = await self._manager.restore_checkpoint(
                request.interactive_session_id,
                request.checkpoint_id,
            )
            return _state_to_proto(state)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except RuntimeError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def ListSensors(self, request, context):
        try:
            sensors = await self._manager.list_sensors(request.interactive_session_id)
            return interactive_runtime_pb2.ListSensorsResponse(
                sensors=[_sensor_to_proto(sensor) for sensor in sensors]
            )
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def GetFrame(self, request, context):
        try:
            frame = await self._manager.get_frame(
                request.interactive_session_id,
                request.sensor_id,
                int(request.tick_id),
            )
            return interactive_runtime_pb2.GetFrameResponse(
                frame_ref=_frame_ref_to_proto(
                    FrameRefModel(
                        sensor_id=frame.sensor_id,
                        tick_id=int(request.tick_id),
                        frame_start_us=frame.frame_start_us,
                        frame_end_us=frame.frame_end_us,
                        frame_encoding=frame.frame_encoding,
                    )
                ),
                image_bytes=frame.image_bytes,
            )
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        raise RuntimeError("context.abort did not terminate request")

    async def StreamSession(self, request, context):
        queue: asyncio.Queue | None = None
        try:
            state = await self._manager.get_state(request.interactive_session_id)
            yield interactive_runtime_pb2.SessionEvent(state=_state_to_proto(state))
            queue = await self._manager.subscribe(request.interactive_session_id)
            while True:
                event = await queue.get()
                if event.snapshot is not None:
                    if request.include_snapshots:
                        yield interactive_runtime_pb2.SessionEvent(
                            snapshot=_snapshot_to_proto(event.snapshot)
                        )
                    continue
                if event.state is not None:
                    yield interactive_runtime_pb2.SessionEvent(
                        state=_state_to_proto(event.state)
                    )
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except asyncio.CancelledError:
            return
        finally:
            if queue is not None:
                try:
                    await self._manager.unsubscribe(
                        request.interactive_session_id, queue
                    )
                except KeyError:
                    pass
        return

    @staticmethod
    async def _with_state(context, awaitable):
        try:
            state = await awaitable
            return _state_to_proto(state)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        raise RuntimeError("context.abort did not terminate request")
