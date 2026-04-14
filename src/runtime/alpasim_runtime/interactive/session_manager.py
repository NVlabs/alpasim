# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from alpasim_runtime.address_pool import AddressPool, release_all, try_acquire_all
from alpasim_runtime.config import UserSimulatorConfig
from alpasim_runtime.worker.ipc import ServiceEndpoints
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata

from eval.schema import EvalConfig

from .models import (
    CandidateSummaryModel,
    CheckpointSummaryModel,
    FrameDataModel,
    SensorDescriptorModel,
    SessionStateModel,
    SessionUpdateModel,
)
from .session_runner import InteractiveSessionRunner


@dataclass
class _SessionEntry:
    runner: InteractiveSessionRunner


class InteractiveSessionManager:
    def __init__(
        self,
        *,
        user_config: UserSimulatorConfig,
        eval_config: EvalConfig,
        version_ids: RolloutMetadata.VersionIds,
        scene_id_to_artifact_path: dict[str, str],
        pools: dict[str, AddressPool],
        rollouts_dir: str,
    ) -> None:
        self._user_config = user_config
        self._eval_config = eval_config
        self._version_ids = version_ids
        self._scene_id_to_artifact_path = scene_id_to_artifact_path
        self._pools = pools
        self._rollouts_dir = rollouts_dir
        self._sessions: dict[str, _SessionEntry] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        *,
        scene_id: str,
        start_paused: bool,
        max_retained_ticks: int,
    ) -> SessionStateModel:
        session_id: str | None = None
        runner: InteractiveSessionRunner | None = None
        async with self._lock:
            if scene_id not in self._scene_id_to_artifact_path:
                raise KeyError(f"Unknown scene_id: {scene_id}")
            acquired = try_acquire_all(self._pools)
            if acquired is None:
                raise RuntimeError("No service capacity available for interactive session")

            async def _release() -> None:
                release_all(self._pools, acquired)

            session_id = uuid.uuid4().hex
            runner = InteractiveSessionRunner(
                interactive_session_id=session_id,
                scene_id=scene_id,
                artifact_path=self._scene_id_to_artifact_path[scene_id],
                endpoints=ServiceEndpoints(
                    driver=acquired["driver"],
                    sensorsim=acquired["sensorsim"],
                    physics=acquired["physics"],
                    trafficsim=acquired["trafficsim"],
                    controller=acquired["controller"],
                ),
                user_config=self._user_config,
                eval_config=self._eval_config,
                version_ids=self._version_ids,
                rollouts_dir=self._rollouts_dir,
                max_retained_ticks=max_retained_ticks,
                on_released=_release,
            )
            self._sessions[session_id] = _SessionEntry(runner=runner)

        assert runner is not None
        assert session_id is not None
        try:
            state = await runner.initialize()
            if not start_paused:
                state = await runner.start()
            return state
        except Exception:
            async with self._lock:
                self._sessions.pop(session_id, None)
            await runner.close()
            raise

    async def get_state(self, session_id: str) -> SessionStateModel:
        runner = await self._get_runner(session_id)
        return await runner.get_state()

    async def list_sessions(self) -> list[SessionStateModel]:
        async with self._lock:
            runners = [entry.runner for entry in self._sessions.values()]
        if not runners:
            return []
        return [await runner.get_state() for runner in runners]

    async def list_sensors(self, session_id: str) -> list[SensorDescriptorModel]:
        runner = await self._get_runner(session_id)
        return await runner.list_sensors()

    async def list_candidates(self, session_id: str) -> list[CandidateSummaryModel]:
        runner = await self._get_runner(session_id)
        return await runner.list_candidates()

    async def recompute_candidate(
        self, session_id: str, backend_id: str
    ) -> SessionStateModel:
        runner = await self._get_runner(session_id)
        return await runner.recompute_candidate(backend_id)

    async def select_candidate(
        self, session_id: str, candidate_id: str
    ) -> SessionStateModel:
        runner = await self._get_runner(session_id)
        return await runner.select_candidate(candidate_id)

    async def set_active_backends(
        self, session_id: str, backend_ids: list[str]
    ) -> SessionStateModel:
        runner = await self._get_runner(session_id)
        return await runner.set_active_backends(backend_ids)

    async def list_checkpoints(self, session_id: str) -> list[CheckpointSummaryModel]:
        runner = await self._get_runner(session_id)
        return await runner.list_checkpoints()

    async def restore_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> SessionStateModel:
        runner = await self._get_runner(session_id)
        return await runner.restore_checkpoint(checkpoint_id)

    async def start_session(self, session_id: str) -> SessionStateModel:
        runner = await self._get_runner(session_id)
        return await runner.start()

    async def pause_session(self, session_id: str) -> SessionStateModel:
        runner = await self._get_runner(session_id)
        return await runner.pause()

    async def resume_session(self, session_id: str) -> SessionStateModel:
        runner = await self._get_runner(session_id)
        return await runner.resume()

    async def step_session(self, session_id: str, num_steps: int) -> SessionUpdateModel:
        runner = await self._get_runner(session_id)
        return await runner.step(num_steps=num_steps)

    async def get_frame(self, session_id: str, sensor_id: str, tick_id: int) -> FrameDataModel:
        runner = await self._get_runner(session_id)
        return await runner.get_frame(sensor_id=sensor_id, tick_id=tick_id)

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        return (await self._get_runner(session_id)).subscribe()

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        runner = await self._get_runner(session_id)
        runner.unsubscribe(queue)

    async def close_all(self) -> None:
        async with self._lock:
            entries = list(self._sessions.items())
            self._sessions.clear()
        for _, entry in entries:
            await entry.runner.close()

    async def _get_runner(self, session_id: str) -> InteractiveSessionRunner:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown interactive_session_id: {session_id}")
            return self._sessions[session_id].runner
