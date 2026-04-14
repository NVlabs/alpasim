# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Decision orchestration primitives for runtime policy evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Protocol, Sequence

from alpasim_runtime.services.driver_service import DriverService
from alpasim_utils import geometry


class CandidateStatus(str, Enum):
    """Lifecycle state for a candidate decision."""

    PENDING = "PENDING"
    READY = "READY"
    SELECTED = "SELECTED"
    REJECTED = "REJECTED"
    STALE = "STALE"
    NEEDS_RECOMPUTE = "NEEDS_RECOMPUTE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class BackendMetadata:
    """Declared capabilities for a driver backend."""

    backend_id: str
    backend_type: str
    supports_parallel: bool = False
    supports_hot_switch: bool = False
    supports_restore: bool = False
    priority: int = 0


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """A stable per-step input snapshot shared by candidate generation."""

    step_id: int
    input_snapshot_id: str
    time_now_us: int
    time_query_us: int
    ego_pose_history_timestamps_us: list[int]
    traffic_actor_ids: list[str]
    route_waypoints_in_rig: list[list[float]]
    planner_context: dict[str, Any] | None
    renderer_data: bytes | None
    camera_frame_timestamps_us: dict[str, int]


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    """One backend's candidate output for a given input snapshot."""

    candidate_id: str
    step_id: int
    input_snapshot_id: str
    backend_id: str
    status: CandidateStatus
    trajectory: geometry.Trajectory | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    generated_at_us: int = 0
    recompute_count: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionBundle:
    """Candidate set and selection state for one policy step."""

    snapshot: DecisionSnapshot
    candidates: list[CandidateDecision]
    selected_candidate_id: str | None = None
    arbitration_reason: str | None = None


def _normalize_for_hash(value: Any) -> Any:
    """Recursively normalize arbitrary values into a hashable JSON shape."""
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_hash(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        try:
            return _normalize_for_hash(value.tolist())
        except Exception:
            pass
    return repr(value)


def build_input_snapshot_id(
    *,
    step_id: int,
    time_now_us: int,
    time_query_us: int,
    planner_context: dict[str, Any] | None,
    route_waypoints_in_rig: Sequence[Sequence[float]],
    traffic_actor_ids: Sequence[str],
    ego_pose_history_timestamps_us: Sequence[int],
    camera_frame_timestamps_us: dict[str, int],
    renderer_data: bytes | None,
) -> str:
    """Build a deterministic identity for a per-step policy input snapshot."""
    payload = {
        "camera_frame_timestamps_us": dict(sorted(camera_frame_timestamps_us.items())),
        "ego_pose_history_timestamps_us": list(ego_pose_history_timestamps_us),
        "planner_context": _normalize_for_hash(planner_context),
        "renderer_data_sha256": hashlib.sha256(renderer_data or b"").hexdigest(),
        "route_waypoints_in_rig": [
            list(waypoint) for waypoint in route_waypoints_in_rig
        ],
        "step_id": step_id,
        "time_now_us": time_now_us,
        "time_query_us": time_query_us,
        "traffic_actor_ids": list(traffic_actor_ids),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


class DriverBackendAdapter(Protocol):
    """Interface used by the runtime orchestrator to query backends."""

    metadata: BackendMetadata

    async def infer(self, snapshot: DecisionSnapshot) -> CandidateDecision: ...

    def capture_backend_state(self) -> Any | None: ...

    async def restore_backend_state(self, state: Any) -> None: ...


class DriverOrchestrator(Protocol):
    """Runtime-facing orchestrator interface for candidate generation."""

    async def generate_candidates(
        self,
        snapshot: DecisionSnapshot,
        backend_ids: Sequence[str] | None = None,
    ) -> DecisionBundle: ...

    async def select_candidate(self, bundle: DecisionBundle) -> CandidateDecision: ...

    async def recompute_candidate(
        self,
        bundle: DecisionBundle,
        backend_id: str,
    ) -> DecisionBundle: ...

    def capture_backend_checkpoint(self) -> tuple[dict[str, Any], list[str]]: ...

    async def restore_backend_checkpoint(self, checkpoint: dict[str, Any]) -> None: ...


class DriverBackendRegistry:
    """Static registry of available driver backends for one rollout session."""

    def __init__(self, backends: Sequence[DriverBackendAdapter]):
        if not backends:
            raise ValueError("DriverBackendRegistry requires at least one backend")
        self._backends = list(backends)

    @property
    def backends(self) -> list[DriverBackendAdapter]:
        return list(self._backends)

    @property
    def backend_ids(self) -> list[str]:
        return [backend.metadata.backend_id for backend in self._backends]

    def get_backend(self, backend_id: str) -> DriverBackendAdapter:
        for backend in self._backends:
            if backend.metadata.backend_id == backend_id:
                return backend
        raise KeyError(f"Unknown backend_id: {backend_id}")


def _next_recompute_index(bundle: DecisionBundle, backend_id: str) -> int:
    return (
        max(
            (
                candidate.recompute_count
                for candidate in bundle.candidates
                if candidate.backend_id == backend_id
            ),
            default=-1,
        )
        + 1
    )


def _merge_recomputed_candidate(
    bundle: DecisionBundle,
    candidate: CandidateDecision,
) -> DecisionBundle:
    updated_candidates: list[CandidateDecision] = []
    for existing in bundle.candidates:
        if (
            existing.backend_id == candidate.backend_id
            and existing.candidate_id != bundle.selected_candidate_id
            and existing.status in {CandidateStatus.READY, CandidateStatus.FAILED}
        ):
            updated_candidates.append(replace(existing, status=CandidateStatus.STALE))
            continue
        updated_candidates.append(existing)
    updated_candidates.append(candidate)
    return DecisionBundle(
        snapshot=bundle.snapshot,
        candidates=updated_candidates,
        selected_candidate_id=bundle.selected_candidate_id,
        arbitration_reason=(
            f"{bundle.arbitration_reason};recomputed:{candidate.backend_id}"
            if bundle.arbitration_reason
            else f"recomputed:{candidate.backend_id}"
        ),
    )


def select_candidate_in_bundle(
    bundle: DecisionBundle,
    candidate_id: str,
) -> DecisionBundle:
    """Return a new bundle with an explicitly selected candidate."""
    selected_found = False
    updated_candidates: list[CandidateDecision] = []
    for candidate in bundle.candidates:
        if candidate.candidate_id == candidate_id:
            selected_found = True
            updated_candidates.append(replace(candidate, status=CandidateStatus.SELECTED))
            continue
        if candidate.candidate_id == bundle.selected_candidate_id:
            updated_candidates.append(replace(candidate, status=CandidateStatus.REJECTED))
            continue
        updated_candidates.append(candidate)

    if not selected_found:
        raise KeyError(f"Unknown candidate_id: {candidate_id}")

    return DecisionBundle(
        snapshot=bundle.snapshot,
        candidates=updated_candidates,
        selected_candidate_id=candidate_id,
        arbitration_reason="manual_selection",
    )


class DriverServiceBackendAdapter:
    """Adapter that treats the existing DriverService as a single backend."""

    def __init__(
        self,
        driver: DriverService,
        *,
        backend_id: str = "default_driver",
        backend_type: str = "grpc_driver_service",
        model_type_override: str | None = None,
        supports_parallel: bool = False,
        supports_hot_switch: bool = True,
        priority: int = 0,
        observation_window_summary_getter: (
            Callable[[str, int], dict[str, Any]] | None
        ) = None,
        observation_window_summary_size: int = 0,
    ) -> None:
        self._driver = driver
        self._model_type_override = model_type_override
        self._observation_window_summary_getter = observation_window_summary_getter
        self._observation_window_summary_size = max(0, observation_window_summary_size)
        self.metadata = BackendMetadata(
            backend_id=backend_id,
            backend_type=backend_type,
            supports_parallel=supports_parallel,
            supports_hot_switch=supports_hot_switch,
            priority=priority,
        )

    async def infer(self, snapshot: DecisionSnapshot) -> CandidateDecision:
        candidate_id = f"{snapshot.input_snapshot_id}:{self.metadata.backend_id}:0"
        planner_context = dict(snapshot.planner_context or {})
        planner_context["decision_metadata"] = {
            "backend_id": self.metadata.backend_id,
            "candidate_id": candidate_id,
            "input_snapshot_id": snapshot.input_snapshot_id,
            "step_id": snapshot.step_id,
        }
        if (
            self._observation_window_summary_getter is not None
            and self._observation_window_summary_size > 0
        ):
            planner_context["decision_metadata"]["observation_window"] = (
                self._observation_window_summary_getter(
                    snapshot.input_snapshot_id,
                    self._observation_window_summary_size,
                )
            )
        if self._model_type_override is not None:
            self._driver.set_next_model_for_next_drive(self._model_type_override)
        self._driver.set_planner_context_for_next_drive(planner_context)
        trajectory = await self._driver.drive(
            time_now_us=snapshot.time_now_us,
            time_query_us=snapshot.time_query_us,
            renderer_data=snapshot.renderer_data,
        )
        return CandidateDecision(
            candidate_id=candidate_id,
            step_id=snapshot.step_id,
            input_snapshot_id=snapshot.input_snapshot_id,
            backend_id=self.metadata.backend_id,
            status=CandidateStatus.READY,
            trajectory=trajectory,
            diagnostics={
                "backend_type": self.metadata.backend_type,
                "input_snapshot_id": snapshot.input_snapshot_id,
                "model_type_override": self._model_type_override,
            },
            generated_at_us=snapshot.time_now_us,
        )

    def capture_backend_state(self) -> Any | None:
        return None

    async def restore_backend_state(self, state: Any) -> None:
        del state
        raise RuntimeError(
            f"Backend {self.metadata.backend_id} does not support restore"
        )


class SingleBackendDriverOrchestrator:
    """Compatibility orchestrator preserving existing single-backend behavior."""

    def __init__(self, backend: DriverBackendAdapter) -> None:
        self._backend = backend

    @property
    def backend_ids(self) -> list[str]:
        return [self._backend.metadata.backend_id]

    async def generate_candidates(
        self,
        snapshot: DecisionSnapshot,
        backend_ids: Sequence[str] | None = None,
    ) -> DecisionBundle:
        if backend_ids and self._backend.metadata.backend_id not in set(backend_ids):
            raise RuntimeError(
                f"Requested backends {list(backend_ids)} exclude the only available backend "
                f"{self._backend.metadata.backend_id}"
            )
        candidate = await self._backend.infer(snapshot)
        return DecisionBundle(snapshot=snapshot, candidates=[candidate])

    async def select_candidate(self, bundle: DecisionBundle) -> CandidateDecision:
        if len(bundle.candidates) != 1:
            raise ValueError("Single-backend orchestrator expects exactly one candidate")
        candidate = bundle.candidates[0]
        if candidate.trajectory is None:
            raise RuntimeError("Single-backend candidate is missing a trajectory")
        return CandidateDecision(
            candidate_id=candidate.candidate_id,
            step_id=candidate.step_id,
            input_snapshot_id=candidate.input_snapshot_id,
            backend_id=candidate.backend_id,
            status=CandidateStatus.SELECTED,
            trajectory=candidate.trajectory,
            diagnostics=candidate.diagnostics,
            generated_at_us=candidate.generated_at_us,
            recompute_count=candidate.recompute_count,
            error=candidate.error,
        )

    async def recompute_candidate(
        self,
        bundle: DecisionBundle,
        backend_id: str,
    ) -> DecisionBundle:
        if backend_id != self._backend.metadata.backend_id:
            raise KeyError(f"Unknown backend_id: {backend_id}")
        recompute_count = _next_recompute_index(bundle, backend_id)
        candidate = await self._backend.infer(bundle.snapshot)
        candidate = replace(
            candidate,
            candidate_id=(
                f"{bundle.snapshot.input_snapshot_id}:{self._backend.metadata.backend_id}:"
                f"{recompute_count}"
            ),
            recompute_count=recompute_count,
        )
        return _merge_recomputed_candidate(bundle, candidate)

    def capture_backend_checkpoint(self) -> tuple[dict[str, Any], list[str]]:
        if not self._backend.metadata.supports_restore:
            return ({}, [self._backend.metadata.backend_id])
        return (
            {self._backend.metadata.backend_id: self._backend.capture_backend_state()},
            [],
        )

    async def restore_backend_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if not self._backend.metadata.supports_restore:
            return
        backend_id = self._backend.metadata.backend_id
        if backend_id not in checkpoint:
            raise RuntimeError(f"Missing backend checkpoint for {backend_id}")
        await self._backend.restore_backend_state(checkpoint[backend_id])


class MultiBackendDriverOrchestrator:
    """Multi-backend orchestrator with default priority-based arbitration."""

    def __init__(self, registry: DriverBackendRegistry) -> None:
        self._registry = registry

    @property
    def backend_ids(self) -> list[str]:
        return self._registry.backend_ids

    async def generate_candidates(
        self,
        snapshot: DecisionSnapshot,
        backend_ids: Sequence[str] | None = None,
    ) -> DecisionBundle:
        selected_backend_ids = set(backend_ids or self._registry.backend_ids)
        candidate_backends = [
            backend
            for backend in self._registry.backends
            if backend.metadata.backend_id in selected_backend_ids
        ]
        if not candidate_backends:
            raise RuntimeError(
                f"No registered backends match requested backend_ids={sorted(selected_backend_ids)}"
            )

        candidates: list[CandidateDecision] = []
        parallel_backends = [
            backend
            for backend in candidate_backends
            if backend.metadata.supports_parallel
        ]
        serial_backends = [
            backend
            for backend in candidate_backends
            if not backend.metadata.supports_parallel
        ]

        if parallel_backends:
            results = await asyncio.gather(
                *[backend.infer(snapshot) for backend in parallel_backends],
                return_exceptions=True,
            )
            for backend, result in zip(parallel_backends, results, strict=True):
                if isinstance(result, Exception):
                    candidates.append(
                        CandidateDecision(
                            candidate_id=f"{snapshot.input_snapshot_id}:{backend.metadata.backend_id}:0",
                            step_id=snapshot.step_id,
                            input_snapshot_id=snapshot.input_snapshot_id,
                            backend_id=backend.metadata.backend_id,
                            status=CandidateStatus.FAILED,
                            trajectory=None,
                            diagnostics={"backend_type": backend.metadata.backend_type},
                            generated_at_us=snapshot.time_now_us,
                            error=str(result),
                        )
                    )
                    continue
                candidates.append(result)

        for backend in serial_backends:
            try:
                result = await backend.infer(snapshot)
            except Exception as exc:  # noqa: BLE001
                result = exc
            if isinstance(result, Exception):
                candidates.append(
                    CandidateDecision(
                        candidate_id=f"{snapshot.input_snapshot_id}:{backend.metadata.backend_id}:0",
                        step_id=snapshot.step_id,
                        input_snapshot_id=snapshot.input_snapshot_id,
                        backend_id=backend.metadata.backend_id,
                        status=CandidateStatus.FAILED,
                        trajectory=None,
                        diagnostics={"backend_type": backend.metadata.backend_type},
                        generated_at_us=snapshot.time_now_us,
                        error=str(result),
                    )
                )
                continue
            candidates.append(result)
        return DecisionBundle(snapshot=snapshot, candidates=candidates)

    async def select_candidate(self, bundle: DecisionBundle) -> CandidateDecision:
        ready_candidates = [
            candidate
            for candidate in bundle.candidates
            if candidate.status in {CandidateStatus.READY, CandidateStatus.SELECTED}
            and candidate.trajectory is not None
        ]
        if not ready_candidates:
            raise RuntimeError(
                f"No usable candidates for input_snapshot_id={bundle.snapshot.input_snapshot_id}"
            )

        priority_by_backend = {
            backend.metadata.backend_id: backend.metadata.priority
            for backend in self._registry.backends
        }
        selected = min(
            ready_candidates,
            key=lambda candidate: priority_by_backend.get(candidate.backend_id, 1_000_000),
        )
        return CandidateDecision(
            candidate_id=selected.candidate_id,
            step_id=selected.step_id,
            input_snapshot_id=selected.input_snapshot_id,
            backend_id=selected.backend_id,
            status=CandidateStatus.SELECTED,
            trajectory=selected.trajectory,
            diagnostics=selected.diagnostics,
            generated_at_us=selected.generated_at_us,
            recompute_count=selected.recompute_count,
            error=selected.error,
        )

    async def recompute_candidate(
        self,
        bundle: DecisionBundle,
        backend_id: str,
    ) -> DecisionBundle:
        backend = self._registry.get_backend(backend_id)
        recompute_count = _next_recompute_index(bundle, backend_id)
        candidate = await backend.infer(bundle.snapshot)
        candidate = replace(
            candidate,
            candidate_id=f"{bundle.snapshot.input_snapshot_id}:{backend_id}:{recompute_count}",
            recompute_count=recompute_count,
        )
        return _merge_recomputed_candidate(bundle, candidate)

    def capture_backend_checkpoint(self) -> tuple[dict[str, Any], list[str]]:
        checkpoint: dict[str, Any] = {}
        unsupported_backend_ids: list[str] = []
        for backend in self._registry.backends:
            if not backend.metadata.supports_restore:
                unsupported_backend_ids.append(backend.metadata.backend_id)
                continue
            checkpoint[backend.metadata.backend_id] = backend.capture_backend_state()
        return (checkpoint, unsupported_backend_ids)

    async def restore_backend_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        for backend in self._registry.backends:
            if not backend.metadata.supports_restore:
                continue
            backend_id = backend.metadata.backend_id
            if backend_id not in checkpoint:
                raise RuntimeError(f"Missing backend checkpoint for {backend_id}")
            await backend.restore_backend_state(checkpoint[backend_id])
