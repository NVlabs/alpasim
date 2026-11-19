# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Resolve an authored scene selection into a config-feasible population."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import cast

from alpasim_utils.scene_timing import (
    SceneTimingRejectionReason,
    derive_clip_start_offset_us,
    evaluate_scene_timing,
)
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene

from .sceneset import USDZManager


@dataclass(frozen=True)
class SceneRejection:
    """One authored scene rejected by the exact simulation timing config."""

    scene: Scene
    reason: SceneTimingRejectionReason
    clip_duration_us: int | None
    requested_end_us: int | None


@dataclass(frozen=True)
class SceneMaterialization:
    """Accepted population and its config-specific rejection audit."""

    requested_count: int
    accepted_scenes: tuple[Scene, ...]
    rejections: tuple[SceneRejection, ...]
    force_gt_duration_us: int
    n_sim_steps: int
    control_timestep_us: int
    accepted_sha256: str

    @property
    def accepted_count(self) -> int:
        """Return the number of authored scenes retained for staging."""
        return len(self.accepted_scenes)

    @property
    def rejected_count(self) -> int:
        """Return the number of authored scenes rejected by timing validation."""
        return len(self.rejections)

    @property
    def invalid_percent(self) -> float:
        """Return the rejected percentage, or zero for an empty selection."""
        if self.requested_count == 0:
            return 0.0
        return 100.0 * self.rejected_count / self.requested_count

    @property
    def rejection_counts(self) -> dict[str, int]:
        """Return counts for every stable rejection reason, including zeroes."""
        counts = Counter(rejection.reason.value for rejection in self.rejections)
        return {
            reason.value: counts[reason.value] for reason in SceneTimingRejectionReason
        }


def _accepted_scene_sha256(scenes: list[Scene]) -> str:
    """Hash accepted scene identities, authored T0 values, and ordering."""
    payload = []
    for scene in scenes:
        source = cast(ClipDataSource, scene.source)
        payload.append(
            {
                "source": {
                    "clip_id": source.clip_id,
                    "usdz_uuid": source.usdz_uuid,
                },
                "policy_start_us": scene.policy_start_us,
            }
        )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def materialize_scene_selection(
    scenes: list[Scene],
    manager: USDZManager,
    *,
    force_gt_duration_us: int,
    n_sim_steps: int,
    control_timestep_us: int,
) -> SceneMaterialization:
    """Resolve authored scenes into the population feasible for one config.

    AUTO scenes bypass metadata validation. Explicit-T0 scenes without enough
    history are rejected locally; the remaining sources are passed together to
    ``USDZManager`` so artifact identity conflicts and recording bounds are
    validated before any scene is accepted.

    Raises:
        TypeError: A selected scene does not use a clip data source.
        ValueError: One USDZ UUID maps to conflicting clip identities.
        USDZQueryError: Metadata for a required USDZ artifact cannot be loaded.
    """
    history_rejections: dict[int, SceneRejection] = {}
    metadata_sources: list[ClipDataSource] = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene.source, ClipDataSource):
            raise TypeError(
                f"Unsupported scene data source: {type(scene.source).__name__}"
            )
        if scene.policy_start_us == T0_AUTO:
            continue
        try:
            derive_clip_start_offset_us(
                scene.policy_start_us,
                force_gt_duration_us,
            )
        except ValueError:
            history_rejections[index] = SceneRejection(
                scene=scene,
                reason=SceneTimingRejectionReason.INSUFFICIENT_HISTORY,
                clip_duration_us=None,
                requested_end_us=None,
            )
            continue
        metadata_sources.append(scene.source)

    metadata_by_uuid = manager.load_timing_metadata(metadata_sources)
    accepted_scenes: list[Scene] = []
    rejections: list[SceneRejection] = []
    for index, scene in enumerate(scenes):
        history_rejection = history_rejections.get(index)
        if history_rejection is not None:
            rejections.append(history_rejection)
            continue
        if scene.policy_start_us == T0_AUTO:
            accepted_scenes.append(scene)
            continue

        source = cast(ClipDataSource, scene.source)
        metadata = metadata_by_uuid[source.usdz_uuid]
        evaluation = evaluate_scene_timing(
            policy_start_us=scene.policy_start_us,
            force_gt_duration_us=force_gt_duration_us,
            n_sim_steps=n_sim_steps,
            control_timestep_us=control_timestep_us,
            clip_duration_us=metadata.duration_us,
        )
        if evaluation.rejection_reason is None:
            accepted_scenes.append(scene)
        else:
            rejections.append(
                SceneRejection(
                    scene=scene,
                    reason=evaluation.rejection_reason,
                    clip_duration_us=metadata.duration_us,
                    requested_end_us=evaluation.requested_end_us,
                )
            )

    return SceneMaterialization(
        requested_count=len(scenes),
        accepted_scenes=tuple(accepted_scenes),
        rejections=tuple(rejections),
        force_gt_duration_us=force_gt_duration_us,
        n_sim_steps=n_sim_steps,
        control_timestep_us=control_timestep_us,
        accepted_sha256=_accepted_scene_sha256(accepted_scenes),
    )
