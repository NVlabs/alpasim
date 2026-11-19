# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Lookup and timing validation for configured executable scenes."""

from __future__ import annotations

from alpasim_runtime.config import SceneConfig, SimulationConfig
from alpasim_runtime.data_source_loader import DataSourceInfo
from alpasim_utils.scene_timing import (
    SceneTimingRejectionReason,
    derive_clip_start_offset_us,
    evaluate_scene_timing,
)
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene, SceneKey


def scene_from_config(scene_config: SceneConfig) -> Scene:
    """Convert the runtime configuration projection to the canonical Scene."""
    return Scene(
        source=ClipDataSource(
            clip_id=scene_config.source.clip_id,
            usdz_uuid=scene_config.source.usdz_uuid,
        ),
        policy_start_us=scene_config.policy_start_us,  # type: ignore[arg-type]
    )


class SceneRegistry:
    """Own complete scenes indexed by scene key."""

    def __init__(
        self,
        scenes: list[SceneConfig],
        data_source_infos: list[DataSourceInfo],
        simulation_config: SimulationConfig,
    ) -> None:
        self._simulation_config = simulation_config
        self._scenes: dict[SceneKey, Scene] = {}
        self._data_source_info_by_id = {
            data_source.source.source_id: data_source
            for data_source in data_source_infos
        }

        for scene_config in scenes:
            scene = scene_from_config(scene_config)
            if scene.scene_key in self._scenes:
                raise ValueError(f"Duplicate scene key: {scene.scene_key!r}")
            data_source_info = self._data_source_info_by_id.get(scene.source.source_id)
            if data_source_info is None:
                raise ValueError(
                    f"Scene {scene.scene_key!r} references unknown data source "
                    f"{scene.source.source_id!r}"
                )
            if scene.source != data_source_info.source:
                raise ValueError(
                    f"Scene {scene.scene_key!r} configures {scene.source!r}, but "
                    f"the loaded descriptor is {data_source_info.source!r}"
                )
            self._validate_timing(scene, data_source_info)
            self._scenes[scene.scene_key] = scene

    def _validate_timing(self, scene: Scene, data_source_info: DataSourceInfo) -> None:
        duration_us = (
            0
            if scene.policy_start_us == T0_AUTO
            else int(data_source_info.metadata.time_range.end)
            - int(data_source_info.metadata.time_range.start)
        )
        evaluation = evaluate_scene_timing(
            policy_start_us=scene.policy_start_us,
            force_gt_duration_us=self._simulation_config.force_gt_duration_us,
            n_sim_steps=self._simulation_config.n_sim_steps,
            control_timestep_us=self._simulation_config.control_timestep_us,
            clip_duration_us=duration_us,
        )
        if (
            evaluation.rejection_reason
            == SceneTimingRejectionReason.INSUFFICIENT_HISTORY
        ):
            raise ValueError(
                f"policy_start_us={scene.policy_start_us} does not provide "
                f"force_gt_duration_us={self._simulation_config.force_gt_duration_us} "
                "of history"
            )
        if (
            evaluation.rejection_reason
            == SceneTimingRejectionReason.INSUFFICIENT_HORIZON
        ):
            raise ValueError(
                f"Scene {scene.scene_key!r}: full horizon ends at recording-relative "
                f"{evaluation.requested_end_us} us, after clip duration "
                f"{duration_us} us"
            )
        if evaluation.rejection_reason is not None:
            raise ValueError(
                "Unhandled scene timing rejection reason: "
                f"{evaluation.rejection_reason}"
            )

    @property
    def scenes(self) -> list[Scene]:
        return list(self._scenes.values())

    def resolve(self, requested: Scene) -> Scene:
        scene_key = requested.scene_key
        configured = self._scenes[scene_key]
        if configured.source != requested.source:
            raise ValueError(
                f"Scene {scene_key!r} requests {requested.source!r}, but the "
                f"configured descriptor is {configured.source!r}"
            )
        return configured

    def resolve_fork(self, requested: Scene) -> Scene:
        """Resolve a request-time fork against a configured data source.

        Ordinary scenes must match a configured scene key. Forks select their
        policy start at request time, so only the data-source descriptor must
        match a configured source. A fork needs the force-GT history before its
        policy start and one control step after it, but not the full configured
        horizon: its rollout stops at the clip end when that comes first.
        """
        data_source_info = self._data_source_info_by_id.get(requested.source.source_id)
        if data_source_info is None:
            raise KeyError(requested.scene_key)
        if requested.source != data_source_info.source:
            raise ValueError(
                f"Fork requests {requested.source!r}, but the configured "
                f"descriptor is {data_source_info.source!r}"
            )

        if requested.policy_start_us == T0_AUTO:
            raise ValueError("Fork request requires an explicit policy_start_us")
        derive_clip_start_offset_us(
            requested.policy_start_us,
            self._simulation_config.force_gt_duration_us,
        )
        clip_duration_us = int(data_source_info.metadata.time_range.end) - int(
            data_source_info.metadata.time_range.start
        )
        if (
            requested.policy_start_us + self._simulation_config.control_timestep_us
            > clip_duration_us
        ):
            raise ValueError(
                f"Fork {requested.scene_key!r}: no control step fits between the "
                f"policy start and the clip duration {clip_duration_us} us"
            )
        return requested
