# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Builds the render-config signature used to key the force-GT frame cache.

The signature captures every input that changes the deterministic force-GT
render for a scene so the globally shared cache never serves a frame produced
under a different configuration. It is computed once per rollout and used as the
``render_signature`` subfolder under the per-USDZ cache directory.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from alpasim_grpc.v0.sensorsim_pb2 import ImageFormat
from alpasim_runtime.camera_catalog import CameraDefinition
from alpasim_runtime.config import PhysicsUpdateMode
from alpasim_runtime.unbound_rollout import UnboundRollout


def build_force_gt_render_signature(
    unbound: "UnboundRollout",
    camera_definitions: Mapping[str, "CameraDefinition"],
    smooth_trajectories: bool | None,
) -> str:
    """Return the cache subfolder name encoding the render configuration.

    The result is ``<extension>-<hash>`` where ``<hash>`` is a digest of the
    full canonical payload (the authoritative key). Camera resolution and
    intrinsics are captured inside the hash via the serialized camera specs.
    """
    physics_mode = unbound.physics_update_mode
    physics_name = (
        physics_mode.name
        if isinstance(physics_mode, PhysicsUpdateMode)
        else str(physics_mode)
    )

    # Renderer build identity (probed get_version RPC, stored on version_ids).
    # A different NRE build/image never reuses another build's cached frames.
    # Launch flags are covered separately by ``force_gt_frame_cache_extra_key``.
    version_ids = getattr(unbound, "version_ids", None)
    sensorsim_version = getattr(version_ids, "sensorsim_version", None)
    if sensorsim_version is None:
        renderer_version: dict = {}
    else:
        api = sensorsim_version.grpc_api_version
        renderer_version = {
            "version_id": sensorsim_version.version_id,
            "git_hash": sensorsim_version.git_hash,
            "grpc_api_version": [api.major, api.minor, api.patch],
        }

    # Vehicle geometry that influences force-GT poses.
    vehicle = unbound.vehicle_config
    vehicle_fields = (
        "aabb_x_m",
        "aabb_y_m",
        "aabb_z_m",
        "aabb_x_offset_m",
        "aabb_y_offset_m",
        "aabb_z_offset_m",
    )

    payload = {
        "image_format": int(unbound.image_format),
        # Ego masking is keyed by its rig-config id, not by hashing the mask
        # asset bytes: like the renderer build (keyed by version id above), mask
        # assets are treated as immutable for a given id. Editing a mask in place
        # without changing its id will not bust the cache; bump
        # ``force_gt_frame_cache_extra_key`` (or the id) if that ever happens.
        "ego_mask_rig_config_id": unbound.ego_mask_rig_config_id,
        "physics_update_mode": physics_name,
        "force_gt_duration_us": int(unbound.force_gt_duration_us),
        "control_timestep_us": int(unbound.control_timestep_us),
        "render_start_timestamp_us": int(unbound.render_start_timestamp_us),
        "smooth_trajectories": smooth_trajectories,
        "vehicle": {
            field: float(getattr(vehicle, field))
            for field in vehicle_fields
            if hasattr(vehicle, field)
        },
        "cameras": {
            logical_id: {
                "logical_id": definition.logical_id,
                "intrinsics": definition.intrinsics.SerializeToString(
                    deterministic=True
                ).hex(),
                "rig_to_camera_translation": [
                    float(v) for v in definition.rig_to_camera.vec3
                ],
                "rig_to_camera_rotation": [
                    float(v) for v in definition.rig_to_camera.quat
                ],
            }
            for logical_id, definition in sorted(camera_definitions.items())
        },
        "renderer_version": renderer_version,
        "extra_key": getattr(unbound, "force_gt_frame_cache_extra_key", None),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{ImageFormat.Name(int(unbound.image_format)).lower()}-{digest}"
