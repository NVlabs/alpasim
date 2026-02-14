# SPDX-License-Identifier: Apache-2.0

"""Runtime-facing catalog for camera definitions."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_runtime.config import (
    CameraDefinitionConfig,
    FthetaConfig,
    OpenCVFisheyeConfig,
    OpenCVPinholeConfig,
)
from alpasim_utils.qvec import QVec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraDefinition:
    """This class defines the cameras available in the scenario.

    - `logical_id` is the unique identifier for the camera. This is referenced
        by `RuntimeCamera`, which defines how and what camera to render.
    - `intrinsics` is the intrinsic parameters of the camera.
    - `rig_to_camera` is the pose of the camera in the rig.
    """

    logical_id: str
    intrinsics: sensorsim_pb2.CameraSpec
    rig_to_camera: QVec

    def copy(self) -> CameraDefinition:
        """Return a deep copy of this camera definition."""
        spec = sensorsim_pb2.CameraSpec()
        spec.CopyFrom(self.intrinsics)
        return CameraDefinition(
            logical_id=self.logical_id,
            intrinsics=spec,
            rig_to_camera=self.rig_to_camera.clone(),
        )

    @property
    def native_resolution_hw(self) -> tuple[int, int]:
        """Return the intrinsic sensor resolution as (height, width)."""
        return (self.intrinsics.resolution_h, self.intrinsics.resolution_w)

    def as_proto(self) -> sensorsim_pb2.AvailableCamerasReturn.AvailableCamera:
        """Materialise the definition as an `AvailableCamera` protobuf."""
        proto = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
            logical_id=self.logical_id,
        )
        proto.intrinsics.CopyFrom(self.intrinsics)
        proto.rig_to_camera.CopyFrom(self.rig_to_camera.as_grpc_pose())
        return proto

    @staticmethod
    def _pose_from_config(
        translation_m: tuple[float, float, float],
        rotation_xyzw: tuple[float, float, float, float],
    ) -> QVec:
        """Validate the pose from config and convert it into a `QVec`."""
        translation = np.asarray(translation_m, dtype=np.float64)
        quat = np.asarray(rotation_xyzw, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError("rig_to_camera.rotation_xyzw must be a unit quaternion")
        return QVec(vec3=translation, quat=quat)

    @staticmethod
    def _apply_pinhole_intrinsics(
        spec: sensorsim_pb2.CameraSpec,
        config: OpenCVPinholeConfig | None,
    ) -> None:
        """Populate `spec` with OpenCV pinhole parameters from config."""
        if config is None:
            raise ValueError(
                "opencv_pinhole intrinsics must be provided for model=opencv_pinhole"
            )

        pinhole = spec.opencv_pinhole_param
        pinhole.principal_point_x = config.principal_point[0]
        pinhole.principal_point_y = config.principal_point[1]
        pinhole.focal_length_x = config.focal_length[0]
        pinhole.focal_length_y = config.focal_length[1]

        for [field, expected_len] in [
            ("radial", 6),
            ("tangential", 2),
            ("thin_prism", 4),
        ]:
            if len(getattr(config, field)) != expected_len:
                raise ValueError(
                    f"opencv_pinhole.{field} must provide exactly {expected_len} "
                    f"coefficients, got {len(getattr(config, field))}"
                )

        pinhole.radial_coeffs[:] = config.radial
        pinhole.tangential_coeffs[:] = config.tangential
        pinhole.thin_prism_coeffs[:] = config.thin_prism

    @staticmethod
    def _apply_fisheye_intrinsics(
        spec: sensorsim_pb2.CameraSpec,
        config: OpenCVFisheyeConfig | None,
    ) -> None:
        """Populate `spec` with OpenCV fisheye parameters from config."""
        if config is None:
            raise ValueError(
                "opencv_fisheye intrinsics must be provided for model=opencv_fisheye"
            )

        fisheye = spec.opencv_fisheye_param
        fisheye.principal_point_x = config.principal_point[0]
        fisheye.principal_point_y = config.principal_point[1]
        fisheye.focal_length_x = config.focal_length[0]
        fisheye.focal_length_y = config.focal_length[1]
        if len(config.radial) != 4:
            raise ValueError(
                "opencv_fisheye.radial must provide exactly 4 coefficients, got "
                f"{len(config.radial)}"
            )
        fisheye.radial_coeffs[:] = config.radial
        if config.max_angle is not None:
            fisheye.max_angle = config.max_angle

    @staticmethod
    def _apply_ftheta_intrinsics(
        spec: sensorsim_pb2.CameraSpec,
        config: FthetaConfig | None,
    ) -> None:
        """Populate `spec` with f-theta camera parameters from config."""
        if config is None:
            raise ValueError("ftheta intrinsics must be provided for model=ftheta")

        ftheta = spec.ftheta_param
        ftheta.principal_point_x = config.principal_point[0]
        ftheta.principal_point_y = config.principal_point[1]

        mapping = {
            "pixel_to_angle": sensorsim_pb2.FthetaCameraParam.PolynomialType.PIXELDIST_TO_ANGLE,
            "angle_to_pixel": sensorsim_pb2.FthetaCameraParam.PolynomialType.ANGLE_TO_PIXELDIST,
        }
        ftheta.reference_poly = mapping[config.reference_poly]

        ftheta.pixeldist_to_angle_poly.extend(config.pixeldist_to_angle)
        ftheta.angle_to_pixeldist_poly.extend(config.angle_to_pixeldist)
        if config.max_angle is not None:
            ftheta.max_angle = config.max_angle
        if config.linear_cde is not None:
            linear = ftheta.linear_cde
            linear.linear_c = config.linear_cde.linear_c
            linear.linear_d = config.linear_cde.linear_d
            linear.linear_e = config.linear_cde.linear_e

    @classmethod
    def _build_camera_spec(
        cls, cfg: CameraDefinitionConfig
    ) -> sensorsim_pb2.CameraSpec:
        """Construct a `CameraSpec` proto from a definition config."""
        spec = sensorsim_pb2.CameraSpec(
            logical_id=cfg.logical_id,
            resolution_h=cfg.resolution_hw[0],
            resolution_w=cfg.resolution_hw[1],
        )

        try:
            spec.shutter_type = sensorsim_pb2.ShutterType.Value(cfg.shutter_type)
        except ValueError as exc:
            valid_values = ", ".join(sensorsim_pb2.ShutterType.keys())
            raise ValueError(
                f"Unsupported shutter_type '{cfg.shutter_type}'."
                f" Expected one of: {valid_values}."
            ) from exc

        if cfg.intrinsics.model == "opencv_pinhole":
            cls._apply_pinhole_intrinsics(spec, cfg.intrinsics.opencv_pinhole)
        elif cfg.intrinsics.model == "opencv_fisheye":
            cls._apply_fisheye_intrinsics(spec, cfg.intrinsics.opencv_fisheye)
        elif cfg.intrinsics.model == "ftheta":
            cls._apply_ftheta_intrinsics(spec, cfg.intrinsics.ftheta)
        else:
            raise ValueError(f"Unsupported intrinsics model {cfg.intrinsics.model}")

        return spec

    @classmethod
    def from_config(cls, cfg: CameraDefinitionConfig) -> CameraDefinition:
        """Create a definition from a local config entry."""
        rig_to_camera = cls._pose_from_config(
            cfg.rig_to_camera.translation_m,
            cfg.rig_to_camera.rotation_xyzw,
        )
        camera_spec = cls._build_camera_spec(cfg)
        return cls(
            logical_id=cfg.logical_id,
            intrinsics=camera_spec,
            rig_to_camera=rig_to_camera,
        )

    @classmethod
    def from_proto(
        cls, available_camera: sensorsim_pb2.AvailableCamerasReturn.AvailableCamera
    ) -> CameraDefinition:
        """Clone a definition from an `AvailableCamera` protobuf."""
        rig_to_camera = QVec.from_grpc_pose(available_camera.rig_to_camera)
        spec = sensorsim_pb2.CameraSpec()
        spec.CopyFrom(available_camera.intrinsics)
        return cls(
            logical_id=available_camera.logical_id,
            intrinsics=spec,
            rig_to_camera=rig_to_camera,
        )


class CameraCatalog:
    """Central catalogue for sensorsim and locally-defined cameras."""

    def __init__(self, config_defs: Iterable[CameraDefinitionConfig] | None):
        """Initialize the catalog with optional locally configured overrides.

        Args:
            config_defs: Partial camera definition configs. Only `logical_id` is
                required; other fields (rig_to_camera, intrinsics, resolution_hw,
                shutter_type) are optional and will override sensorsim values
                when provided.
        """
        self._local_overrides: Dict[str, CameraDefinitionConfig] = {}
        for cfg in config_defs or []:
            if cfg.logical_id in self._local_overrides:
                raise ValueError(
                    f"Duplicate local camera definition for {cfg.logical_id}"
                )
            self._local_overrides[cfg.logical_id] = cfg

        self._scene_definitions: Dict[str, Dict[str, CameraDefinition]] = {}
        self._scene_locks: Dict[str, asyncio.Lock] = {}

    def _apply_override(
        self,
        base: CameraDefinition,
        override: CameraDefinitionConfig,
    ) -> CameraDefinition:
        """Apply partial config override to a base camera definition.

        Each field in the override is applied only if it is not None.
        This enables partial overrides (e.g. only rig_to_camera).
        """
        logical_id = base.logical_id

        # Start with copies of the base values
        intrinsics = sensorsim_pb2.CameraSpec()
        intrinsics.CopyFrom(base.intrinsics)
        rig_to_camera = base.rig_to_camera.clone()

        # Apply rig_to_camera override if provided
        if override.rig_to_camera is not None:
            old_translation = tuple(base.rig_to_camera.vec3)
            old_rotation = tuple(base.rig_to_camera.quat)
            new_translation = override.rig_to_camera.translation_m
            new_rotation = override.rig_to_camera.rotation_xyzw
            logger.info(
                "[%s] Overriding rig_to_camera: "
                "translation_m %s -> %s, rotation_xyzw %s -> %s",
                logical_id,
                old_translation,
                new_translation,
                old_rotation,
                new_rotation,
            )
            rig_to_camera = CameraDefinition._pose_from_config(
                new_translation,
                new_rotation,
            )

        # Apply resolution override if provided
        if override.resolution_hw is not None:
            old_resolution = (intrinsics.resolution_h, intrinsics.resolution_w)
            new_resolution = override.resolution_hw
            logger.info(
                "[%s] Overriding resolution_hw: %s -> %s",
                logical_id,
                old_resolution,
                new_resolution,
            )
            intrinsics.resolution_h = new_resolution[0]
            intrinsics.resolution_w = new_resolution[1]

        # Apply shutter_type override if provided
        if override.shutter_type is not None:
            old_shutter = sensorsim_pb2.ShutterType.Name(intrinsics.shutter_type)
            new_shutter = override.shutter_type
            logger.info(
                "[%s] Overriding shutter_type: %s -> %s",
                logical_id,
                old_shutter,
                new_shutter,
            )
            try:
                intrinsics.shutter_type = sensorsim_pb2.ShutterType.Value(new_shutter)
            except ValueError as exc:
                valid_values = ", ".join(sensorsim_pb2.ShutterType.keys())
                raise ValueError(
                    f"Unsupported shutter_type '{new_shutter}'."
                    f" Expected one of: {valid_values}."
                ) from exc

        # Apply intrinsics override if provided (replaces all intrinsic params)
        if override.intrinsics is not None:
            logger.info(
                "[%s] Overriding intrinsics with model=%s",
                logical_id,
                override.intrinsics.model,
            )
            # Clear existing intrinsics params and apply new ones
            intrinsics.ClearField("opencv_pinhole_param")
            intrinsics.ClearField("opencv_fisheye_param")
            intrinsics.ClearField("ftheta_param")

            if override.intrinsics.model == "opencv_pinhole":
                CameraDefinition._apply_pinhole_intrinsics(
                    intrinsics, override.intrinsics.opencv_pinhole
                )
            elif override.intrinsics.model == "opencv_fisheye":
                CameraDefinition._apply_fisheye_intrinsics(
                    intrinsics, override.intrinsics.opencv_fisheye
                )
            elif override.intrinsics.model == "ftheta":
                CameraDefinition._apply_ftheta_intrinsics(
                    intrinsics, override.intrinsics.ftheta
                )
            else:
                raise ValueError(
                    f"Unsupported intrinsics model {override.intrinsics.model}"
                )

        return CameraDefinition(
            logical_id=base.logical_id,
            intrinsics=intrinsics,
            rig_to_camera=rig_to_camera,
        )

    async def merge_local_and_sensorsim_cameras(
        self,
        scene_id: str,
        sensorsim_cameras: Iterable[
            sensorsim_pb2.AvailableCamerasReturn.AvailableCamera
        ],
    ) -> None:
        """Ensure `scene_id` has merged definitions using provided sensorsim data."""
        if scene_id in self._scene_definitions:
            return

        lock = self._scene_locks.setdefault(scene_id, asyncio.Lock())
        async with lock:
            if scene_id not in self._scene_definitions:
                scene_defs: Dict[str, CameraDefinition] = {}
                for camera in sensorsim_cameras:
                    key = camera.logical_id
                    scene_defs[key] = CameraDefinition.from_proto(camera)

                # Check that all local overrides reference cameras available
                # in sensorsim (sensorsim only supports changing, not adding).
                missing_cameras = set(self._local_overrides.keys()) - set(
                    scene_defs.keys()
                )
                if missing_cameras:
                    raise ValueError(
                        f"Local camera definitions {missing_cameras} are not "
                        f"available in sensorsim for scene '{scene_id}'."
                    )

                # Apply local overrides field-by-field
                if self._local_overrides:
                    logger.info(
                        "Applying %d camera override(s) for scene '%s': %s",
                        len(self._local_overrides),
                        scene_id,
                        list(self._local_overrides.keys()),
                    )
                for logical_id, override in self._local_overrides.items():
                    base = scene_defs[logical_id]
                    scene_defs[logical_id] = self._apply_override(base, override)

                self._scene_definitions[scene_id] = scene_defs

    def get_camera_definitions(self, scene_id: str) -> Dict[str, CameraDefinition]:
        """Return copy of the cached camera definitions for `scene_id`."""
        return {
            key: defn.copy() for key, defn in self._scene_definitions[scene_id].items()
        }

    def get_camera_definition(self, scene_id: str, logical_id: str) -> CameraDefinition:
        """Return copy of cached camera definition for `logical_id`, `scene_id`."""
        return self._scene_definitions[scene_id][logical_id].copy()

    def ensure_camera_defined(self, scene_id: str, logical_id: str) -> None:
        """Raise if `logical_id` is unknown for `scene_id`."""
        if logical_id not in self._scene_definitions[scene_id]:
            raise KeyError(
                f"Camera '{logical_id}' is not defined for scene '{scene_id}'"
            )
    
    def get_local_override_cameras(self) -> list[CameraDefinitionConfig]:
        """Return list of locally configured camera overrides."""
        return list(self._local_overrides.values())


__all__ = ["CameraCatalog", "CameraDefinition"]
