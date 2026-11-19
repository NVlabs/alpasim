# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Shared data types used across alpasim packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from alpasim_grpc.v0 import runtime_pb2  # type: ignore[import-not-found]

T0_AUTO: Literal["AUTO"] = "AUTO"
SceneKey: TypeAlias = str


class DataSource(Protocol):
    """Descriptor for data used to construct a scene."""

    @property
    def source_id(self) -> str:
        """Return the stable identifier used to address the source."""
        ...


@dataclass(frozen=True)
class ClipDataSource(DataSource):
    """A DataSource that identifies a clip and pins its USDZ artifact."""

    clip_id: str
    usdz_uuid: str

    def __post_init__(self) -> None:
        _validate_source_id(self.clip_id, field_name="clip_id")
        if not self.usdz_uuid:
            raise ValueError("usdz_uuid must not be empty")

    @property
    def source_id(self) -> str:
        """Return the clip identifier."""
        return self.clip_id


def _validate_t0_selector(value: int | Literal["AUTO"], *, field_name: str) -> None:
    if value != T0_AUTO and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{field_name} must be {T0_AUTO!r} or a non-negative integer")


def _validate_source_id(source_id: str, *, field_name: str = "source_id") -> None:
    if not source_id:
        raise ValueError(f"{field_name} must not be empty")
    if "@t0=" in source_id:
        raise ValueError(f"{field_name} must not contain the reserved '@t0=' suffix")
    if source_id in {".", ".."} or "/" in source_id or "\\" in source_id:
        raise ValueError(f"{field_name} must be one safe path component")


def parse_scene_key(scene_key: SceneKey) -> tuple[str, int | Literal["AUTO"]]:
    """Parse a scene key into its source ID and policy-start selector."""
    source_id, separator, raw_policy_start = scene_key.partition("@t0=")
    if not separator:
        raise ValueError(
            "scene key must have the form "
            "'<source-id>@t0=AUTO' or '<source-id>@t0=<policy-start-us>'"
        )
    _validate_source_id(source_id)
    if raw_policy_start == T0_AUTO:
        return source_id, T0_AUTO
    if (
        not raw_policy_start
        or not raw_policy_start.isascii()
        or not raw_policy_start.isdigit()
        or (len(raw_policy_start) > 1 and raw_policy_start.startswith("0"))
    ):
        raise ValueError(
            "explicit policy start must be a canonical unsigned base-10 integer"
        )
    policy_start_us = int(raw_policy_start)
    _validate_t0_selector(policy_start_us, field_name="policy_start_us")
    return source_id, policy_start_us


@dataclass(frozen=True)
class Scene:
    """A DataSource selected at a recording-relative policy start."""

    source: DataSource
    policy_start_us: int | Literal["AUTO"]

    def __post_init__(self) -> None:
        _validate_source_id(self.source.source_id)
        _validate_t0_selector(self.policy_start_us, field_name="policy_start_us")

    @property
    def scene_key(self) -> SceneKey:
        """Return the stable key used for lookup, deduplication, and paths."""
        return f"{self.source.source_id}@t0={self.policy_start_us}"


def data_source_to_proto(source: DataSource) -> runtime_pb2.DataSource:
    if isinstance(source, ClipDataSource):
        return runtime_pb2.DataSource(
            clip=runtime_pb2.ClipDataSource(
                clip_id=source.clip_id,
                usdz_uuid=source.usdz_uuid,
            )
        )

    raise TypeError(f"Unsupported data source type: {type(source).__name__}")


def scene_to_proto(scene: Scene) -> runtime_pb2.SceneInfo:
    info = runtime_pb2.SceneInfo(source=data_source_to_proto(scene.source))
    if scene.policy_start_us != T0_AUTO:
        info.policy_start_us = scene.policy_start_us
    return info


def scene_from_proto(info: runtime_pb2.SceneInfo) -> Scene:
    if info.source.WhichOneof("kind") != "clip":
        raise ValueError("Scene data source is not set")
    return Scene(
        source=ClipDataSource(
            clip_id=info.source.clip.clip_id,
            usdz_uuid=info.source.clip.usdz_uuid,
        ),
        policy_start_us=(
            info.policy_start_us if info.HasField("policy_start_us") else T0_AUTO
        ),
    )


@dataclass
class ImageWithMetadata:
    """Metadata for a camera image captured during simulation.

    This class is defined in alpasim_utils to avoid circular dependencies
    between runtime and eval packages.
    """

    start_timestamp_us: int
    end_timestamp_us: int
    image_bytes: bytes
    camera_logical_id: str

    def __repr__(self) -> str:
        return (
            "ImageWithMetadata("
            f"start_timestamp_us={self.start_timestamp_us:_d}, "
            f"end_timestamp_us={self.end_timestamp_us:_d}, "
            f"camera_logical_id={self.camera_logical_id}, "
            f"len(image_bytes)={len(self.image_bytes)})"
        )
