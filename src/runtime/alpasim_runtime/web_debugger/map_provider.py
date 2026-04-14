# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import math
import threading
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alpasim_utils.artifact import Artifact

try:
    from trajdata import maps
except ImportError:  # pragma: no cover - handled at runtime based on env
    maps = None


@dataclass(frozen=True)
class SceneMapPayload:
    scene_id: str
    bounds: dict[str, float]
    layers: dict[str, list[list[list[float]]]]
    source: str = "empty"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "bounds": self.bounds,
            "layers": self.layers,
            "source": self.source,
        }


def default_usdz_glob() -> str:
    repo_root = Path(__file__).resolve().parents[4]
    return str(repo_root / "data/nre-artifacts/all-usdzs/**/*.usdz")


class SceneMapProvider:
    def __init__(self, usdz_glob: str):
        self._usdz_glob = usdz_glob
        self._lock = threading.Lock()
        self._artifacts: dict[str, Artifact] | None = None
        self._payloads: dict[str, SceneMapPayload] = {}

    def list_scene_ids(self) -> list[str]:
        with self._lock:
            if self._artifacts is None:
                self._artifacts = Artifact.discover_from_glob(self._usdz_glob)
            return sorted(self._artifacts.keys())

    def get_scene_map(self, scene_id: str) -> SceneMapPayload:
        with self._lock:
            if scene_id in self._payloads:
                return self._payloads[scene_id]

            if self._artifacts is None:
                self._artifacts = Artifact.discover_from_glob(self._usdz_glob)

            if scene_id not in self._artifacts:
                raise KeyError(f"Unknown scene_id: {scene_id}")

            payload = self._build_payload(self._artifacts[scene_id])
            self._payloads[scene_id] = payload
            return payload

    def _build_payload(self, artifact: Artifact) -> SceneMapPayload:
        if maps is not None and artifact.map is not None:
            return _build_payload_from_vec_map(artifact)

        xodr_payload = self._build_payload_from_xodr(artifact)
        if xodr_payload is not None:
            return xodr_payload

        return SceneMapPayload(
            scene_id=artifact.scene_id,
            bounds={"min_x": -50.0, "max_x": 50.0, "min_y": -50.0, "max_y": 50.0},
            layers={},
            source="empty",
        )

    def _build_payload_from_xodr(self, artifact: Artifact) -> SceneMapPayload | None:
        try:
            with zipfile.ZipFile(artifact.source, "r") as zip_file:
                if "map.xodr" not in zip_file.namelist():
                    return None

                xodr_xml = zip_file.read("map.xodr").decode("utf-8")
                root = ET.fromstring(xodr_xml)
                transform = self._load_xodr_transform(artifact, zip_file, xodr_xml)
        except (zipfile.BadZipFile, ET.ParseError):
            return None

        layers: dict[str, list[list[list[float]]]] = {
            "road_lane_center": [],
            "road_lane_left_edge": [],
            "road_lane_right_edge": [],
            "road_edge": [],
            "stop_line": [],
            "other_line": [],
        }
        bounds = {
            "min_x": float("inf"),
            "max_x": float("-inf"),
            "min_y": float("inf"),
            "max_y": float("-inf"),
        }

        def _append(layer_name: str, points: list[list[float]]) -> None:
            normalized = _normalize_points(points)
            if len(normalized) < 2:
                return
            layers[layer_name].append(normalized)
            for x, y in normalized:
                bounds["min_x"] = min(bounds["min_x"], x)
                bounds["max_x"] = max(bounds["max_x"], x)
                bounds["min_y"] = min(bounds["min_y"], y)
                bounds["max_y"] = max(bounds["max_y"], y)

        for road in root.findall("road"):
            road_layers = _build_xodr_road_layers(road, transform)
            for layer_name, polylines in road_layers.items():
                for polyline in polylines:
                    _append(layer_name, polyline)

        if not np.isfinite(list(bounds.values())).all():
            return None

        pad = 8.0
        return SceneMapPayload(
            scene_id=artifact.scene_id,
            bounds={
                "min_x": bounds["min_x"] - pad,
                "max_x": bounds["max_x"] + pad,
                "min_y": bounds["min_y"] - pad,
                "max_y": bounds["max_y"] + pad,
            },
            layers={k: v for k, v in layers.items() if v},
            source="xodr",
        )

    @staticmethod
    def _load_xodr_transform(
        artifact: Artifact,
        zip_file: zipfile.ZipFile,
        xodr_xml: str,
    ) -> np.ndarray | None:
        try:
            return artifact._get_xodr_transform(zip_file, xodr_xml)  # noqa: SLF001
        except Exception:
            return None


def _build_payload_from_vec_map(artifact: Artifact) -> SceneMapPayload:
    assert maps is not None
    assert artifact.map is not None

    vec_map = artifact.map
    layers: dict[str, list[list[list[float]]]] = {
        "road_lane_center": [],
        "road_lane_left_edge": [],
        "road_lane_right_edge": [],
        "road_edge": [],
        "stop_line": [],
        "other_line": [],
    }
    bounds = {
        "min_x": float("inf"),
        "max_x": float("-inf"),
        "min_y": float("inf"),
        "max_y": float("-inf"),
    }

    def _append(name: str, xy: Any) -> None:
        points = _normalize_points(xy)
        if len(points) < 2:
            return
        layers[name].append(points)
        for x, y in points:
            bounds["min_x"] = min(bounds["min_x"], x)
            bounds["max_x"] = max(bounds["max_x"], x)
            bounds["min_y"] = min(bounds["min_y"], y)
            bounds["max_y"] = max(bounds["max_y"], y)

    road_lanes = vec_map.elements[maps.vec_map_elements.MapElementType.ROAD_LANE]
    road_edges = vec_map.elements[maps.vec_map_elements.MapElementType.ROAD_EDGE]
    wait_lines = vec_map.elements[maps.vec_map_elements.MapElementType.WAIT_LINE]

    for element in road_lanes.values():
        _append("road_lane_center", element.center.xy)
        if element.left_edge is not None:
            _append("road_lane_left_edge", element.left_edge.xy)
        if element.right_edge is not None:
            _append("road_lane_right_edge", element.right_edge.xy)

    for element in road_edges.values():
        _append("road_edge", element.polyline.xy)

    for element in wait_lines.values():
        layer_name = "stop_line" if element.wait_line_type == "STOP" else "other_line"
        _append(layer_name, element.polyline.xy)

    if not np.isfinite(list(bounds.values())).all():
        bounds = {"min_x": -50.0, "max_x": 50.0, "min_y": -50.0, "max_y": 50.0}
    else:
        pad = 8.0
        bounds = {
            "min_x": bounds["min_x"] - pad,
            "max_x": bounds["max_x"] + pad,
            "min_y": bounds["min_y"] - pad,
            "max_y": bounds["max_y"] + pad,
        }

    return SceneMapPayload(
        scene_id=artifact.scene_id,
        bounds=bounds,
        layers={k: v for k, v in layers.items() if v},
        source="xodr-vector-map",
    )


@dataclass(frozen=True)
class _PlanGeometry:
    s: float
    x: float
    y: float
    hdg: float
    length: float
    kind: str
    curvature: float = 0.0

    def sample(self, s_abs: float) -> tuple[float, float, float]:
        ds = min(max(s_abs - self.s, 0.0), self.length)
        if self.kind == "line":
            x = self.x + math.cos(self.hdg) * ds
            y = self.y + math.sin(self.hdg) * ds
            return (x, y, self.hdg)

        if self.kind == "arc":
            if abs(self.curvature) < 1e-9:
                x = self.x + math.cos(self.hdg) * ds
                y = self.y + math.sin(self.hdg) * ds
                return (x, y, self.hdg)

            radius = 1.0 / self.curvature
            center_x = self.x - math.sin(self.hdg) * radius
            center_y = self.y + math.cos(self.hdg) * radius
            start_angle = math.atan2(self.y - center_y, self.x - center_x)
            angle = start_angle + ds * self.curvature
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            return (x, y, self.hdg + ds * self.curvature)

        raise ValueError(f"Unsupported geometry kind: {self.kind}")


@dataclass(frozen=True)
class _Poly3:
    s: float
    a: float
    b: float
    c: float
    d: float

    def value_at(self, s_abs: float) -> float:
        ds = max(s_abs - self.s, 0.0)
        return self.a + self.b * ds + self.c * ds * ds + self.d * ds * ds * ds


@dataclass(frozen=True)
class _LaneDef:
    lane_id: int
    lane_type: str
    widths: list[_Poly3]

    def width_at(self, s_section: float, section_s: float) -> float:
        if not self.widths:
            return 0.0
        active = self.widths[0]
        s_local = max(s_section - section_s, 0.0)
        for width in self.widths:
            if width.s <= s_local:
                active = width
            else:
                break
        return max(active.value_at(s_local), 0.0)


def _build_xodr_road_layers(
    road: ET.Element,
    transform: np.ndarray | None,
) -> dict[str, list[list[list[float]]]]:
    plan_view = road.find("planView")
    lanes_elem = road.find("lanes")
    if plan_view is None or lanes_elem is None:
        return {}

    geometries = _parse_plan_geometries(plan_view)
    if not geometries:
        return {}

    lane_offsets = _parse_poly3_nodes(lanes_elem.findall("laneOffset"))
    lane_sections = lanes_elem.findall("laneSection")
    if not lane_sections:
        return {}

    road_length = float(road.attrib.get("length", "0") or 0.0)
    layers: dict[str, list[list[list[float]]]] = {
        "road_lane_center": [],
        "road_lane_left_edge": [],
        "road_lane_right_edge": [],
        "road_edge": [],
    }

    for index, lane_section in enumerate(lane_sections):
        section_s = float(lane_section.attrib.get("s", "0") or 0.0)
        next_s = (
            float(lane_sections[index + 1].attrib.get("s", "0") or 0.0)
            if index + 1 < len(lane_sections)
            else road_length
        )
        if next_s <= section_s:
            continue

        sample_s = _sample_range(section_s, next_s)
        ref_samples = [_sample_reference(geometries, s_abs) for s_abs in sample_s]
        lane_offset_values = [_eval_poly3(lane_offsets, s_abs) for s_abs in sample_s]

        left_lanes = _parse_lane_group(lane_section.find("left"), reverse=False)
        right_lanes = _parse_lane_group(lane_section.find("right"), reverse=True)

        road_left, road_right = [], []
        outer_left_offsets = lane_offset_values.copy()
        outer_right_offsets = lane_offset_values.copy()

        for lane in left_lanes:
            center, left_edge, right_edge = _build_lane_polylines(
                lane=lane,
                section_s=section_s,
                sample_s=sample_s,
                ref_samples=ref_samples,
                inner_offsets=outer_left_offsets,
                side=1.0,
                transform=transform,
            )
            if center:
                layers["road_lane_center"].append(center)
                layers["road_lane_left_edge"].append(left_edge)
                layers["road_lane_right_edge"].append(right_edge)
                outer_left_offsets = [
                    inner + lane.width_at(s_abs, section_s)
                    for inner, s_abs in zip(outer_left_offsets, sample_s, strict=True)
                ]
        road_left = [
            _offset_point(sample, offset, transform)
            for sample, offset in zip(ref_samples, outer_left_offsets, strict=True)
        ]

        for lane in right_lanes:
            center, left_edge, right_edge = _build_lane_polylines(
                lane=lane,
                section_s=section_s,
                sample_s=sample_s,
                ref_samples=ref_samples,
                inner_offsets=outer_right_offsets,
                side=-1.0,
                transform=transform,
            )
            if center:
                layers["road_lane_center"].append(center)
                layers["road_lane_left_edge"].append(left_edge)
                layers["road_lane_right_edge"].append(right_edge)
                outer_right_offsets = [
                    inner - lane.width_at(s_abs, section_s)
                    for inner, s_abs in zip(outer_right_offsets, sample_s, strict=True)
                ]
        road_right = [
            _offset_point(sample, offset, transform)
            for sample, offset in zip(ref_samples, outer_right_offsets, strict=True)
        ]

        if len(road_left) >= 2:
            layers["road_edge"].append(road_left)
        if len(road_right) >= 2:
            layers["road_edge"].append(road_right)

    return layers


def _parse_plan_geometries(plan_view: ET.Element) -> list[_PlanGeometry]:
    geometries: list[_PlanGeometry] = []
    for geometry in plan_view.findall("geometry"):
        children = list(geometry)
        if not children:
            continue
        child = children[0]
        if child.tag not in {"line", "arc"}:
            continue
        geometries.append(
            _PlanGeometry(
                s=float(geometry.attrib.get("s", "0") or 0.0),
                x=float(geometry.attrib.get("x", "0") or 0.0),
                y=float(geometry.attrib.get("y", "0") or 0.0),
                hdg=float(geometry.attrib.get("hdg", "0") or 0.0),
                length=float(geometry.attrib.get("length", "0") or 0.0),
                kind=child.tag,
                curvature=float(child.attrib.get("curvature", "0") or 0.0),
            )
        )
    geometries.sort(key=lambda item: item.s)
    return geometries


def _parse_poly3_nodes(nodes: list[ET.Element]) -> list[_Poly3]:
    entries = [
        _Poly3(
            s=float(node.attrib.get("s", "0") or 0.0),
            a=float(node.attrib.get("a", "0") or 0.0),
            b=float(node.attrib.get("b", "0") or 0.0),
            c=float(node.attrib.get("c", "0") or 0.0),
            d=float(node.attrib.get("d", "0") or 0.0),
        )
        for node in nodes
    ]
    entries.sort(key=lambda item: item.s)
    return entries


def _parse_lane_group(group: ET.Element | None, reverse: bool) -> list[_LaneDef]:
    if group is None:
        return []
    lanes = []
    for lane in group.findall("lane"):
        widths = [
            _Poly3(
                s=float(width.attrib.get("sOffset", "0") or 0.0),
                a=float(width.attrib.get("a", "0") or 0.0),
                b=float(width.attrib.get("b", "0") or 0.0),
                c=float(width.attrib.get("c", "0") or 0.0),
                d=float(width.attrib.get("d", "0") or 0.0),
            )
            for width in lane.findall("width")
        ]
        widths.sort(key=lambda item: item.s)
        lanes.append(
            _LaneDef(
                lane_id=int(lane.attrib.get("id", "0") or 0),
                lane_type=lane.attrib.get("type", "none"),
                widths=widths,
            )
        )

    lanes.sort(key=lambda item: item.lane_id, reverse=reverse)
    return [lane for lane in lanes if lane.lane_id != 0]


def _sample_range(start_s: float, end_s: float) -> list[float]:
    length = max(end_s - start_s, 0.0)
    if length <= 0.0:
        return [start_s]
    target_step = 2.0
    num_points = max(12, min(96, int(math.ceil(length / target_step)) + 1))
    return np.linspace(start_s, end_s, num_points).tolist()


def _sample_reference(
    geometries: list[_PlanGeometry],
    s_abs: float,
) -> tuple[float, float, float]:
    active = geometries[0]
    for geometry in geometries:
        if geometry.s <= s_abs:
            active = geometry
        else:
            break
    return active.sample(s_abs)


def _eval_poly3(entries: list[_Poly3], s_abs: float) -> float:
    if not entries:
        return 0.0
    active = entries[0]
    for entry in entries:
        if entry.s <= s_abs:
            active = entry
        else:
            break
    return active.value_at(s_abs)


def _build_lane_polylines(
    lane: _LaneDef,
    section_s: float,
    sample_s: list[float],
    ref_samples: list[tuple[float, float, float]],
    inner_offsets: list[float],
    side: float,
    transform: np.ndarray | None,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    centers: list[list[float]] = []
    left_edge: list[list[float]] = []
    right_edge: list[list[float]] = []

    for s_abs, ref_sample, inner_offset in zip(
        sample_s, ref_samples, inner_offsets, strict=True
    ):
        width = lane.width_at(s_abs, section_s)
        if width <= 1e-3:
            continue
        if side > 0:
            right_offset = inner_offset
            left_offset = inner_offset + width
        else:
            left_offset = inner_offset
            right_offset = inner_offset - width
        center_offset = 0.5 * (left_offset + right_offset)
        centers.append(_offset_point(ref_sample, center_offset, transform))
        left_edge.append(_offset_point(ref_sample, left_offset, transform))
        right_edge.append(_offset_point(ref_sample, right_offset, transform))

    return centers, left_edge, right_edge


def _offset_point(
    ref_sample: tuple[float, float, float],
    lateral_offset: float,
    transform: np.ndarray | None,
) -> list[float]:
    x, y, hdg = ref_sample
    point = np.array(
        [
            x - math.sin(hdg) * lateral_offset,
            y + math.cos(hdg) * lateral_offset,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )
    if transform is not None:
        point = transform @ point
    return [float(point[0]), float(point[1])]


def _normalize_points(xy: Any) -> list[list[float]]:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2:
        return []
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[1] > 2:
        arr = arr[:, :2]
    if arr.shape[1] != 2:
        return []

    original = arr.copy()
    if len(arr) > 160:
        stride = max(1, len(arr) // 160)
        arr = arr[::stride]
        if not np.array_equal(arr[-1], original[-1]):
            arr = np.vstack([arr, original[-1]])

    return [[round(float(point[0]), 3), round(float(point[1]), 3)] for point in arr]
