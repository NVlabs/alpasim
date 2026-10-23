# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Model-input logic copied from rsim and drivr so Alpasim feeds the policy
exactly what the CARLA harness and the car feed it.

Sources (keep in sync by hand; ``tests/test_rsim_port.py`` checks parity
against fixtures produced by the originals):

* rsim ``2d1c93d`` ``src/rsim/agents/onnx_decoder.py``: RoPE, graph
  validation, the host-owned KV-cache ring.
* rsim ``2d1c93d`` ``src/rsim/agents/onnx.py::_preprocess_image``: resize,
  crop and scale.
* rsim ``2d1c93d`` ``src/rsim/carla/waypoints.py``: route resampling.
* drivr ``715f85a`` ``src/drivr/io/defish.py``: fisheye undistortion, and
  ``config/camera_calibration.json`` (copied verbatim as ``cam-90-deg.json``).

The authoritative cache contract is rmind ``docs/decoder_only_kv_cache.md``
sections 3 and 7.

Frames: the model's ego frame is (x = right, y = forward) for both the
waypoint input and the trajectory output.  Alpasim's rig frame is
(x = forward, y = left).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as T

MASK_BIAS = -1e4
ROPE_BASE = 1000.0

CACHE_INPUT_NAMES = frozenset(
    {
        "inputs_past_k",
        "inputs_past_v",
        "inputs_cache_bias",
        "inputs_rope_cos",
        "inputs_rope_sin",
    }
)
SPEED_INPUT = "inputs_speed"
WAYPOINTS_INPUT = "inputs_waypoints"
JOINT_ACTIONS_OUTPUT = "policy.joint_actions"
TRAJECTORY_OUTPUT = "policy.trajectory"
TRAJECTORY_MODE_LOGITS_OUTPUT = "policy.trajectory_mode_logits"

# Training frames are 576x324; the model crops and resizes from that size.
PRE_CROP_HW = (324, 576)
CENTER_CROP_HW = (320, 576)

# Waypoints are sampled along the route every WAYPOINT_DIST_M, starting
# WAYPOINT_FIRST_OFFSET_M ahead of the ego's projection onto it, and fed in
# hundreds of metres clipped to [-1, 1].
WAYPOINT_DIST_M = 10.0
WAYPOINT_FIRST_OFFSET_M = 1.0
WAYPOINT_SCALE_M = 100.0

FLEET_CALIBRATION_PATH = Path(__file__).with_name("cam-90-deg.json")


def frame_rope_cos_sin(
    frame_index: int, head_dim: int, *, base: float = ROPE_BASE
) -> tuple[np.ndarray, np.ndarray]:
    """Frame-granular RoPE ``(1, head_dim)`` cos/sin.

    Computed in float64 and cast to float32: at ``base=1000`` the frequencies
    span enough orders of magnitude that float32 accumulation differs.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angle = float(frame_index) * inv_freq
    cos = np.concatenate((np.cos(angle), np.cos(angle))).reshape(1, -1)
    sin = np.concatenate((np.sin(angle), np.sin(angle))).reshape(1, -1)
    return cos.astype(np.float32), sin.astype(np.float32)


def _static_shape(dims: Any) -> tuple[int, ...]:
    shape = tuple(dims or ())
    if len(shape) == 0 or any(not isinstance(d, int) or d <= 0 for d in shape):
        raise ValueError(f"expected a static positive-integer shape, got {dims!r}")
    return shape


def _camera_alias(graph_name: str) -> str:
    """``inputs_image`` -> ``cam_front_left``, ``inputs_image_<cam>`` -> ``<cam>``."""
    if graph_name == "inputs_image":
        return "cam_front_left"
    prefix = "inputs_image_"
    if graph_name.startswith(prefix) and len(graph_name) > len(prefix):
        return graph_name[len(prefix) :]
    raise ValueError(
        f"camera input {graph_name!r} doesn't match 'inputs_image' or "
        "'inputs_image_<camera>'"
    )


@dataclass(frozen=True)
class DecoderGraphSpec:
    """Geometry of a decoder-only KV-cache export, read off the graph."""

    camera_inputs: dict[str, str]  # camera id -> graph input name
    image_hw: tuple[int, int]
    num_waypoints: int
    past_kv_shape: tuple[int, ...]  # (layers, batch, heads, cache_tokens, head_dim)
    tokens_per_frame: int
    num_hypotheses: int
    num_poses: int

    @property
    def head_dim(self) -> int:
        return self.past_kv_shape[4]

    @property
    def cache_tokens(self) -> int:
        return self.past_kv_shape[3]

    @property
    def cache_frames(self) -> int:
        return self.cache_tokens // self.tokens_per_frame

    @classmethod
    def from_session(cls, session: Any) -> DecoderGraphSpec:
        """Validate an ``onnxruntime.InferenceSession`` and read its geometry.

        Raises:
            ValueError: If the graph is not a single-batch float32
                decoder-only export with a trajectory head.
        """
        inputs = {i.name: i for i in session.get_inputs()}
        outputs = {o.name: o for o in session.get_outputs()}

        required_inputs = CACHE_INPUT_NAMES | {SPEED_INPUT, WAYPOINTS_INPUT}
        required_outputs = {
            "new_k",
            "new_v",
            JOINT_ACTIONS_OUTPUT,
            TRAJECTORY_OUTPUT,
            TRAJECTORY_MODE_LOGITS_OUTPUT,
        }
        missing = sorted(required_inputs - inputs.keys()) + sorted(
            required_outputs - outputs.keys()
        )
        if missing:
            raise ValueError(
                "not a decoder-only KV-cache export with a trajectory head; "
                f"missing {missing}"
            )
        non_float = sorted(
            n for n, t in {**inputs, **outputs}.items() if t.type != "tensor(float)"
        )
        if non_float:
            raise ValueError(
                f"expected float32 tensors only, got non-float {non_float}"
            )

        camera_graph_names = [
            n
            for n in inputs
            if n not in required_inputs and len(_static_shape(inputs[n].shape)) == 5
        ]
        if not camera_graph_names:
            raise ValueError(f"no camera input found; inputs: {sorted(inputs)}")
        image_hws = {_static_shape(inputs[n].shape)[-2:] for n in camera_graph_names}
        if len(image_hws) != 1:
            raise ValueError(f"camera inputs disagree on image size: {image_hws}")
        (image_hw,) = image_hws

        waypoints_shape = _static_shape(inputs[WAYPOINTS_INPUT].shape)
        if len(waypoints_shape) != 4 or waypoints_shape[-1] != 2:
            raise ValueError(
                f"{WAYPOINTS_INPUT}: expected (1, 1, N, 2), got {waypoints_shape}"
            )

        past_k = _static_shape(inputs["inputs_past_k"].shape)
        past_v = _static_shape(inputs["inputs_past_v"].shape)
        new_k = _static_shape(outputs["new_k"].shape)
        new_v = _static_shape(outputs["new_v"].shape)
        if len(past_k) != 5 or past_v != past_k or new_v != new_k or len(new_k) != 5:
            raise ValueError(
                f"inconsistent K/V shapes: past_k={past_k}, past_v={past_v}, "
                f"new_k={new_k}, new_v={new_v}"
            )
        layers, batch, heads, cache_tokens, head_dim = past_k
        if batch != 1:
            raise ValueError(f"expected batch 1, got K/V {past_k}")
        tokens_per_frame = new_k[3]
        if (new_k[0], new_k[1], new_k[2], new_k[4]) != (layers, batch, heads, head_dim):
            raise ValueError(f"new K/V {new_k} incompatible with past K/V {past_k}")
        if cache_tokens % tokens_per_frame:
            raise ValueError(
                f"cache_tokens={cache_tokens} is not a multiple of "
                f"tokens_per_frame={tokens_per_frame}"
            )
        bias_shape = _static_shape(inputs["inputs_cache_bias"].shape)
        if bias_shape != (1, 1, 1, cache_tokens):
            raise ValueError(
                f"inputs_cache_bias: expected (1, 1, 1, {cache_tokens}), got {bias_shape}"
            )
        for name in ("inputs_rope_cos", "inputs_rope_sin"):
            shape = _static_shape(inputs[name].shape)
            if shape != (1, head_dim):
                raise ValueError(f"{name}: expected (1, {head_dim}), got {shape}")

        joint_actions = _static_shape(outputs[JOINT_ACTIONS_OUTPUT].shape)
        if len(joint_actions) != 3 or joint_actions[0] != 1 or joint_actions[2] != 4:
            raise ValueError(
                f"{JOINT_ACTIONS_OUTPUT}: expected (1, H, 4), got {joint_actions}"
            )
        trajectory = _static_shape(outputs[TRAJECTORY_OUTPUT].shape)
        if len(trajectory) != 4 or trajectory[0] != 1 or trajectory[3] != 3:
            raise ValueError(
                f"{TRAJECTORY_OUTPUT}: expected (1, Q, P, 3), got {trajectory}"
            )
        logits = _static_shape(outputs[TRAJECTORY_MODE_LOGITS_OUTPUT].shape)
        if logits != (1, trajectory[1]):
            raise ValueError(
                f"{TRAJECTORY_MODE_LOGITS_OUTPUT}: expected (1, {trajectory[1]}), got {logits}"
            )

        return cls(
            camera_inputs={_camera_alias(n): n for n in camera_graph_names},
            image_hw=image_hw,
            num_waypoints=waypoints_shape[2],
            past_kv_shape=past_k,
            tokens_per_frame=tokens_per_frame,
            num_hypotheses=trajectory[1],
            num_poses=trajectory[2],
        )


class KvCacheRing:
    """Host-owned KV cache for one driving session.

    Each tick writes the new frame's K/V into one ring slot and never shifts
    the others.  Cache and frame counter are created together and never reset
    separately, so RoPE offsets always match what is in the ring.
    """

    def __init__(self, spec: DecoderGraphSpec):
        self._spec = spec
        self.past_k = np.zeros(spec.past_kv_shape, dtype=np.float32)
        self.past_v = np.zeros(spec.past_kv_shape, dtype=np.float32)
        self.cache_bias = np.full(
            (1, 1, 1, spec.cache_tokens), MASK_BIAS, dtype=np.float32
        )
        self.frame_index = 0

    def feed(self) -> dict[str, np.ndarray]:
        """Cache and RoPE inputs for the current tick."""
        rope_cos, rope_sin = frame_rope_cos_sin(self.frame_index, self._spec.head_dim)
        return {
            "inputs_past_k": self.past_k,
            "inputs_past_v": self.past_v,
            "inputs_cache_bias": self.cache_bias,
            "inputs_rope_cos": rope_cos,
            "inputs_rope_sin": rope_sin,
        }

    def advance(self, new_k: np.ndarray, new_v: np.ndarray) -> None:
        """Store this tick's K/V and move to the next frame."""
        tokens = self._spec.tokens_per_frame
        lo = (self.frame_index % self._spec.cache_frames) * tokens
        hi = lo + tokens
        self.past_k[..., lo:hi, :] = new_k
        self.past_v[..., lo:hi, :] = new_v
        self.cache_bias[..., lo:hi] = 0.0
        self.frame_index += 1
        self.assert_invariant()

    def assert_invariant(self) -> None:
        """Filled slots must equal ``min(frame_index, cache_frames) * tokens_per_frame``."""
        valid = int(np.sum(self.cache_bias == 0.0))
        expected = (
            min(self.frame_index, self._spec.cache_frames) * self._spec.tokens_per_frame
        )
        if valid != expected:
            raise AssertionError(
                f"KV cache invariant violated: valid_slots={valid}, expected={expected}"
            )


@dataclass(frozen=True)
class FisheyeCalibration:
    """Kannala-Brandt fisheye intrinsics at their native resolution."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, float, float, float]
    native_width: int
    native_height: int

    @classmethod
    def from_json(cls, path: Path) -> FisheyeCalibration:
        data = json.loads(path.read_text())
        if data["model"] != "kannala_brandt":
            raise ValueError(
                f"unsupported calibration model {data['model']!r} in {path}"
            )
        k1, k2, k3, k4 = data["dist"]
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            dist=(float(k1), float(k2), float(k3), float(k4)),
            native_width=int(data["native_width"]),
            native_height=int(data["native_height"]),
        )


class Defisher:
    """Fisheye -> rectilinear remap reproducing Yaak's ``lsd defish_video``.

    The output keeps the input's size and principal point and uses an ideal
    pinhole with focal length ``f_out = cx`` (~90 deg HFOV).  Frames must be at
    the calibration's native resolution.
    """

    def __init__(self, calib: FisheyeCalibration):
        self.width = calib.native_width
        self.height = calib.native_height
        cx, cy = calib.cx, calib.cy
        k1, k2, k3, k4 = calib.dist
        f_out = cx

        # For every output pixel, find the fisheye source pixel.
        v_g, u_g = np.meshgrid(
            np.arange(self.height, dtype=np.float64),
            np.arange(self.width, dtype=np.float64),
            indexing="ij",
        )
        x = (u_g - cx) / f_out
        y = (v_g - cy) / f_out
        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)
        t_sq = theta * theta
        theta_d = theta * (1.0 + k1 * t_sq + k2 * t_sq**2 + k3 * t_sq**3 + k4 * t_sq**4)
        safe_r = np.where(r > 1e-9, r, 1.0)
        factor = theta_d / safe_r
        self._map_x = np.where(r < 1e-9, cx, cx + calib.fx * factor * x).astype(
            np.float32
        )
        self._map_y = np.where(r < 1e-9, cy, cy + calib.fy * factor * y).astype(
            np.float32
        )

    def apply(self, image_hwc: np.ndarray) -> np.ndarray:
        if image_hwc.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"defish expects {self.height}x{self.width} frames, got "
                f"{image_hwc.shape[0]}x{image_hwc.shape[1]}; render the camera at "
                "its native fisheye resolution"
            )
        return cv2.remap(
            image_hwc,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def preprocess_image(
    image_hwc_uint8: np.ndarray, image_hw: tuple[int, int]
) -> np.ndarray:
    """Rectilinear HWC uint8 RGB -> CHW float32 in [0, 1] at ``image_hw``.

    Resize to 324x576, center-crop 320x576, resize to the graph's size, all in
    the uint8 domain as in training.  ImageNet normalization lives inside the
    graph, so it is not applied here.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(image_hwc_uint8))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = T.Resize(list(PRE_CROP_HW))(tensor)
    tensor = T.CenterCrop(list(CENTER_CROP_HW))(tensor)
    tensor = T.Resize(list(image_hw))(tensor)
    return (tensor.float() / 255.0)[0].numpy().astype(np.float32)


def rig_xy_to_model(xy_rig: np.ndarray) -> np.ndarray:
    """(x_forward, y_left) -> (x_right, y_forward)."""
    return np.stack([-xy_rig[..., 1], xy_rig[..., 0]], axis=-1)


def model_xy_to_rig(xy_model: np.ndarray) -> np.ndarray:
    """(x_right, y_forward) -> (x_forward, y_left)."""
    return np.stack([xy_model[..., 1], -xy_model[..., 0]], axis=-1)


def _cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _project_onto_polyline(
    points: np.ndarray, cumlen: np.ndarray, target: np.ndarray
) -> float:
    """Arc length of the closest point on the polyline (segments, not vertices)."""
    segs = np.diff(points, axis=0)
    seg_len2 = np.sum(segs * segs, axis=1)
    rel = target[None, :] - points[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(seg_len2 > 1e-12, np.sum(rel * segs, axis=1) / seg_len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    feet = points[:-1] + t[:, None] * segs
    d2 = np.sum((target[None, :] - feet) ** 2, axis=1)
    j = int(np.argmin(d2))
    return float(cumlen[j] + t[j] * np.sqrt(seg_len2[j]))


def _interpolate_at(
    points: np.ndarray, cumlen: np.ndarray, target_s: float
) -> np.ndarray:
    """Point at arc length ``target_s``, clamped to the polyline's ends."""
    if target_s <= 0.0:
        return points[0].copy()
    if target_s >= cumlen[-1]:
        return points[-1].copy()
    idx = int(np.searchsorted(cumlen, target_s, side="left"))
    if idx == 0:
        return points[0].copy()
    s0, s1 = cumlen[idx - 1], cumlen[idx]
    if s1 == s0:
        return points[idx].copy()
    t = (target_s - s0) / (s1 - s0)
    return (1.0 - t) * points[idx - 1] + t * points[idx]


def route_to_model_waypoints(
    route_xy_rig: np.ndarray, num_waypoints: int
) -> np.ndarray:
    """Route polyline in the rig frame at t0 -> ``(num_waypoints, 2)`` model input.

    NaN rows (padding of a short route) are dropped.  A route with fewer than
    two points yields zeros, as rsim feeds when there is no route.
    """
    points = route_xy_rig[~np.isnan(route_xy_rig).any(axis=1)].astype(np.float64)
    if len(points) < 2:
        return np.zeros((num_waypoints, 2), dtype=np.float32)
    points = rig_xy_to_model(points)
    cumlen = _cumulative_arc_length(points)
    s0 = _project_onto_polyline(points, cumlen, np.zeros(2)) + WAYPOINT_FIRST_OFFSET_M
    samples = np.stack(
        [
            _interpolate_at(points, cumlen, s0 + WAYPOINT_DIST_M * i)
            for i in range(num_waypoints)
        ]
    )
    return np.clip(samples / WAYPOINT_SCALE_M, -1.0, 1.0).astype(np.float32)
