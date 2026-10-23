# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Writes tiny stand-ins for a decoder-only KV-cache trajectory export.

``dummy_decoder.onnx`` has the real export's input/output names with small
shapes: a 2-frame ring of 3 tokens per frame, 5 hypotheses of 5 poses.
``new_k`` echoes the speed input and ``new_v`` the mean image value, so tests
can see what reached the graph.  The trajectory and mode logits are
constants (``trajectory()`` and ``MODE_LOGITS`` below).
``dummy_no_trajectory.onnx`` drops the trajectory head.

Run from this directory:
    uv run --no-project --with onnx --with numpy python make_dummy_graphs.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

KV_SHAPE = [2, 1, 2, 6, 4]
NEW_KV_SHAPE = [2, 1, 2, 3, 4]
NUM_HYPOTHESES = 5
NUM_POSES = 5
MODE_LOGITS = np.array([[0.0, 1.0, 0.0, 3.0, 0.0]], dtype=np.float32)


def trajectory() -> np.ndarray:
    """Hypothesis q, pose p at (x_right = q - 2, y_forward = 2 (p + 1)), theta 0."""
    out = np.zeros((1, NUM_HYPOTHESES, NUM_POSES, 3), dtype=np.float32)
    for q in range(NUM_HYPOTHESES):
        for p in range(NUM_POSES):
            out[0, q, p] = (q - 2, 2 * (p + 1), 0.0)
    return out


def _input(name: str, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def build(with_trajectory: bool) -> onnx.ModelProto:
    inputs = [
        _input("inputs_image_cam_front_left", [1, 1, 3, 8, 8]),
        _input("inputs_speed", [1, 1, 1]),
        _input("inputs_waypoints", [1, 1, 10, 2]),
        _input("inputs_past_k", KV_SHAPE),
        _input("inputs_past_v", KV_SHAPE),
        _input("inputs_cache_bias", [1, 1, 1, KV_SHAPE[3]]),
        _input("inputs_rope_cos", [1, KV_SHAPE[4]]),
        _input("inputs_rope_sin", [1, KV_SHAPE[4]]),
    ]
    outputs = [
        _input("new_k", NEW_KV_SHAPE),
        _input("new_v", NEW_KV_SHAPE),
        _input("policy.joint_actions", [1, 6, 4]),
    ]
    initializers = [
        numpy_helper.from_array(np.array(NEW_KV_SHAPE, dtype=np.int64), "new_kv_shape"),
        numpy_helper.from_array(np.zeros((1, 6, 4), dtype=np.float32), "joint_actions"),
    ]
    nodes = [
        helper.make_node("Expand", ["inputs_speed", "new_kv_shape"], ["new_k"]),
        helper.make_node(
            "ReduceMean", ["inputs_image_cam_front_left"], ["image_mean"], keepdims=1
        ),
        helper.make_node("Expand", ["image_mean", "new_kv_shape"], ["new_v"]),
        helper.make_node("Identity", ["joint_actions"], ["policy.joint_actions"]),
    ]
    if with_trajectory:
        outputs += [
            _input("policy.trajectory", [1, NUM_HYPOTHESES, NUM_POSES, 3]),
            _input("policy.trajectory_mode_logits", [1, NUM_HYPOTHESES]),
        ]
        initializers += [
            numpy_helper.from_array(trajectory(), "trajectory"),
            numpy_helper.from_array(MODE_LOGITS, "mode_logits"),
        ]
        nodes += [
            helper.make_node("Identity", ["trajectory"], ["policy.trajectory"]),
            helper.make_node(
                "Identity", ["mode_logits"], ["policy.trajectory_mode_logits"]
            ),
        ]
    graph = helper.make_graph(nodes, "dummy_decoder", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    here = Path(__file__).parent
    onnx.save(build(with_trajectory=True), here / "dummy_decoder.onnx")
    onnx.save(build(with_trajectory=False), here / "dummy_no_trajectory.onnx")


if __name__ == "__main__":
    main()
