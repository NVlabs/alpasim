# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from navsim_transfuser_challenge.policy import InferenceInput, LtfPolicy
from navsim_transfuser_challenge.preprocessing import CAMERA_IDS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark released LTF inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be at least 0")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    return args


def _request() -> InferenceInput:
    return InferenceInput(
        images={
            camera_id: np.zeros((1080, 1920, 3), dtype=np.uint8)
            for camera_id in CAMERA_IDS
        },
        command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
        velocity_xy=np.zeros(2, dtype=np.float32),
        acceleration_xy=np.zeros(2, dtype=np.float32),
    )


def _timed_forward(policy: LtfPolicy, requests: list[InferenceInput]) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    policy.predict_batch(requests)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; benchmark requires an NVIDIA GPU")

    requests = [_request() for _ in range(args.batch_size)]
    policy = LtfPolicy(args.checkpoint, device="cuda", warm_up=False)

    torch.cuda.reset_peak_memory_stats()
    first_forward_ms = _timed_forward(policy, requests)

    for _ in range(args.warmup):
        policy.predict_batch(requests)
    torch.cuda.synchronize()

    samples = [_timed_forward(policy, requests) for _ in range(args.iterations)]
    result = {
        "batch_size": args.batch_size,
        "first_forward_ms": first_forward_ms,
        "iterations": args.iterations,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
