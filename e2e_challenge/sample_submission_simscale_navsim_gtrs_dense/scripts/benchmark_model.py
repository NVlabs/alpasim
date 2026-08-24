# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from navsim_gtrs_dense_challenge.policy import GTRSDensePolicy, InferenceInput
from navsim_gtrs_dense_challenge.preprocessing import CAMERA_IDS

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "assets/gtrs_dense/gtrs_dense_resnet_sim_reward_navhard.ckpt"
)
DEFAULT_VOCABULARY = (
    Path(__file__).resolve().parents[1] / "assets/gtrs_dense/navsim_16384.npy"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GTRS-Dense ResNet reward NAVHARD production benchmark in FP32",
        epilog=(
            "example: benchmark_model.py --checkpoint "
            "e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/assets/gtrs_dense/"
            "gtrs_dense_resnet_sim_reward_navhard.ckpt "
            "--batch-size 1 --warmup 1 --iterations 5\n"
            "The backbone emits 4096 image key/value tokens.\n"
            "The scorer evaluates the verified 16,384-candidate trajectory vocabulary in FP32."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="ResNet reward NAVHARD checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=DEFAULT_VOCABULARY,
        help="Official NAVHARD trajectory vocabulary (default: %(default)s)",
    )
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


def _timed_forward(policy: GTRSDensePolicy, requests: list[InferenceInput]) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    policy.predict_batch(requests)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _benchmark_result(
    *,
    batch_size: int,
    first_forward_ms: float,
    samples: list[float],
    peak_vram_bytes: int,
) -> dict[str, int | float]:
    return {
        "batch_size": batch_size,
        "first_forward_ms": first_forward_ms,
        "iterations": len(samples),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "peak_vram_bytes": peak_vram_bytes,
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; benchmark requires an NVIDIA GPU")

    requests = [_request() for _ in range(args.batch_size)]
    policy = GTRSDensePolicy(
        args.checkpoint,
        args.vocabulary,
        device="cuda",
        warm_up=False,
    )

    torch.cuda.reset_peak_memory_stats()
    first_forward_ms = _timed_forward(policy, requests)

    for _ in range(args.warmup):
        policy.predict_batch(requests)
    torch.cuda.synchronize()

    samples = [_timed_forward(policy, requests) for _ in range(args.iterations)]
    result = _benchmark_result(
        batch_size=args.batch_size,
        first_forward_ms=first_forward_ms,
        samples=samples,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
