# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from navsim_diffusiondrive_challenge.policy import (
    DiffusionDrivePolicy,
    InferenceInput,
    Prediction,
)
from navsim_diffusiondrive_challenge.preprocessing import CAMERA_IDS

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "assets/diffusiondrive/diffusiondrive_sim_navhard.ckpt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the SimScale DiffusionDrive NAVHARD inference path"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
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


def _validated_output(
    predictions: list[Prediction],
    batch_size: int,
) -> tuple[int, int, int]:
    if len(predictions) != batch_size:
        raise RuntimeError(
            f"policy returned {len(predictions)} predictions for batch {batch_size}"
        )
    for prediction in predictions:
        if prediction.trajectory.shape != (8, 3):
            raise RuntimeError(
                "prediction trajectory must have shape (8, 3); "
                f"got {prediction.trajectory.shape}"
            )
        if not np.isfinite(prediction.trajectory).all():
            raise RuntimeError("prediction trajectory contains non-finite values")
    return (batch_size, 8, 3)


def _forward(
    policy: DiffusionDrivePolicy,
    requests: list[InferenceInput],
    device: torch.device,
) -> tuple[float, tuple[int, int, int]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    predictions = policy.predict_batch(requests)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, _validated_output(predictions, len(requests))


def _benchmark_result(
    *,
    batch_size: int,
    first_forward_ms: float,
    samples: list[float],
    peak_vram_bytes: int,
    device_name: str,
    output_shape: tuple[int, int, int],
) -> dict[str, int | float | str | list[int] | None]:
    return {
        "batch_size": batch_size,
        "cuda_version": torch.version.cuda,
        "device_name": device_name,
        "first_forward_ms": first_forward_ms,
        "iterations": len(samples),
        "numpy_version": np.__version__,
        "output_shape": list(output_shape),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "peak_vram_bytes": peak_vram_bytes,
        "torch_version": torch.__version__,
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; choose an available device")

    requests = [_request() for _ in range(args.batch_size)]
    policy = DiffusionDrivePolicy(
        args.checkpoint,
        device=device,
        warm_up=False,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first_forward_ms, output_shape = _forward(policy, requests, device)
    for _ in range(args.warmup):
        _, output_shape = _forward(policy, requests, device)
    samples = [_forward(policy, requests, device)[0] for _ in range(args.iterations)]

    if device.type == "cuda":
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(device))
        device_name = torch.cuda.get_device_name(device)
    else:
        peak_vram_bytes = 0
        device_name = str(device)
    print(
        json.dumps(
            _benchmark_result(
                batch_size=args.batch_size,
                first_forward_ms=first_forward_ms,
                samples=samples,
                peak_vram_bytes=peak_vram_bytes,
                device_name=device_name,
                output_shape=output_shape,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
