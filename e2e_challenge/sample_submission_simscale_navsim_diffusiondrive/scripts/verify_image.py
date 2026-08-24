#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Verify the standalone DiffusionDrive image after it is built."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

CHECKPOINT = Path("/app/assets/diffusiondrive/diffusiondrive_sim_navhard.ckpt")
FORBIDDEN_PATHS = (
    Path("/app/assets/ltf"),
    Path("/app/navsim_transfuser_challenge"),
    Path("/app/assets/gtrs_dense"),
    Path("/app/navsim_gtrs_dense_challenge"),
)


def verify_filesystem(root: Path = Path("/")) -> None:
    for image_path in FORBIDDEN_PATHS:
        host_path = root / image_path.relative_to("/")
        if host_path.exists():
            raise RuntimeError(f"unexpected inherited path: {image_path}")


def verify_environment(environment: Mapping[str, str]) -> None:
    inherited = sorted(
        name
        for name in environment
        if name.startswith("LTF_") or name.startswith("GTRS_")
    )
    if inherited:
        raise RuntimeError(f"unexpected inherited environment: {inherited}")


def verify_modules() -> None:
    inherited = {
        "navsim_transfuser_challenge": importlib.util.find_spec(
            "navsim_transfuser_challenge"
        ),
        "navsim_gtrs_dense_challenge": importlib.util.find_spec(
            "navsim_gtrs_dense_challenge"
        ),
    }
    present = sorted(name for name, spec in inherited.items() if spec is not None)
    if present:
        raise RuntimeError(f"unexpected inherited module: {present}")


def verify_identity() -> None:
    uid = os.getuid()
    gid = os.getgid()
    if (uid, gid) != (10001, 10001):
        raise RuntimeError(
            f"runtime identity {uid}:{gid}, expected uid:gid 10001:10001"
        )


def verify_checkpoint(checkpoint: Path, expected_size: int, expected_sha: str) -> None:
    digest = hashlib.sha256()
    size = 0
    with checkpoint.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)

    actual_sha = digest.hexdigest()
    if size != expected_size:
        raise RuntimeError(f"checkpoint size {size}, expected {expected_size}")
    if actual_sha != expected_sha:
        raise RuntimeError(f"checkpoint sha256 {actual_sha}, expected {expected_sha}")


def verify_runtime(
    checkpoint: Path,
    expected_size: int,
    expected_sha: str,
    environment: Mapping[str, str],
) -> None:
    verify_identity()
    verify_checkpoint(checkpoint, expected_size, expected_sha)
    verify_environment(environment)
    verify_modules()
    importlib.import_module("navsim_diffusiondrive_challenge.driver")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["filesystem"]:
        verify_filesystem()
        verify_modules()
        return
    if arguments == ["runtime"]:
        verify_runtime(
            CHECKPOINT,
            int(os.environ["DIFFUSIONDRIVE_PROBE_EXPECTED_SIZE"]),
            os.environ["DIFFUSIONDRIVE_PROBE_EXPECTED_SHA256"],
            os.environ,
        )
        return
    raise SystemExit("usage: verify_image.py {filesystem|runtime}")


if __name__ == "__main__":
    main()
