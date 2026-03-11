#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Clean up old squash files in the dev/ subdirectory.

Removes .sqsh files older than 7 days via SSH to the remote SLURM filesystem.
This is the GitHub Actions equivalent of GitLab's cleanup-dev-squash job.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime


def log(level: str, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def _require_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if not value:
        log("WARN", f"{name} is not set")
        return None
    return value


def _first_env(names: list[str]) -> tuple[str | None, str | None]:
    """Return the first non-empty env value and the env name used."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value, name
    return None, None


def make_ssh_cmd(command: str) -> str:
    # Prefer generic GitHub names; keep legacy GitLab names as fallback.
    ssh_key, key_name = _first_env(
        ["DEV_SQUASH_SSH_KEY_B64", "SLURM_FRONTEND_USER_KEY"]
    )
    ssh_user, user_name = _first_env(["DEV_SQUASH_SSH_USER", "SLURM_FRONTEND_USER"])
    ssh_host, host_name = _first_env(["DEV_SQUASH_SSH_HOST", "SLURM_ORD_HOST"])

    if not ssh_key:
        _require_env("DEV_SQUASH_SSH_KEY_B64")
        _require_env("SLURM_FRONTEND_USER_KEY")
    if not ssh_user:
        _require_env("DEV_SQUASH_SSH_USER")
        _require_env("SLURM_FRONTEND_USER")
    if not ssh_host:
        _require_env("DEV_SQUASH_SSH_HOST")
        _require_env("SLURM_ORD_HOST")

    if not (ssh_key and ssh_user and ssh_host):
        raise RuntimeError("Missing one or more required SSH environment variables")

    log(
        "INFO",
        f"Using SSH user from {user_name}, host from {host_name}, key from {key_name}",
    )

    return (
        "ssh-agent bash -c "
        '"ssh-add <(echo ' + ssh_key + " | base64 -d) 2>/dev/null && "
        "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        "-o LogLevel=QUIET -n " + ssh_user + "@" + ssh_host + " '" + command + "'\""
    )


def main() -> int:
    # Prefer generic GitHub name; keep legacy name as fallback.
    squash_cache_dir = (
        os.getenv("DEV_SQUASH_CACHE_DIR", "").strip()
        or os.getenv("SQUASH_CACHE_DIR", "").strip()
    )
    if not squash_cache_dir:
        log("WARN", "DEV_SQUASH_CACHE_DIR/SQUASH_CACHE_DIR is not set")
        log("INFO", "Skipping cleanup because cache directory is not configured")
        return 0

    # For testability and safety in GitHub, allow dry-run mode.
    dry_run = os.getenv("CLEANUP_DEV_SQUASH_DRY_RUN", "false").lower() == "true"

    log("INFO", f"DEV_SQUASH_CACHE_DIR={squash_cache_dir}")
    log("INFO", f"Dry run mode: {dry_run}")

    dev_dir = os.path.join(squash_cache_dir, "dev")
    log("INFO", f"Target directory: {dev_dir}")

    if not dev_dir.endswith("dev"):
        log("ERROR", f"Safety check failed: path {dev_dir} does not end with dev")
        return 1

    # If SSH env isn't configured, skip rather than hard-fail.
    missing_ssh = []
    if not (
        os.getenv("DEV_SQUASH_SSH_KEY_B64", "").strip()
        or os.getenv("SLURM_FRONTEND_USER_KEY", "").strip()
    ):
        missing_ssh.append("DEV_SQUASH_SSH_KEY_B64")
    if not (
        os.getenv("DEV_SQUASH_SSH_USER", "").strip()
        or os.getenv("SLURM_FRONTEND_USER", "").strip()
    ):
        missing_ssh.append("DEV_SQUASH_SSH_USER")
    if not (
        os.getenv("DEV_SQUASH_SSH_HOST", "").strip()
        or os.getenv("SLURM_ORD_HOST", "").strip()
    ):
        missing_ssh.append("DEV_SQUASH_SSH_HOST")
    if missing_ssh:
        log("WARN", f"Missing SSH config env vars: {', '.join(missing_ssh)}")
        log("INFO", "Skipping cleanup because remote SSH connection is not configured")
        return 0

    try:
        check_cmd = make_ssh_cmd(
            f"[ -d {dev_dir} ] && echo 'exists' || echo 'not_found'"
        )
        result = subprocess.run(
            check_cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        if "not_found" in result.stdout:
            log(
                "WARN",
                f"Dev directory {dev_dir} not found on remote. Nothing to clean.",
            )
            return 0

        list_cmd = make_ssh_cmd(
            f"find {dev_dir} -type f -name '\\''*.sqsh'\\'' -mtime +7 2>/dev/null"
        )
        result = subprocess.run(
            list_cmd, shell=True, capture_output=True, text=True, timeout=60
        )
        paths_to_delete = [
            p.strip() for p in (result.stdout or "").strip().splitlines() if p.strip()
        ]

        count_all_cmd = make_ssh_cmd(
            f"find {dev_dir} -type f -name '\\''*.sqsh'\\'' 2>/dev/null | wc -l"
        )
        result_all = subprocess.run(
            count_all_cmd, shell=True, capture_output=True, text=True, timeout=60
        )
        file_count_all = (
            result_all.stdout.strip() if result_all.returncode == 0 else "?"
        )

        if not paths_to_delete:
            log("INFO", "No old squash files to clean up")
            log("INFO", f"Total files: {file_count_all}")
            return 0

        file_count = len(paths_to_delete)
        log(
            "INFO",
            f"Found {file_count} of {file_count_all} squash files older than 7 days",
        )
        log("INFO", "Sample files to be deleted:")
        for path in paths_to_delete[:10]:
            log("INFO", f"  - {os.path.basename(path)}")

        if dry_run:
            log("INFO", "Dry run enabled, skipping deletion")
            return 0

        quoted_paths = " ".join(
            "'\\''" + p.replace("'", "'\\''") + "'\\''" for p in paths_to_delete
        )
        delete_cmd = make_ssh_cmd(f"rm -- {quoted_paths} || true")
        result = subprocess.run(
            delete_cmd, shell=True, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            log("ERROR", f"Failed to delete old files (exit code {result.returncode})")
            if result.stderr.strip():
                log("ERROR", f"stderr: {result.stderr.strip()}")
            return 1

        log("INFO", f"Cleanup complete - removed up to {file_count} old dev files")

        verify_cmd = make_ssh_cmd(
            f"find {dev_dir} -type f -name '\\''*.sqsh'\\'' -mtime +7 2>/dev/null | wc -l"
        )
        result = subprocess.run(
            verify_cmd, shell=True, capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            remaining = int(result.stdout.strip())
            if remaining > 0:
                log("WARN", f"Verification shows {remaining} old files still remain")
            else:
                log("INFO", "Verification successful - all old files removed")

    except subprocess.TimeoutExpired:
        log("ERROR", "SSH command timed out")
        return 1
    except Exception as exc:  # pragma: no cover
        log("ERROR", f"Unexpected error during cleanup: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
