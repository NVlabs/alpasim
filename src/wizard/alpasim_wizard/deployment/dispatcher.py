# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Command dispatcher with logging capabilities for the wizard."""

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import TextIO

logger = logging.getLogger("alpasim_wizard_dispatcher")


class OsDispatchError(RuntimeError):
    """Error raised when command execution fails."""

    pass


def dispatch_command(
    cmd: str,
    log_dir: Path | str,
    dry_run: bool = False,
) -> str:
    """Execute a blocking command with logging."""
    log_dir = Path(log_dir)
    output_log_file = _prepare_logs(cmd, log_dir)
    if dry_run:
        logger.info(f"[DRY-RUN] Would execute: {cmd}")
        return ""

    logger.info(f"Executing: {cmd}")
    with open(output_log_file, "a") as log_file:
        _write_command_header(log_file, cmd)
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines = []
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    log_file.write(line)
                    logger.debug(line.rstrip())
                    output_lines.append(line)
                process.stdout.close()
            return_code = process.wait()
            if return_code != 0:
                raise OsDispatchError(
                    f"Command failed with return code {return_code}: {cmd}"
                )
            return "".join(output_lines)
        except subprocess.SubprocessError as e:
            raise OsDispatchError(f"Failed to execute command '{cmd}': {e}")


def dispatch_background(
    cmd: str,
    log_dir: Path | str,
    dry_run: bool = False,
) -> subprocess.Popen[str] | None:
    """Start a background command in its own session with logging."""
    log_dir = Path(log_dir)
    output_log_file = _prepare_logs(cmd, log_dir)
    if dry_run:
        logger.info(f"[DRY-RUN] Would execute: {cmd}")
        return None

    logger.info(f"Executing: {cmd}")
    with open(output_log_file, "a") as log_file:
        _write_command_header(log_file, cmd)
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except subprocess.SubprocessError as e:
            raise OsDispatchError(f"Failed to execute command '{cmd}': {e}")
    logger.info(
        "Started background process (output → %s): %s",
        output_log_file,
        cmd,
    )
    return process


def _prepare_logs(cmd: str, log_dir: Path) -> Path:
    txt_logs_dir = log_dir / "txt-logs"
    txt_logs_dir.mkdir(parents=True, exist_ok=True)
    with open(txt_logs_dir / "os_dispatch_log.txt", "a") as command_log:
        command_log.write(f"{cmd}\n")
    return txt_logs_dir / "os_dispatch_output.txt"


def _write_command_header(log_file: TextIO, cmd: str) -> None:
    log_file.write(f"\n{'=' * 60}\n")
    log_file.write(f"Command: {cmd}\n")
    log_file.write(f"{'=' * 60}\n")


def terminate_process(
    process: subprocess.Popen[str],
    timeout: float = 10,
) -> None:
    """Terminate a non-blocking command and all processes in its session."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()
