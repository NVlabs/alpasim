# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path
from unittest.mock import MagicMock, call

from alpasim_wizard.deployment import dispatcher
from alpasim_wizard.deployment.dispatcher import dispatch_background, terminate_process


def test_nonblocking_command_returns_running_process(tmp_path: Path) -> None:
    process = dispatch_background(
        "true",
        log_dir=tmp_path,
    )

    assert process is not None
    assert process.wait(timeout=1) == 0


def test_nonblocking_command_can_be_terminated_with_its_session(
    tmp_path: Path,
) -> None:
    process = dispatch_background(
        "sleep 60",
        log_dir=tmp_path,
    )

    assert process is not None
    terminate_process(process, timeout=1)
    assert process.poll() is not None


def test_terminate_process_escalates_group_after_leader_exits(monkeypatch) -> None:
    process = MagicMock()
    process.pid = 123
    process.poll.return_value = 0
    killpg = MagicMock()
    monkeypatch.setattr(dispatcher.os, "killpg", killpg)

    terminate_process(process, timeout=0)

    assert killpg.call_args_list == [
        call(123, dispatcher.signal.SIGTERM),
        call(123, dispatcher.signal.SIGKILL),
    ]
    process.wait.assert_called_once_with()
