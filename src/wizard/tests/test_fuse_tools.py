# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import io
import stat
import subprocess
from pathlib import Path

import pytest
from alpasim_wizard.deployment import fuse_tools


def _write_complete_cache(fuse_dir: Path) -> Path:
    bin_dir, lib_dir = fuse_tools.fuse_tool_paths(fuse_dir)
    bin_dir.mkdir()
    lib_dir.mkdir()
    for name in ("fuse-overlayfs", "squashfuse"):
        path = bin_dir / name
        path.write_bytes(name.encode())
        path.chmod(0o755)
    for name in fuse_tools._SQUASHFUSE_LIBRARIES:
        (lib_dir / name).write_bytes(name.encode())
    (bin_dir / "fusermount3").symlink_to("/usr/bin/fusermount")
    fuse_tools._write_manifest(fuse_dir)
    return bin_dir


def _assume_host_fusermount_is_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    is_executable = fuse_tools._is_executable
    monkeypatch.setattr(
        fuse_tools,
        "_is_executable",
        lambda path: path.name == "fusermount3" or is_executable(path),
    )


def test_ensure_fuse_tools_skips_lock_when_tools_are_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assume_host_fusermount_is_executable(monkeypatch)
    _write_complete_cache(tmp_path)

    def unexpected_file_lock(*_args: object, **_kwargs: object) -> None:
        pytest.fail("complete caches must not acquire the download lock")

    monkeypatch.setattr(fuse_tools, "FileLock", unexpected_file_lock)

    fuse_tools.ensure_fuse_tools(tmp_path)


def test_ensure_fuse_tools_creates_world_writable_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fuse_tools, "_fetch_fuse_overlayfs", lambda _path: None)
    monkeypatch.setattr(fuse_tools, "_fetch_squashfuse", lambda _path: None)
    monkeypatch.setattr(fuse_tools, "_ensure_fusermount3_symlink", lambda _path: None)
    monkeypatch.setattr(fuse_tools, "_write_manifest", lambda _path: None)

    fuse_tools.ensure_fuse_tools(tmp_path)

    lock_mode = stat.S_IMODE((tmp_path / ".fuse-tools.lock").stat().st_mode)
    assert lock_mode == 0o666


def test_download_verified_accepts_matching_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "tool"
    payload = b"verified payload"

    def fake_urlopen(
        request: fuse_tools.urllib.request.Request,
        *,
        timeout: int,
    ) -> io.BytesIO:
        assert timeout == 60
        assert request.full_url == "https://example.invalid/tool"
        assert request.get_header("User-agent") == "AlpaSim/1.0"
        return io.BytesIO(payload)

    monkeypatch.setattr(fuse_tools.urllib.request, "urlopen", fake_urlopen)

    fuse_tools._download_verified(
        "https://example.invalid/tool",
        target,
        "3aac0a1146ffe55bac7c05f61401fb1e7e4e6a94110b91585c646fe8cf745f28",
    )

    assert target.read_bytes() == payload


def test_download_verified_rejects_mismatched_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "tool"

    def fake_urlopen(
        request: fuse_tools.urllib.request.Request,
        *,
        timeout: int,
    ) -> io.BytesIO:
        assert timeout == 60
        assert request.get_header("User-agent") == "AlpaSim/1.0"
        return io.BytesIO(b"corrupt")

    monkeypatch.setattr(fuse_tools.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fuse_tools._download_verified(
            "https://example.invalid/tool",
            target,
            "0" * 64,
        )

    assert not target.exists()


def test_download_verified_retries_transient_download_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "tool"
    attempts = 0

    def fake_urlopen(
        request: fuse_tools.urllib.request.Request,
        *,
        timeout: int,
    ) -> io.BytesIO:
        nonlocal attempts
        assert timeout == 60
        assert request.get_header("User-agent") == "AlpaSim/1.0"
        attempts += 1
        if attempts < 3:
            raise TimeoutError("timed out")
        return io.BytesIO(b"verified payload")

    monkeypatch.setattr(fuse_tools.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fuse_tools.time, "sleep", lambda _seconds: None)

    fuse_tools._download_verified(
        "https://example.invalid/tool",
        target,
        "3aac0a1146ffe55bac7c05f61401fb1e7e4e6a94110b91585c646fe8cf745f28",
    )

    assert attempts == 3
    assert target.read_bytes() == b"verified payload"


def test_fetch_squashfuse_installs_package_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def fake_download(_url: str, target: Path, _sha256: str) -> None:
        target.write_bytes(b"deb")

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        assert command[:2] == ["dpkg-deb", "-x"]
        assert check
        assert capture_output
        extracted_root = Path(command[-1])
        extracted_bin = extracted_root / "usr" / "bin"
        extracted_bin.mkdir(parents=True, exist_ok=True)
        (extracted_bin / "squashfuse").write_bytes(b"binary")
        extracted_lib = extracted_root / "usr" / "lib" / "x86_64-linux-gnu"
        extracted_lib.mkdir(parents=True, exist_ok=True)
        for name in ("libsquashfuse.so.0.0.0", "libfuseprivate.so.0.0.0"):
            (extracted_lib / name).write_bytes(name.encode())
        for link_name, target_name in (
            ("libsquashfuse.so.0", "libsquashfuse.so.0.0.0"),
            ("libfuseprivate.so.0", "libfuseprivate.so.0.0.0"),
        ):
            link = extracted_lib / link_name
            link.unlink(missing_ok=True)
            link.symlink_to(target_name)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(fuse_tools, "_download_verified", fake_download)
    monkeypatch.setattr(fuse_tools.subprocess, "run", fake_run)

    fuse_tools._fetch_squashfuse(bin_dir)

    assert (bin_dir / "squashfuse").read_bytes() == b"binary"
    assert (bin_dir / "squashfuse").stat().st_mode & 0o111
    assert not (bin_dir / "squashfuse_ll").exists()
    assert (tmp_path / "lib" / "libsquashfuse.so.0").is_file()
    assert (tmp_path / "lib" / "libfuseprivate.so.0").is_file()


def test_all_present_verifies_manifest_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assume_host_fusermount_is_executable(monkeypatch)
    bin_dir = _write_complete_cache(tmp_path)

    assert fuse_tools._all_present(bin_dir)

    (bin_dir / "squashfuse").write_bytes(b"corrupt")
    assert not fuse_tools._all_present(bin_dir)
