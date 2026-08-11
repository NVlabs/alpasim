# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Download FUSE helper binaries for direct Enroot squash launch."""

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock

logger = logging.getLogger(__name__)

# fuse-overlayfs: pre-built static binary from GitHub releases.
_FUSE_OVERLAYFS_VERSION = "1.17"
_FUSE_OVERLAYFS_URL = (
    "https://github.com/containers/fuse-overlayfs/releases/download/"
    f"v{_FUSE_OVERLAYFS_VERSION}/fuse-overlayfs-x86_64"
)
_FUSE_OVERLAYFS_SHA256 = (
    "1684ef18c337702a0378a4e9942802770c83b11aed6a93c445d43e641a1f3c90"
)

# squashfuse: Ubuntu 18.04 packages. Version 0.1.103 supports the uid/gid FUSE
# options used by Enroot while remaining compatible with IAD's glibc and libfuse2.
# Its private libraries are installed beside the tools and added to
# LD_LIBRARY_PATH by the direct-Enroot launcher.
_SQUASHFUSE_DEB_URL = (
    "https://archive.ubuntu.com/ubuntu/pool/universe/s/squashfuse/"
    "squashfuse_0.1.103-2_amd64.deb"
)
_SQUASHFUSE_DEB_SHA256 = (
    "3b51e7b4502876d03a745f017d5910ef43fb367f9f91db8cb517b3f4de84dac7"
)
_LIBSQUASHFUSE_DEB_URL = (
    "https://archive.ubuntu.com/ubuntu/pool/universe/s/squashfuse/"
    "libsquashfuse0_0.1.103-2_amd64.deb"
)
_LIBSQUASHFUSE_DEB_SHA256 = (
    "92f21e67051ed629d2b6117e0d38d48e4899c9ae6edd9f83fcd288b18590ce91"
)
_REQUIRED_TOOLS = ("fuse-overlayfs", "fusermount3", "squashfuse")
_SQUASHFUSE_LIBRARIES = ("libsquashfuse.so.0", "libfuseprivate.so.0")
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_TIMEOUT_SECONDS = 60
_DOWNLOAD_USER_AGENT = "AlpaSim/1.0"
_MANIFEST_VERSION = 1
_MANIFEST_NAME = ".fuse-tools-manifest.json"
_SOURCE_DIGESTS = {
    _FUSE_OVERLAYFS_URL: _FUSE_OVERLAYFS_SHA256,
    _SQUASHFUSE_DEB_URL: _SQUASHFUSE_DEB_SHA256,
    _LIBSQUASHFUSE_DEB_URL: _LIBSQUASHFUSE_DEB_SHA256,
}


def fuse_tool_paths(fuse_dir: str | Path) -> tuple[Path, Path]:
    """Return the managed binary and library directories."""
    root = Path(fuse_dir)
    return root / "bin", root / "lib"


def ensure_fuse_tools(fuse_dir: str | Path) -> None:
    """Ensure direct-Enroot FUSE tools are present under fuse_dir.

    Downloads on first call; subsequent calls are no-ops once the binaries exist.
    Thread- and process-safe via a file lock in fuse_dir.
    """
    fuse_dir = Path(fuse_dir)
    bin_dir, _ = fuse_tool_paths(fuse_dir)
    if _all_present(bin_dir):
        return

    bin_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(str(fuse_dir / ".fuse-tools.lock"), mode=0o666):
        if _all_present(bin_dir):
            return
        _fetch_fuse_overlayfs(bin_dir)
        _fetch_squashfuse(bin_dir)
        _ensure_fusermount3_symlink(bin_dir)
        _write_manifest(fuse_dir)

    logger.info("FUSE tools ready at %s", bin_dir)


def _all_present(bin_dir: Path) -> bool:
    fuse_dir = bin_dir.parent
    manifest_path = fuse_dir / _MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
        expected_files = _managed_file_paths(fuse_dir)
        return (
            manifest["version"] == _MANIFEST_VERSION
            and manifest["source_digests"] == _SOURCE_DIGESTS
            and manifest["file_digests"]
            == {
                str(path.relative_to(fuse_dir)): _file_sha256(path)
                for path in expected_files
            }
            and all(_is_executable(bin_dir / name) for name in _REQUIRED_TOOLS)
            and _squashfuse_libraries_present(bin_dir)
            and (bin_dir / "fusermount3").is_symlink()
            and (bin_dir / "fusermount3").readlink() == Path("/usr/bin/fusermount")
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _write_manifest(fuse_dir: Path) -> None:
    file_digests = {
        str(path.relative_to(fuse_dir)): _file_sha256(path)
        for path in _managed_file_paths(fuse_dir)
    }
    manifest = {
        "version": _MANIFEST_VERSION,
        "source_digests": _SOURCE_DIGESTS,
        "file_digests": file_digests,
    }
    with _atomic_target(fuse_dir / _MANIFEST_NAME) as staging:
        staging.write_text(json.dumps(manifest, sort_keys=True) + "\n")


def _managed_file_paths(fuse_dir: Path) -> tuple[Path, ...]:
    bin_dir, lib_dir = fuse_tool_paths(fuse_dir)
    return (
        bin_dir / "fuse-overlayfs",
        bin_dir / "squashfuse",
        *(lib_dir / name for name in _SQUASHFUSE_LIBRARIES),
    )


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fetch_fuse_overlayfs(bin_dir: Path) -> None:
    target = bin_dir / "fuse-overlayfs"
    logger.info("Downloading fuse-overlayfs %s", _FUSE_OVERLAYFS_VERSION)
    _download_verified(_FUSE_OVERLAYFS_URL, target, _FUSE_OVERLAYFS_SHA256)
    _make_executable(target)


def _fetch_squashfuse(bin_dir: Path) -> None:
    target = bin_dir / "squashfuse"
    logger.info("Downloading squashfuse from Ubuntu package")
    with tempfile.TemporaryDirectory() as tmp:
        deb_path = Path(tmp) / "squashfuse.deb"
        library_deb_path = Path(tmp) / "libsquashfuse.deb"
        _download_verified(
            _SQUASHFUSE_DEB_URL,
            deb_path,
            _SQUASHFUSE_DEB_SHA256,
        )
        _download_verified(
            _LIBSQUASHFUSE_DEB_URL,
            library_deb_path,
            _LIBSQUASHFUSE_DEB_SHA256,
        )
        for package in (deb_path, library_deb_path):
            subprocess.run(
                ["dpkg-deb", "-x", str(package), tmp],
                check=True,
                capture_output=True,
            )
        _install_file(Path(tmp) / "usr" / "bin" / "squashfuse", target)
        _make_executable(target)
        source_lib_dir = Path(tmp) / "usr" / "lib" / "x86_64-linux-gnu"
        _, target_lib_dir = fuse_tool_paths(bin_dir.parent)
        target_lib_dir.mkdir(parents=True, exist_ok=True)
        for name in _SQUASHFUSE_LIBRARIES:
            _install_file((source_lib_dir / name).resolve(), target_lib_dir / name)


def _ensure_fusermount3_symlink(bin_dir: Path) -> None:
    symlink = bin_dir / "fusermount3"
    symlink.unlink(missing_ok=True)
    symlink.symlink_to("/usr/bin/fusermount")


def _download_verified(url: str, target: Path, expected_sha256: str) -> None:
    with _atomic_target(target) as download:
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            download.unlink(missing_ok=True)
            try:
                with (
                    urllib.request.urlopen(
                        urllib.request.Request(
                            url,
                            headers={"User-Agent": _DOWNLOAD_USER_AGENT},
                        ),
                        timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                    ) as response,
                    download.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output)
                break
            except urllib.error.HTTPError as e:
                if e.code not in {408, 429} and e.code < 500:
                    raise
                error: OSError = e
            except OSError as e:
                error = e

            if attempt == _DOWNLOAD_ATTEMPTS:
                raise error
            logger.warning(
                "Download attempt %d/%d failed for %s: %s",
                attempt,
                _DOWNLOAD_ATTEMPTS,
                url,
                error,
            )
            time.sleep(attempt)

        with download.open("rb") as f:
            actual_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {url}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )


def _install_file(source: Path, target: Path) -> None:
    with _atomic_target(target) as staging:
        shutil.copy2(source, staging)


@contextmanager
def _atomic_target(target: Path) -> Iterator[Path]:
    staging = target.with_name(f".{target.name}.staging")
    staging.unlink(missing_ok=True)
    try:
        yield staging
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)


def _squashfuse_libraries_present(bin_dir: Path) -> bool:
    _, lib_dir = fuse_tool_paths(bin_dir.parent)
    return all((lib_dir / name).is_file() for name in _SQUASHFUSE_LIBRARIES)


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
