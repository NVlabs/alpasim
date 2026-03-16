#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Wrapper for Lucas-C `insert-license` pre-commit hook.

Only applies license updates to files changed on the current branch (vs main),
except during actual commits where it processes staged files normally.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _get_changed_files() -> set[str] | None:
    """Get files changed on current branch vs main."""
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF")

    try:
        if from_ref and to_ref:
            diff_spec = f"{from_ref}...{to_ref}"
        else:
            for base in ("origin/main", "main", "origin/master", "master"):
                try:
                    _git("rev-parse", "--verify", "--quiet", base)
                    diff_spec = f"{_git('merge-base', 'HEAD', base)}...HEAD"
                    break
                except subprocess.CalledProcessError:
                    continue
            else:
                return None

        changed = _git("diff", "--name-only", "--diff-filter=ACMRTUXB", diff_spec)
        return set(changed.splitlines())
    except subprocess.CalledProcessError:
        return None


def _get_staged_files() -> set[str]:
    """Get currently staged files."""
    try:
        staged = _git("diff", "--name-only", "--cached", "--diff-filter=ACMRTUXB")
        return set(staged.splitlines())
    except subprocess.CalledProcessError:
        return set()


def _is_commit_time(files: list[str]) -> bool:
    """Check if running during an actual commit (vs --all-files)."""
    staged = _get_staged_files()
    return bool(staged) and set(files).issubset(staged)


def main(argv: list[str] | None = None) -> int:
    from pre_commit_hooks import insert_license

    argv = argv if argv is not None else sys.argv[1:]
    opts = [a for a in argv if a.startswith("-")]
    files = [a for a in argv if not a.startswith("-")]

    if not files:
        return 0

    # Filter to branch changes unless this is an actual commit
    if not _is_commit_time(files):
        changed = _get_changed_files()
        if changed is not None:
            files = [f for f in files if f in changed]

    if not files:
        return 0

    return int(insert_license.main([*opts, *files]))


if __name__ == "__main__":
    raise SystemExit(main())
