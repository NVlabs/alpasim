# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Apply Transformers 5 fixes to the Git SDKs installed in this environment."""

import json
import subprocess
from importlib.metadata import distribution
from pathlib import Path

PATCHES = (
    ("alpamayo-r1", "alpamayo-r1.patch"),
    ("alpamayo1-5", "alpamayo1.5.patch"),
    ("alpamayo2-super", "alpamayo2.patch"),
)


def main() -> None:
    plans = []
    for package, filename in PATCHES:
        installed = distribution(package)
        origin = json.loads(installed.read_text("direct_url.json"))
        commit = origin["vcs_info"]["commit_id"]
        patch = Path(__file__).resolve().parent / filename
        command = [
            "git",
            "apply",
            "--unsafe-paths",
            f"--directory={installed.locate_file('')}",
            "-p2",  # a/src/package/file.py -> package/file.py in site-packages
            str(patch),
        ]
        check = subprocess.run([*command, "--check"], capture_output=True, text=True)
        if check.returncode == 0:
            plans.append((package, commit, command, True))
        else:
            reverse = subprocess.run(
                [*command, "--check", "--reverse"], capture_output=True, text=True
            )
            if reverse.returncode != 0:
                raise RuntimeError(
                    f"{filename} does not match installed {package} at {commit}. "
                    f"Review the upstream changes before updating the patch.\n{check.stderr}"
                )
            plans.append((package, commit, command, False))

    # Validate all three packages before modifying any of them.
    for package, commit, command, needs_patch in plans:
        if needs_patch:
            subprocess.run(command, check=True)
        status = "patched" if needs_patch else "already applied"
        print(f"{package}@{commit}: {status}")


if __name__ == "__main__":
    main()
