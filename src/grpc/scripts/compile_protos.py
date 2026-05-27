#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Compile proto files to Python modules."""
import os
from contextlib import chdir
from pathlib import Path
from typing import Optional, Union

from grpc_tools import command

PathLike = Union[str, os.PathLike[str]]


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_proto_files(root: Optional[PathLike] = None) -> None:
    """Delete all generated proto files (*.py except __init__.py and *.pyi)."""
    root_path = Path(root) if root is not None else _default_root()
    proto_dir = root_path / "alpasim_grpc" / "v0"
    for file_path in proto_dir.rglob("*"):
        if not file_path.is_file():
            continue
        is_generated_python = (
            file_path.name.endswith(".py") and file_path.name != "__init__.py"
        )
        if is_generated_python or file_path.name.endswith(".pyi"):
            print(f"Deleting {file_path}")
            file_path.unlink()


def compile_protos(root: Optional[PathLike] = None) -> None:
    root_path = Path(root) if root is not None else _default_root()

    # First clean old proto files
    print("Cleaning old proto files...")
    clean_proto_files(root_path)

    # Use the same grpc_tools.command API for exact compatibility
    with chdir(root_path):
        command.build_package_protos(".", strict_mode=True)
    print("Proto compilation completed successfully!")
