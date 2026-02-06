# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Integration test for the asl-to-roadcast tool.

Uses the git-lfs tracked ASL file to test the full conversion pipeline.
"""

import argparse
import tempfile
from pathlib import Path

import pytest
from asl_to_roadcast.__main__ import transform_from_file


def get_repo_root() -> Path:
    """Get the repository root directory."""
    current = Path(__file__).resolve()
    # Walk up to find the repo root (contains .git directory)
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find repository root")


def get_test_asl_file() -> Path:
    """Get the path to the git-lfs tracked test ASL file."""
    repo_root = get_repo_root()
    asl_file = (
        repo_root / "src" / "runtime" / "tests" / "data" / "integration" / "0.asl"
    )
    return asl_file


@pytest.fixture
def test_asl_file() -> Path:
    """Fixture providing the test ASL file path."""
    asl_file = get_test_asl_file()
    if not asl_file.exists():
        pytest.skip(f"Test ASL file not found: {asl_file}")

    # Check if it's a real file or just an LFS pointer
    if asl_file.stat().st_size < 1000:
        # Likely an LFS pointer, try to pull it
        pytest.skip("ASL file appears to be an LFS pointer. Run 'git lfs pull' first.")

    return asl_file


class TestAslToRoadcastIntegration:
    """Integration tests for the asl-to-roadcast conversion."""

    async def test_converts_asl_to_rclog(self, test_asl_file: Path) -> None:
        """Test that transform_from_file successfully converts an ASL file to RCLog."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Create args namespace matching CLI arguments
            args = argparse.Namespace(
                input_file=str(test_asl_file),
                output_path=str(output_path),
                usdz_glob=None,
                scene_id=None,
                verbose=False,
            )

            # Run the conversion
            result = await transform_from_file(args)

            # Check that conversion succeeded
            assert result == 0, f"transform_from_file returned non-zero: {result}"

            # Check that a 'latest' symlink was created
            latest_link = output_path / "latest"
            assert latest_link.exists(), "Expected 'latest' symlink not found"
            assert latest_link.is_symlink(), "'latest' should be a symlink"

            # Check that the output directory contains the rclog file
            session_dir = latest_link.resolve()
            assert session_dir.exists(), f"Session directory {session_dir} not found"

            rclog_file = session_dir / "roadcast_debug.log"
            assert rclog_file.exists(), f"RCLog file not found at {rclog_file}"

            # Sanity check: file should have reasonable size (not empty, not tiny)
            file_size = rclog_file.stat().st_size
            assert file_size > 1000, (
                f"RCLog file seems too small ({file_size} bytes), "
                "conversion may have failed"
            )

            # Sanity check: output should be reasonable relative to input
            input_size = test_asl_file.stat().st_size
            assert file_size > input_size * 0.01, (
                f"RCLog file ({file_size} bytes) seems suspiciously small "
                f"compared to input ({input_size} bytes)"
            )

    async def test_handles_missing_input_file(self) -> None:
        """Test that transform_from_file raises an error for missing input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                input_file="/nonexistent/path/to/file.asl",
                output_path=tmpdir,
                usdz_glob=None,
                scene_id=None,
                verbose=False,
            )

            with pytest.raises((FileNotFoundError, Exception)):
                await transform_from_file(args)
