# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
import os
from pathlib import Path

import pytest
from alpasim_utils.nav_route import find_route_indices


def test_route_indices_are_resolved_across_suite_subtrees(tmp_path: Path) -> None:
    first = tmp_path / "public_2601" / "aaa.pkl"
    second = tmp_path / "public_2604" / "nested" / "bbb.pkl"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_bytes(path.stem.encode())
    (first.parent / "unrequested.pkl").write_bytes(b"other clip")

    indices = find_route_indices(tmp_path, {"clipgt-aaa", "bbb", "clipgt-missing"})

    assert indices == {"clipgt-aaa": first, "bbb": second}


@pytest.mark.parametrize("conflicting", [False, True])
def test_overlapping_suites_require_identical_clip_payloads(
    tmp_path: Path, conflicting: bool
) -> None:
    first = tmp_path / "suite-a" / "aaa.pkl"
    second = tmp_path / "suite-b" / "aaa.pkl"
    for path in (first, second):
        path.parent.mkdir()
    first.write_bytes(b"first")
    # Equal size and mtime must not disguise different route contents.
    second.write_bytes(b"other" if conflicting else b"first")
    os.utime(second, ns=(first.stat().st_atime_ns, first.stat().st_mtime_ns))

    if conflicting:
        with pytest.raises(ValueError, match="conflicting route indices"):
            find_route_indices(tmp_path, {"clipgt-aaa"})
    else:
        assert find_route_indices(tmp_path, {"clipgt-aaa"}) == {"clipgt-aaa": first}


@pytest.mark.parametrize("conflicting", [False, True])
def test_published_files_map_clips_and_require_identical_duplicates(
    tmp_path: Path, conflicting: bool
) -> None:
    first = tmp_path / "route_indices_a.json"
    second = tmp_path / "route_indices_b.json"
    first.write_text(json.dumps({"aaa": {"x": 1}, "bbb": {"x": 2}}))
    second.write_text(json.dumps({"aaa": {"x": 3 if conflicting else 1}}))

    if conflicting:
        with pytest.raises(ValueError, match="conflicting route indices"):
            find_route_indices(tmp_path, {"clipgt-aaa"})
    else:
        assert find_route_indices(tmp_path, {"clipgt-aaa", "bbb", "ccc"}) == {
            "clipgt-aaa": first,
            "bbb": first,
        }


def test_other_json_files_beside_route_indices_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "suite").mkdir()
    (tmp_path / "suite" / "aaa.pkl").write_bytes(b"route")
    (tmp_path / "suite" / "metadata.json").write_text(json.dumps(["not", "a", "map"]))

    assert find_route_indices(tmp_path, {"clipgt-aaa"}) == {
        "clipgt-aaa": tmp_path / "suite" / "aaa.pkl"
    }
