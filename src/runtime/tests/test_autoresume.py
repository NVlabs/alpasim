# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import os

from alpasim_runtime.autoresume import (
    find_num_complete_rollouts,
    mark_rollout_complete,
    remove_incomplete_rollouts,
)


def test_autoresume_mark_complete_and_remove_incomplete(tmp_path):
    batch_uuid_1 = "uuid_1"
    batch_uuid_2 = "uuid_2"
    os.mkdir(tmp_path / "scene_id")
    SCENE_ID = "scene_id"
    scene_dir = tmp_path / SCENE_ID
    os.mkdir(scene_dir / batch_uuid_1)
    os.mkdir(scene_dir / batch_uuid_2)

    num_complete = find_num_complete_rollouts(tmp_path, SCENE_ID)
    assert num_complete == 0

    mark_rollout_complete(scene_dir, batch_uuid_1)

    num_complete = find_num_complete_rollouts(tmp_path, SCENE_ID)
    assert num_complete == 1

    num_dirs_before_remove = len(os.listdir(scene_dir))
    assert num_dirs_before_remove == 2

    remove_incomplete_rollouts(tmp_path, SCENE_ID)
    num_dirs_after_remove = len(os.listdir(scene_dir))
    assert num_dirs_after_remove == 1
