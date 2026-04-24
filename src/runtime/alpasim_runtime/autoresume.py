# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
The autocomplete module provides a way to indicate which rollouts completed successfully and which
did not. This is useful for resuming a batch of rollouts that were interrupted.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil

logger = logging.getLogger(__name__)


TRACKER_FILE_NAME = "_complete"


def mark_rollout_complete(save_path_root: str, batch_uuid: str) -> None:
    """
    Mark the rollout as complete by creating a file in the batch directory
    Args:
        save_path_root: asl scene directory (i.e. <wizard_log_dir>/<scene_id>/)
        batch_uuid: the directory name of the batch
    """
    marker_file = os.path.join(
        save_path_root,
        batch_uuid,
        TRACKER_FILE_NAME,
    )
    # touch the file to mark the session as complete. allow existing file to be overwritten
    with open(marker_file, "w"):
        pass


def find_num_complete_rollouts(log_dir_root: str, scene_id: str) -> int:
    """
    Find the number of completed rollouts in the scene directory
    Args:
        log_dir_root: root directory of the logs (i.e. wizard_log_dir)
        scene_id: scene id
    Returns:
        number of completed rollouts as measured by the presence of the tracker file
    """
    save_path_root = os.path.join(log_dir_root, scene_id)
    tracker_file_glob = os.path.join(save_path_root, "*", TRACKER_FILE_NAME)
    tracker_files = glob.glob(tracker_file_glob)
    num_finished_rollouts = len(tracker_files)
    logger.info(
        f"Autoresume {scene_id}: found {num_finished_rollouts} completed rollouts with "
        f"{tracker_file_glob=}"
    )
    return num_finished_rollouts


def remove_incomplete_rollouts(log_dir_root: str, scene_id: str) -> None:
    """
    Remove incomplete rollouts (those without the tracker file) in the scene directory
    Args:
        log_dir_root: root directory of the logs (i.e. wizard_log_dir)
        scene_id: scene id
    """
    save_path_root = os.path.join(log_dir_root, scene_id)
    if not os.path.isdir(save_path_root):  # no existing results
        return
    for item in os.listdir(save_path_root):
        if os.path.isdir(os.path.join(save_path_root, item)) and not os.path.exists(
            os.path.join(save_path_root, item, TRACKER_FILE_NAME)
        ):
            logger.info(
                f"Removing incomplete results: {os.path.join(save_path_root, item)}"
            )
            shutil.rmtree(os.path.join(save_path_root, item))
