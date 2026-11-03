# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("challenge_local_evaluation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


class LocalEvaluationTest(unittest.TestCase):
    def test_bundled_nuplan_references_form_a_complete_score_matrix(self) -> None:
        inputs, manifest = evaluation.load_reference_inputs("nuplan", None)
        runs = [evaluation.load_run_scores(run) for run in inputs]
        matrix, subjects, scenes, exclusions = evaluation.build_matrix(runs)

        self.assertEqual(manifest["test_suite_id"], "nuplan_private_1000")
        self.assertEqual(matrix.shape, (36, 1000))
        self.assertEqual(len(set(subjects)), 36)
        self.assertEqual(len(set(scenes)), 1000)
        self.assertEqual(exclusions, [])
        self.assertTrue(all(len(run.scene_scores) == 1000 for run in runs))
        self.assertTrue(all(run.is_reference for run in runs))
        self.assertTrue(
            all(run.avg_dist_between_incidents_at_fault is None for run in runs)
        )
        self.assertEqual(
            evaluation.effective_algorithm("zoib", matrix, subjects, scenes),
            ("zoib", None),
        )

    def test_zero_scene_score_is_valid_and_repeated_rollouts_are_averaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "results-summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "scene_score_enabled": True,
                        "rollouts": [
                            {"clipgt_id": "scene-a", "score": 0.0},
                            {"clipgt_id": "scene-a", "score": 1.0},
                            {"clipgt_id": "scene-b", "score": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run = evaluation.load_run_scores(evaluation.RunInput("model", summary_path))

        self.assertEqual(run.scene_scores, {"scene-a": 0.5, "scene-b": 0.25})

    def test_anchor_scale_matches_declared_targets(self) -> None:
        result = SimpleNamespace(
            subject_ids=["low", "candidate", "high"],
            scores=[1.0, 2.0, 4.0],
        )
        scale = evaluation.score_scale(
            result,
            {
                "score_scale": {
                    "low_subject_id": "low",
                    "high_subject_id": "high",
                    "low_target_score": 1000,
                    "high_target_score": 1600,
                }
            },
        )

        self.assertTrue(scale["applied"])
        self.assertEqual(evaluation._apply_score_scale(1.0, scale), 1000.0)
        self.assertEqual(evaluation._apply_score_scale(4.0, scale), 1600.0)

    def test_reference_path_cannot_escape_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "reference_manifest.json"
            with self.assertRaises(ValueError):
                evaluation._safe_reference_path(manifest_path, "../../outside.json")


if __name__ == "__main__":
    unittest.main()
