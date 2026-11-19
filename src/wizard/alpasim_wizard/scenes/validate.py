# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Validate clip-catalog and scene-suite CSV files for correctness.

See csv_utils.py:validate_csvs() for more details.


Usage:
    uv run alpasim-scenes-validate
    uv run alpasim-scenes-validate --clip-catalog-csv=path/to/clips.csv --scene-suite-csv=path/to/scene-suite.csv

Exit codes:
    0: Validation passed
    1: Validation failed
"""

from __future__ import annotations

import argparse
import sys

from alpasim_wizard.scenes.csv_utils import CSVValidationError, validate_csvs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--clip-catalog-csv",
        default="data/scenes/clip_catalog.csv",
        help=("Path to clip catalog CSV " "(default: data/scenes/clip_catalog.csv)"),
    )
    parser.add_argument(
        "--scene-suite-csv",
        help="Optional path to one scene-suite CSV",
    )
    args = parser.parse_args()

    try:
        validate_csvs(args.clip_catalog_csv, args.scene_suite_csv)
        print("✓ All validations passed")
        return 0
    except CSVValidationError as e:
        print(f"✗ Validation failed:\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
