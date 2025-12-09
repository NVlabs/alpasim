# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Validate scene and suite CSV files for correctness.

See csv_utils.py:validate_csvs() for more details.


Usage:
    uv run alpasim-scenes-validate
    uv run alpasim-scenes-validate --scenes-csv=path/to/scenes.csv --suites-csv=path/to/suites.csv

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
        "--scenes-csv",
        default="data/scenes/sim_scenes.csv",
        help="Path to scenes CSV (default: data/scenes/sim_scenes.csv)",
    )
    parser.add_argument(
        "--suites-csv",
        default="data/scenes/sim_suites.csv",
        help="Path to suites CSV (default: data/scenes/sim_suites.csv)",
    )
    args = parser.parse_args()

    try:
        validate_csvs(args.scenes_csv, args.suites_csv)
        print("✓ All validations passed")
        return 0
    except CSVValidationError as e:
        print(f"✗ Validation failed:\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
