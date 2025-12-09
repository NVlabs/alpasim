# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Scan SwiftStack for .usdz files, add to sim_scenes.csv, and optionally create a suite.

Usage:
    # Just add scenes (no suite)
    alpasim-scenes-populate \\
        --ss-path=alpasim/artifacts/NRE/my-scenes \\
        --scenes-csv=data/scenes/sim_scenes.csv \\
        --dry-run

    # Add scenes AND create a suite
    alpasim-scenes-populate \\
        --ss-path=alpasim/artifacts/NRE/my-scenes \\
        --scenes-csv=data/scenes/sim_scenes.csv \\
        --suites-csv=data/scenes/sim_suites.csv \\
        --suite-id=my-new-suite \\
        --dry-run

    # Filter out invalid USDZ files (missing mesh_ground.ply)
    alpasim-scenes-populate \\
        --ss-path=alpasim/artifacts/NRE/my-scenes \\
        --scenes-csv=data/scenes/sim_scenes.csv \\
        --filter-valid \\
        --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import asdict
from pathlib import Path

import polars as pl  # type: ignore[import-not-found]
from alpasim_wizard.s3_api import S3Connection, S3Path
from alpasim_wizard.scenes.csv_utils import (
    ArtifactRepository,
    merge_scenes_csv,
    merge_suites_csv,
    validate_csvs,
)
from tqdm.asyncio import tqdm  # type: ignore[import-untyped]

# Project root (5 levels up from this file)
PROJECT_ROOT = Path(__file__).parents[5]

logger = logging.getLogger("alpasim_wizard")


async def scan_and_extract_metadata(
    ss_path: str, filter_valid: bool = False
) -> list[dict]:
    """Scan SwiftStack path and extract metadata from all .usdz files.

    Args:
        ss_path: SwiftStack path to scan for .usdz files.
        filter_valid: If True, filter out USDZ files without mesh_ground.ply.

    Returns:
        List of metadata dictionaries for each valid USDZ file.
    """
    connection = S3Connection(
        aws_access_key_id="team-alpamayo",
        aws_secret_access_key=os.environ["ALPAMAYO_S3_SECRET"],
        region_name="us-east-1",
        endpoint_url="https://pdx.s8k.io",
    )

    path = S3Path.from_swiftstack(ss_path)
    objects = connection.list_objects(path)
    usdzs = [obj for obj in objects if obj.path.key.endswith(".usdz")]
    logger.info(f"Found {len(usdzs)} .usdz files in {ss_path}")

    # Filter for valid USDZ files with mesh_ground.ply if requested
    if filter_valid:
        logger.info(
            "Filtering USDZ files to include only those with mesh_ground.ply..."
        )

        # Use gather to preserve order - results will match the order of usdzs
        has_mesh_ground = await asyncio.gather(
            *[
                asyncio.to_thread(connection.check_usdz_has_mesh_ground, obj)
                for obj in usdzs
            ]
        )

        usdzs_filtered = [
            obj for obj, has_ground in zip(usdzs, has_mesh_ground) if has_ground
        ]
        invalid_count = len(usdzs) - len(usdzs_filtered)

        logger.info(f"Filtered out {invalid_count} USDZ files without mesh_ground.ply")
        logger.info(
            f"{len(usdzs_filtered)} valid USDZ files remaining (with mesh_ground.ply)"
        )

        if invalid_count > 0:
            logger.info("Invalid USDZ files (missing mesh_ground.ply):")
            for obj in [obj for obj in usdzs if obj not in usdzs_filtered]:
                logger.info(f"  - {obj.path.to_swiftstack()}")

        usdzs = usdzs_filtered

        if len(usdzs) == 0:
            logger.info("No valid USDZ files remaining after filtering.")
            return []

    tasks = [asyncio.to_thread(connection.read_usdz_metadata, obj) for obj in usdzs]
    metadata = []
    for f in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Reading metadata"
    ):
        metadata.append(await f)
    return [asdict(m) for m in metadata]


def to_csv_row(
    metadata: dict,
    artifact_repository: ArtifactRepository = ArtifactRepository.SWIFTSTACK,
) -> dict:
    """Convert full metadata to the minimal CSV columns we need.

    Args:
        metadata: The metadata dict from USDZ file.
        artifact_repository: The repository where the artifact is stored.

    Returns:
        A dict with the CSV row data.
    """
    # s3_last_modified might be a datetime object, convert to string
    last_modified = metadata["s3_last_modified"]
    if hasattr(last_modified, "strftime"):
        last_modified = last_modified.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "uuid": metadata["uuid"],
        "scene_id": metadata["scene_id"],
        "nre_version_string": metadata["nre_version_string"],
        "path": metadata["ss_path"],
        "last_modified": last_modified,
        "artifact_repository": str(artifact_repository),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ss-path", required=True, help="SwiftStack path to scan")
    parser.add_argument(
        "--scenes-csv",
        default=str(PROJECT_ROOT / "data" / "scenes" / "sim_scenes.csv"),
        help="Path to sim_scenes.csv",
    )
    parser.add_argument(
        "--suites-csv", help="Path to sim_suites.csv (required if --suite-id is set)"
    )
    parser.add_argument(
        "--suite-id", help="Suite ID to create (adds all found scenes to this suite)"
    )
    parser.add_argument(
        "--create-file",
        action="store_true",
        help="Create the CSV file if it does not exist (otherwise raises error)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print rows without writing"
    )
    parser.add_argument(
        "--filter-valid",
        action="store_true",
        help="Filter out USDZ files that don't contain mesh_ground.ply (required by physics)",
    )
    args = parser.parse_args()

    if args.suite_id and not args.suites_csv:
        parser.error("--suites-csv is required when --suite-id is set")

    logging.basicConfig(level=logging.INFO)

    # Extract metadata from SwiftStack
    all_metadata = asyncio.run(
        scan_and_extract_metadata(args.ss_path, filter_valid=args.filter_valid)
    )

    if len(all_metadata) == 0:
        logger.info("No USDZ files found to process, exiting.")
        return

    new_rows = [to_csv_row(m) for m in all_metadata]
    new_scenes_df = pl.DataFrame(new_rows)

    # --- Dry run output ---
    if args.dry_run:
        print("\n=== Scenes found ===")
        print(new_scenes_df if new_scenes_df.height > 0 else "(none)")

        # Preview deduplication for scenes
        scenes_new, scenes_dup = merge_scenes_csv(
            args.scenes_csv, new_scenes_df, dry_run=True
        )
        print(f"\n=== Merge preview for {args.scenes_csv} ===")
        print(f"  Would add:       {scenes_new} new scenes")
        print(f"  Already exist:   {scenes_dup} scenes (by uuid)")

        # Preview deduplication for suites (if suite-id provided)
        if args.suite_id:
            all_scene_ids = (
                set(new_scenes_df["scene_id"].to_list())
                if new_scenes_df.height > 0
                else set()
            )
            suite_entries = pl.DataFrame(
                [
                    {"test_suite_id": args.suite_id, "scene_id": scene_id}
                    for scene_id in all_scene_ids
                ]
            )
            suites_new, suites_dup = merge_suites_csv(
                args.suites_csv, suite_entries, dry_run=True
            )
            print(f"\n=== Merge preview for {args.suites_csv} ===")
            print(f"  Would add:       {suites_new} new suite entries")
            print(f"  Already exist:   {suites_dup} entries (by suite_id+scene_id)")

        return

    # --- Merge scenes CSV (deduplicates automatically) ---
    scenes_added, _ = merge_scenes_csv(
        args.scenes_csv, new_scenes_df, create_if_missing=args.create_file
    )
    logger.info("Added %d new scenes to %s", scenes_added, args.scenes_csv)

    # --- Merge suites CSV (if suite-id provided) ---
    suites_added = 0
    all_scene_ids = (
        set(new_scenes_df["scene_id"].to_list()) if new_scenes_df.height > 0 else set()
    )
    if args.suite_id:
        suite_entries = pl.DataFrame(
            [
                {"test_suite_id": args.suite_id, "scene_id": scene_id}
                for scene_id in all_scene_ids
            ]
        )
        suites_added, _ = merge_suites_csv(
            args.suites_csv, suite_entries, create_if_missing=args.create_file
        )
        logger.info("Added %d new suite entries to %s", suites_added, args.suites_csv)

    # --- Validate CSVs after modification ---
    if scenes_added > 0 or suites_added > 0:
        validate_csvs(args.scenes_csv, args.suites_csv if suites_added > 0 else None)

    # Remind to update README
    if scenes_added > 0 or suites_added > 0:
        readme_path = os.path.join(os.path.dirname(args.scenes_csv), "README.md")
        print(f"\n{'='*60}")
        print("REMINDER: Update README.md with suite documentation!")
        print(f"  {readme_path}")
        if args.suite_id:
            print(f"\n  New/updated suite: {args.suite_id}")
            print(f"  Scenes in suite: {len(all_scene_ids)}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
