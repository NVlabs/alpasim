# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Migrate scenes from Kratos to local CSV files.

Supports two modes:
  --suite-id     Migrate an existing suite from Kratos. Optionally use
                 --new-suite-id to rename the suite in the output.
  --scene-ids    Migrate specific scenes by ID. Optionally use --new-suite-id
                 to also add the scenes to a suite.

Usage:
    # Migrate an existing suite from Kratos
    alpasim-scenes-migrate \\
        --suite-id=dev.alpasim.unit_tests.v0 \\
        --nre-versions=0.2.220-1777390b,25_7_9-dc9a8043 \\
        --scenes-csv=data/scenes/sim_scenes.csv \\
        --suites-csv=data/scenes/sim_suites.csv \\
        --dry-run

    # Migrate a suite and rename it
    alpasim-scenes-migrate \\
        --suite-id=dev.alpasim.unit_tests.v0 \\
        --new-suite-id=my-renamed-suite \\
        --nre-versions=0.2.220-1777390b,25_7_9-dc9a8043 \\
        --scenes-csv=data/scenes/sim_scenes.csv \\
        --suites-csv=data/scenes/sim_suites.csv \\
        --dry-run

    # Migrate specific scenes (without adding to a suite)
    alpasim-scenes-migrate \\
        --scene-ids=scene_001,scene_002,scene_003 \\
        --nre-versions=0.2.220-1777390b,25_7_9-dc9a8043 \\
        --scenes-csv=data/scenes/sim_scenes.csv \\
        --dry-run

    # Migrate specific scenes and add them to a suite
    alpasim-scenes-migrate \\
        --scene-ids=scene_001,scene_002,scene_003 \\
        --new-suite-id=my-new-suite \\
        --nre-versions=0.2.220-1777390b,25_7_9-dc9a8043 \\
        --scenes-csv=data/scenes/sim_scenes.csv \\
        --suites-csv=data/scenes/sim_suites.csv \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from pathlib import Path

import polars as pl  # type: ignore[import-not-found]
from alpasim_wizard.scenes.csv_utils import (
    ArtifactRepository,
    merge_scenes_csv,
    merge_suites_csv,
    validate_csvs,
)

# Project root (5 levels up from this file)
PROJECT_ROOT = Path(__file__).parents[5]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENES_TABLE = "kratos_hive.alpamayo.sim_scenes_staging_003"
SUITES_TABLE = "kratos_hive.alpamayo.sim_suites_staging_007"
KRATOS_WAREHOUSE_ID = "b5e6b53af7c54eee92d71717905059d3"


def query_kratos(query: str) -> pl.DataFrame:
    """Execute a query against Kratos and return results as DataFrame."""
    import kratos.drs_jobs

    response = kratos.drs_jobs.execute_drs_adhoc_job(
        profile="production",
        warehouse_id=KRATOS_WAREHOUSE_ID,
        query=query,
        file_format="parquet",
    )

    if "jobId" not in response["result"]:
        raise RuntimeError("No jobId returned by adhoc query")

    job_id = response["result"]["jobId"]

    # Wait for query to complete
    for _ in range(30):
        if "parquetDirectory" in response["result"]:
            break
        logger.info("Waiting for query response...")
        time.sleep(10)
        response = kratos.drs_jobs.get_drs_job(profile="production", job_id=job_id)
    else:
        raise RuntimeError("Timeout waiting for query")

    if response["result"]["parquetSizeInMb"] == 0:
        return pl.DataFrame()

    parquet_dir = response["result"]["parquetDirectory"]
    df = pl.read_parquet(parquet_dir + "/combined.parquet")
    shutil.rmtree(parquet_dir, ignore_errors=True)
    return df


def query_suite_scenes(
    suite_id: str, nre_versions: list[str], where_clause: str | None = None
) -> pl.DataFrame:
    """Query Kratos for all scenes in a suite with specified NRE versions."""
    nre_versions_sql = ", ".join(f"'{v}'" for v in nre_versions)

    query = f"""
    SELECT DISTINCT
        sc.uuid,
        sc.scene_id,
        sc.nre_version_string,
        sc.ss_path as path,
        sc.s3_last_modified as last_modified
    FROM {SUITES_TABLE} s
    JOIN {SCENES_TABLE} sc ON s.scene_id = sc.scene_id
    WHERE s.test_suite_id = '{suite_id}'
      AND sc.nre_version_string IN ({nre_versions_sql})
    """

    if where_clause:
        query += f" AND ({where_clause})"

    logger.info(
        "Querying scenes for suite '%s' with versions %s", suite_id, nre_versions
    )
    df = query_kratos(query)

    if df.height == 0:
        raise RuntimeError(
            f"No scenes found for suite '{suite_id}' with versions {nre_versions}"
        )

    # Deduplicate: keep most recent per scene_id
    df = df.sort("last_modified", descending=True).unique(
        subset=["scene_id"], keep="first"
    )

    # Add artifact_repository column - Kratos data comes from SwiftStack
    df = df.with_columns(
        pl.lit(str(ArtifactRepository.SWIFTSTACK)).alias("artifact_repository")
    )

    logger.info("Found %d unique scenes", df.height)
    return df


def query_scenes_by_id(
    scene_ids: list[str], nre_versions: list[str], where_clause: str | None = None
) -> pl.DataFrame:
    """Query Kratos for scenes by scene_id with specified NRE versions."""
    scene_ids_sql = ", ".join(f"'{s}'" for s in scene_ids)
    nre_versions_sql = ", ".join(f"'{v}'" for v in nre_versions)

    query = f"""
    SELECT DISTINCT
        sc.uuid,
        sc.scene_id,
        sc.nre_version_string,
        sc.ss_path as path,
        sc.s3_last_modified as last_modified
    FROM {SCENES_TABLE} sc
    WHERE sc.scene_id IN ({scene_ids_sql})
      AND sc.nre_version_string IN ({nre_versions_sql})
    """

    if where_clause:
        query += f" AND ({where_clause})"

    logger.info("Querying %d scenes with versions %s", len(scene_ids), nre_versions)
    df = query_kratos(query)

    if df.height == 0:
        raise RuntimeError(
            f"No scenes found for scene_ids {scene_ids} with versions {nre_versions}"
        )

    # Deduplicate: keep most recent per scene_id
    df = df.sort("last_modified", descending=True).unique(
        subset=["scene_id"], keep="first"
    )

    # Add artifact_repository column - Kratos data comes from SwiftStack
    df = df.with_columns(
        pl.lit(str(ArtifactRepository.SWIFTSTACK)).alias("artifact_repository")
    )

    logger.info("Found %d unique scenes", df.height)
    return df


def query_suite_membership(suite_id: str) -> pl.DataFrame:
    """Query Kratos for suite membership (scene_ids in suite)."""
    query = f"""
    SELECT test_suite_id, scene_id
    FROM {SUITES_TABLE}
    WHERE test_suite_id = '{suite_id}'
    """
    return query_kratos(query)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite-id", help="Suite ID to migrate from Kratos")
    parser.add_argument("--scene-ids", help="Comma-separated scene IDs to migrate")
    parser.add_argument(
        "--new-suite-id",
        help="Suite ID for the output (optional; renames suite or adds scenes to this suite)",
    )
    parser.add_argument(
        "--nre-versions", required=True, help="Comma-separated NRE versions"
    )
    parser.add_argument(
        "--scenes-csv",
        default=str(PROJECT_ROOT / "data" / "scenes" / "sim_scenes.csv"),
        help="Path to scenes CSV",
    )
    parser.add_argument(
        "--suites-csv",
        default=str(PROJECT_ROOT / "data" / "scenes" / "sim_suites.csv"),
        help="Path to suites CSV",
    )
    parser.add_argument(
        "--where",
        help="Optional SQL WHERE clause segment to filter scenes (e.g. \"sc.s3_last_modified > '2024-01-01'\")",
    )
    parser.add_argument(
        "--create-file",
        action="store_true",
        help="Create the CSV file if it does not exist (otherwise raises error)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be added without writing",
    )
    args = parser.parse_args()

    # Validate mutually exclusive options
    if not args.suite_id and not args.scene_ids:
        parser.error("Must provide either --suite-id or --scene-ids")
    if args.suite_id and args.scene_ids:
        parser.error("Cannot use both --suite-id and --scene-ids")

    nre_versions = [v.strip() for v in args.nre_versions.split(",")]

    # --- Query Kratos ---
    if args.suite_id:
        # Mode 1: Migrate an existing suite
        kratos_scenes = query_suite_scenes(
            args.suite_id, nre_versions, where_clause=args.where
        )
        kratos_suite = query_suite_membership(args.suite_id)
        output_suite_id = args.new_suite_id or args.suite_id

        # Rename suite if requested
        if args.new_suite_id:
            kratos_suite = kratos_suite.with_columns(
                pl.lit(args.new_suite_id).alias("test_suite_id")
            )
            logger.info(
                "Renaming suite from '%s' to '%s'", args.suite_id, args.new_suite_id
            )

        # Filter suite membership to only include scenes that matched the query (e.g. if --where was used)
        found_scene_ids = kratos_scenes["scene_id"].to_list()
        kratos_suite = kratos_suite.filter(pl.col("scene_id").is_in(found_scene_ids))

        suite_entries = kratos_suite
    else:
        # Mode 2: Migrate specific scenes
        scene_ids = [s.strip() for s in args.scene_ids.split(",")]
        kratos_scenes = query_scenes_by_id(
            scene_ids, nre_versions, where_clause=args.where
        )
        output_suite_id = args.new_suite_id

        # Build suite entries from the found scenes (only if suite specified)
        if args.new_suite_id:
            found_scene_ids = kratos_scenes["scene_id"].to_list()
            suite_entries = pl.DataFrame(
                [
                    {"test_suite_id": args.new_suite_id, "scene_id": sid}
                    for sid in found_scene_ids
                ]
            )
        else:
            suite_entries = pl.DataFrame()

    logger.info(
        "Found %d scenes from Kratos%s",
        kratos_scenes.height,
        f", {suite_entries.height} suite entries" if suite_entries.height > 0 else "",
    )

    # --- Dry run output ---
    if args.dry_run:
        print("\n=== Scenes from Kratos ===")
        print(kratos_scenes if kratos_scenes.height > 0 else "(none)")

        # Preview deduplication for scenes
        scenes_new, scenes_dup = merge_scenes_csv(
            args.scenes_csv, kratos_scenes, dry_run=True
        )
        print(f"\n=== Merge preview for {args.scenes_csv} ===")
        print(f"  Would add:       {scenes_new} new scenes")
        print(f"  Already exist:   {scenes_dup} scenes (by uuid)")

        if output_suite_id:
            print(f"\n=== Suite entries for '{output_suite_id}' ===")
            print(suite_entries if suite_entries.height > 0 else "(none)")

            # Preview deduplication for suites
            if suite_entries.height > 0 and args.suites_csv:
                suites_new, suites_dup = merge_suites_csv(
                    args.suites_csv, suite_entries, dry_run=True
                )
                print(f"\n=== Merge preview for {args.suites_csv} ===")
                print(f"  Would add:       {suites_new} new suite entries")
                print(f"  Already exist:   {suites_dup} entries (by suite_id+scene_id)")
        else:
            print("\n(No suite assignment requested)")
        return

    # --- Merge into CSVs (deduplicates automatically) ---
    scenes_added, _ = merge_scenes_csv(
        args.scenes_csv, kratos_scenes, create_if_missing=args.create_file
    )
    logger.info("Added %d new scenes to %s", scenes_added, args.scenes_csv)

    suites_added = 0
    if suite_entries.height > 0:
        suites_added, _ = merge_suites_csv(
            args.suites_csv, suite_entries, create_if_missing=args.create_file
        )
        logger.info("Added %d new suite entries to %s", suites_added, args.suites_csv)

    # --- Validate CSVs after modification ---
    if scenes_added > 0 or suites_added > 0:
        validate_csvs(args.scenes_csv, args.suites_csv if suites_added > 0 else None)

    # Remind to update README
    if suites_added > 0:
        readme_path = os.path.join(os.path.dirname(args.scenes_csv), "README.md")
        print(f"\n{'='*60}")
        print("REMINDER: Update the README with suite documentation!")
        print(f"  {readme_path}")
        print(f"\n  Suite: {output_suite_id}")
        print(f"  Scenes added: {scenes_added}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
