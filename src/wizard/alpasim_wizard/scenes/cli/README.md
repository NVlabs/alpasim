# Scene Management CLI Tools

Command-line tools for managing Alpasim scene and test suite definitions.

## Overview

Scenes and suites are stored in CSV files at `data/scenes/`:
- `sim_scenes.csv` - Scene artifact metadata (uuid, scene_id, NRE version, path, artifact_repository)
- `sim_suites.csv` - Suite-to-scene mappings (which scenes belong to which test suites)

### Artifact Repositories

The `artifact_repository` column indicates where scene files are stored:
- `swiftstack` - SwiftStack/S3 storage (current default for all tools)
- `huggingface` - HuggingFace Hub (planned for future support)

## Available Commands

### `alpasim-scenes-populate`

Scan SwiftStack for `.usdz` files and add them to the scenes CSV. Optionally create a suite.

```bash
# Add scenes from SwiftStack (no suite)
alpasim-scenes-populate \
    --ss-path=alpasim/artifacts/NRE/my-scenes \
    --scenes-csv=data/scenes/sim_scenes.csv \
    --dry-run

# Add scenes AND create/update a suite
alpasim-scenes-populate \
    --ss-path=alpasim/artifacts/NRE/my-scenes \
    --scenes-csv=data/scenes/sim_scenes.csv \
    --suites-csv=data/scenes/sim_suites.csv \
    --suite-id=my-new-suite \
    --dry-run

# Filter out invalid USDZ files (missing mesh_ground.ply required by physics)
alpasim-scenes-populate \
    --ss-path=alpasim/artifacts/NRE/my-scenes \
    --scenes-csv=data/scenes/sim_scenes.csv \
    --filter-valid \
    --dry-run
```

**Options:**
- `--ss-path` (required): SwiftStack path to scan for `.usdz` files
- `--scenes-csv`: Path to scenes CSV (default: `data/scenes/sim_scenes.csv`)
- `--suites-csv`: Path to suites CSV (required if `--suite-id` is set)
- `--suite-id`: Suite ID to create (adds all found scenes to this suite)
- `--filter-valid`: Filter out USDZ files without `mesh_ground.ply`
- `--dry-run`: Preview changes without writing
- `--create-file`: Create CSV file if it doesn't exist

### `alpasim-scenes-migrate`

Migrate scenes from Kratos database to CSV files. Use this for one-time migration of existing suites.

```bash
# Migrate an existing suite
alpasim-scenes-migrate \
    --suite-id=dev.alpasim.unit_tests.v0 \
    --nre-versions=0.2.220-1777390b,25_7_9-dc9a8043 \
    --scenes-csv=data/scenes/sim_scenes.csv \
    --suites-csv=data/scenes/sim_suites.csv \
    --dry-run

# Migrate a suite and rename it
alpasim-scenes-migrate \
    --suite-id=dev.alpasim.unit_tests.v0 \
    --new-suite-id=my-renamed-suite \
    --nre-versions=0.2.220-1777390b,25_7_9-dc9a8043 \
    --scenes-csv=data/scenes/sim_scenes.csv \
    --suites-csv=data/scenes/sim_suites.csv \
    --dry-run

# Migrate specific scenes (not a suite)
alpasim-scenes-migrate \
    --scene-ids=scene_001,scene_002,scene_003 \
    --new-suite-id=my-new-suite \
    --nre-versions=0.2.220-1777390b,25_7_9-dc9a8043 \
    --scenes-csv=data/scenes/sim_scenes.csv \
    --suites-csv=data/scenes/sim_suites.csv \
    --dry-run
```

**Options:**
- `--suite-id`: Existing suite ID to migrate from Kratos
- `--scene-ids`: Comma-separated list of specific scene IDs to migrate
- `--new-suite-id`: New suite name (for renaming or when migrating individual scenes)
- `--nre-versions` (required): Comma-separated NRE versions to include
- `--scenes-csv`: Path to scenes CSV (default: `data/scenes/sim_scenes.csv`)
- `--suites-csv`: Path to suites CSV (default: `data/scenes/sim_suites.csv`)
- `--dry-run`: Preview changes without writing

### `alpasim-scenes-validate`

Validate CSV files for correctness and consistency.

```bash
alpasim-scenes-validate
alpasim-scenes-validate --scenes-csv=data/scenes/sim_scenes.csv --suites-csv=data/scenes/sim_suites.csv
```

**Checks performed:**
- Required columns present
- No duplicate UUIDs or (scene_id, nre_version_string) pairs
- Valid UUID, scene_id, and timestamp formats
- Valid artifact_repository values (swiftstack, huggingface)
- No empty/null values in required fields
- All scene_ids in suites exist in scenes

### `alpasim-scenes-car2sim`

Create test suites from Car2Sim workflow runs. This is a multi-phase workflow for importing scenes from the perception team's S3 bucket.

**Prerequisites:**
- Access to the perception bucket (`PERCEPTION_GT_S3_SECRET` environment variable)
- Run from `tools/run-on-ord` directory on ORD cluster

```bash
# Phase 1: Download scenes from perception bucket (run on data copier node)
alpasim-scenes-car2sim --workflow_run=2025.03.13-1828-35ez1fcn4uu4f --phase=download

# Phase 2: Sanity check scenes (run on login node, repeat until all pass)
alpasim-scenes-car2sim --workflow_run=2025.03.13-1828-35ez1fcn4uu4f --phase=iterate_sanity_check

# Phase 3: Upload to alpasim bucket and update CSVs (run on data copier node)
alpasim-scenes-car2sim --workflow_run=2025.03.13-1828-35ez1fcn4uu4f --phase=upload
```

**Options:**
- `--workflow_run` (required): Workflow run ID (e.g., `2025.03.13-1828-35ez1fcn4uu4f`)
- `--phase` (required): Phase to run (`download`, `iterate_sanity_check`, `upload`)
- `--account`: Slurm account name (default: `nvr_av_end2endav`)
- `--scenes-csv`: Path to scenes CSV (default: `data/scenes/sim_scenes.csv`)
- `--suites-csv`: Path to suites CSV (default: `data/scenes/sim_suites.csv`)

**Finding workflow runs:**
- Check the [car2sim Slack channel](https://nvidia.enterprise.slack.com/archives/C085LUZJKRB)
- Browse [workflows by Konstantinos Zampogianni](https://maglev.aws-us-west-2.prod.nvda.ai/ide/workflows/?user=kzampogianni@nvidia.com)

## Workflow: Adding New Scenes

1. Upload `.usdz` files to SwiftStack:
   ```bash
   rclone copy -P /path/to/artifacts pbss-team-alpamayo:alpasim/artifacts/NRE/my-suite-name
   ```

2. Scan and add to CSV:
   ```bash
   alpasim-scenes-populate \
       --ss-path=alpasim/artifacts/NRE/my-suite-name \
       --scenes-csv=data/scenes/sim_scenes.csv \
       --suites-csv=data/scenes/sim_suites.csv \
       --suite-id=my-suite-name \
       --filter-valid
   ```

3. Validate:
   ```bash
   alpasim-scenes-validate
   ```

4. Update `data/scenes/README.md` with suite documentation

5. Commit and push changes

## Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `ALPAMAYO_S3_SECRET` | SwiftStack S3 access key | `populate`, `car2sim` |
| `PERCEPTION_GT_S3_SECRET` | Perception team bucket access | `car2sim` |
| `PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID` | Kratos database access | `migrate`, `car2sim` |
| `PRODUCTION_KRATOS_CLI_SSA_CLIENT_SECRET` | Kratos database secret | `migrate`, `car2sim` |

## Scene Cache

On ORD, `.usdz` files are automatically cached to a shared location to minimize downloads.
The cache location is configured in `base_config.yaml` under `scenes.cache_dir`.

A `.lock` file is created during downloads to prevent concurrent jobs from downloading the same file.
