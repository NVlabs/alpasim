# Scene Suites

A scene suite is an ordered CSV of clips pinned to artifact UUIDs and event
times. Its filename stem is the suite ID; rows contain only the three fields
below.

The CSV schema is:

```csv
clip_id,uuid,event_t0_us
clipgt-example-a,00000000-0000-0000-0000-000000000000,17400000
clipgt-example-b,11111111-1111-1111-1111-111111111111,AUTO
```

`clip_id` identifies the clip. `uuid` pins its USDZ artifact.
`event_t0_us` is a recording-relative event time in microseconds, or `AUTO`.
For numeric values:

```text
policy_start_us = event_t0_us + t0_offset_us
clip_start_offset_us = policy_start_us - force_gt_duration_us
```

`t0_offset_us` belongs to the experiment's suite entry.
`force_gt_duration_us` belongs to the runtime. `AUTO` selects the runtime's
earliest valid handoff and ignores `t0_offset_us`.

## Selecting scenes

| Mode | Configuration |
| --- | --- |
| Direct catalog clips | `scenes.clip_ids=[<clip-id>,...]` |
| Checked-in suites at zero offset | `scenes.suite_ids=[<suite-id>,...]` |
| Explicit suite configurations | `scenes.suite_configs=[...]` |
| All local USDZs | `scenes.local_usdz_dir=<directory>` |
| Filtered local USDZs | `scenes.local_usdz_dir=<directory>` and `scenes.clip_ids=[...]` |

Select direct catalog clips by ID:

```bash
scenes.clip_ids=[clipgt-example-a,clipgt-example-b]
```

Select checked-in suites with zero offsets by ID:

```bash
scenes.suite_ids=[public_2601,public_2601_video_model]
```

Each ID must resolve to one `<suite-id>.csv` under `scenes.scene_suite_dirs`.
Suites are composed in the listed order.

Use explicit configurations for nonzero offsets or side-loaded CSVs:

```yaml
scenes:
  clip_ids: null
  suite_configs:
    - csv_path: ${repo-relative:'data/scenes/scene_suites/public_2601.csv'}
      t0_offset_us: -1000000
    - csv_path: /path/to/custom_suite.csv
      t0_offset_us: 0
```

Select all local USDZs, or filter them by clip ID:

```bash
# All USDZs under the directory
scenes.local_usdz_dir=/path/to/usdzs

# Only the named clips under the directory
scenes.local_usdz_dir=/path/to/usdzs scenes.clip_ids=[clipgt-example-a]
```

CLI suite and unfiltered-local selectors replace the inherited default clip.
In a composed YAML config, set `scenes.clip_ids: null` explicitly. Other selector
combinations are invalid. `scenes.limit_to_first_n` may be applied to any mode.

`scenes.clip_catalog_csvs` defines the available catalog artifacts, and
`scenes.scene_suite_dirs` defines the checked-in suite search roots; neither is
a selector.

## Adding a suite

1. Add `data/scenes/scene_suites/<suite-id>.csv`.
2. Validate it:

   ```bash
   uv run alpasim-scenes-validate \
     --clip-catalog-csv data/scenes/clip_catalog.csv \
     --scene-suite-csv data/scenes/scene_suites/<suite-id>.csv

   uv run --package alpasim_wizard pytest \
     src/wizard/tests/test_scene_csvs.py \
     src/wizard/tests/test_scene_suite.py
   ```

## Suite-specific context

When a suite has distinct human context, keep a nearby README short and include:

- Purpose
- Selection method and source
- Artifact locations and versions
- Validation performed

## Curated NuRec split

The [AlpaSim E2E challenge
split](https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/data/scenes/sim_suites_curated.csv)
(`curated_train` and `curated_val` ) selects public NuRec scenes from the 26.01
and 26.04 releases. Scenes that share a road are assigned to the same split to
reduce location leakage. The [challenge
documentation](https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/local_evaluation/README.md#curated-nurec-trainvalidation-splits)
describes the selection and exclusions.
