# Local Evaluation

Local evaluation has two complementary pieces: the curated public NuRec
train/validation splits for the Physical AI AV (PAI) track, and the Drive-IRT
aggregation tool for comparing completed runs to organizer-published reference
results. Both run entirely from a local AlpaSim checkout.

## Curated NuRec train/validation splits

An 80/20 train/validation split is curated from the public NuRec 26.01 and
26.04 releases to train and evaluate models before leaderboard submissions.

| suite | scenes | share |
|---|---:|---:|
| `nurec_curated_train` | 1761 | 80% |
| `nurec_curated_val` | 441 | 20% |

### Run a PAI validation suite

These are ordinary test suites, so nothing about running a driver changes.
Follow the driver's usual workflow and add the `nurec_scenes` flag:

```bash
uv run alpasim_wizard +e2e_challenge=dev +nurec_scenes=curated_train ...
uv run alpasim_wizard +e2e_challenge=dev +nurec_scenes=curated_val ...
```

### What was excluded

2202 of 2523 publicly available scenes are included.

The 26.04 catalog carries 1448 of the release's 1607 clips. The other 159 are
re-renders of clips the upstream 26.01 catalog already holds under the same
`scene_id`. The curated splits use the 26.01 render for the 153 of those 159
that they include.

Most of the other missing scenes are excluded because of issues in the
underlying HD maps used for scoring.

### Location leakage

Scenes recorded in the same place can drive the same road, so evaluating a
model on the same road it trained on can be optimistic. Road-sharing scenes
are grouped and assigned to the same split to avoid that leakage. Different
roads or driving directions at the same location may occur across splits.

## Drive-IRT challenge aggregation

This tool fits the same pinned [Drive-IRT](https://github.com/kesai-labs/drive-irt)
algorithm used for the challenge leaderboard to local
`aggregate/results-summary.json` files. It averages repeated rollouts for each
scene, accepts a score of `0.0` as a valid driving result, and creates a
capability ranking, posterior rank interval, rank spread, and average scene
score.

It does not copy a model's results. Give the run directory directly with
`--run MODEL_ID=PATH`; when the published reference bundle is present, its
precomputed runs are added automatically for the selected track.

### Install and run

Run from the AlpaSim repository root. The `local-evaluation` optional extra
pins Drive-IRT to the same revision as the competition aggregator.

```bash
uv run --extra local-evaluation \
  python e2e_challenge/local_evaluation/evaluate.py --help
```

#### Physical AI AV (PAI)

Use the public curated validation split for local PAI model comparison:

```bash
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge=dev +nurec_scenes=curated_val \
  wizard.log_dir=./runs/my-pai-model-val

uv run --extra local-evaluation \
  python e2e_challenge/local_evaluation/evaluate.py \
  --track pai \
  --run my-pai-model=./runs/my-pai-model-val \
  --output-dir ./runs/my-pai-model-val/local-evaluation
```

`curated_val` is the 441-scene holdout defined in
`src/wizard/configs/nurec_scenes/curated_val.yaml`. Do not mix this output with
another scene suite: every run included in a fit must have the same scored
scene IDs.

#### NuPlan / MTGS

First run the same local NuPlan suite as the corresponding published reference
bundle. For example, the existing public full smoke suite is:

```bash
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/worldengine-root \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  nuplan_scenes=navtest_full \
  scenes.limit_to_first_n=0 \
  wizard.log_dir=./runs/my-nuplan-model

uv run --extra local-evaluation \
  python e2e_challenge/local_evaluation/evaluate.py \
  --track nuplan \
  --run my-nuplan-model=./runs/my-nuplan-model \
  --output-dir ./runs/my-nuplan-model/local-evaluation
```

The PAI curated NuRec split and the NuPlan/MTGS scene suites are different;
their reference data is intentionally kept separate.

### Published reference data

`data/` is intentionally empty until organizer-published precomputed reference
results are added. The future bundle layout is:

```text
data/
  pai/reference_manifest.json
  pai/<reference run>/aggregate/results-summary.json
  nuplan/reference_manifest.json
  nuplan/<reference run>/aggregate/results-summary.json
```

Each manifest identifies its track and reference subject IDs. Once present,
the evaluator includes those runs automatically. You may point at a separately
downloaded bundle with `--reference-manifest /path/to/reference_manifest.json`.
It also supplies the two named anchor subjects and their target scores, so the
reported Policy Capability Score uses the same affine scale as the leaderboard.

The required manifest interface is intentionally small:

```json
{
  "track": "pai",
  "test_suite_id": "nurec_curated_val",
  "runs": [
    {"subject_id": "anchor-low", "summary_path": "anchor-low/aggregate/results-summary.json"},
    {"subject_id": "anchor-high", "summary_path": "anchor-high/aggregate/results-summary.json"}
  ],
  "score_scale": {
    "low_subject_id": "anchor-low",
    "low_target_score": 1000,
    "high_subject_id": "anchor-high",
    "high_target_score": 1600
  }
}
```

Until then, `--without-references` allows an experimental comparison of two or
more local runs, but it is not leaderboard-like:

```bash
uv run --extra local-evaluation \
  python e2e_challenge/local_evaluation/evaluate.py \
  --track pai --without-references \
  --run model-a=./runs/model-a --run model-b=./runs/model-b \
  --output-dir ./runs/local-comparison
```

For `zoib`, the tool uses the same sufficiency guard as the service: at least
`S + 5N` observations for `S` subjects and `N` scenes. If there are too few,
it records a warning in `manifest.json` and uses the arithmetic-average
fallback. That fallback has no posterior rank spread.

### Outputs and ranking

- `scene_score_matrix.csv`: one canonical per-scene score per subject.
- `capability_ranking.csv`: policy capability score, average scene score,
  posterior rank interval, and `rank_spread = rank_hi - rank_lo` when
  available.
- `manifest.json`: inputs, exclusions, algorithm, fallback, and fixed ranking
  settings.
- `fit/`: serialized Drive-IRT fit output.

The default is zero-one-inflated beta IRT (`zoib`) with fixed seed and 100,000
posterior rank samples. Rankings use lower `rank_hi` first, then higher
`avg_dist_between_incidents_at_fault`, then point-estimate rank and model ID.
Use `--algorithm average` only for a quick baseline.
