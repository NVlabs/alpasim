# Local Evaluation

Local evaluation has two complementary pieces: public scene selections for
closed-loop runs, and the Drive-IRT aggregation tool for analyzing completed
runs. When an organizer-published reference bundle is installed, the tool can
also compare a run with those reference results. Everything runs from a local
AlpaSim checkout.

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
scene and creates a
capability ranking, posterior rank interval, rank spread, and average scene
score.

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
  --without-references --algorithm average \
  --run my-pai-model=./runs/my-pai-model-val \
  --output-dir ./runs/my-pai-model-val/local-evaluation
```

The command above works without a reference bundle and reports the model's
average scene score. After installing a published PAI reference bundle, remove
`--without-references --algorithm average` to run the reference-based Drive-IRT
comparison.

`curated_val` is the 441-scene holdout defined in
`src/wizard/configs/nurec_scenes/curated_val.yaml`. Do not mix this output with
another scene suite: every run included in a fit must have the same scored
scene IDs.

#### NuPlan / MTGS

First run the standard full `navtest` suite used by the corresponding published
reference bundle:

```bash
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/worldengine-root \
uv run alpasim_wizard +e2e_challenge_nuplan=full \
  wizard.log_dir=./runs/my-nuplan-model

uv run --extra local-evaluation \
  python e2e_challenge/local_evaluation/evaluate.py \
  --track nuplan \
  --without-references --algorithm average \
  --run my-nuplan-model=./runs/my-nuplan-model \
  --output-dir ./runs/my-nuplan-model/local-evaluation
```

The command above works without a reference bundle and reports the model's
average navtest scene score. After installing a published nuPlan reference
bundle, remove `--without-references --algorithm average` to run the
reference-based Drive-IRT comparison.

The PAI curated NuRec split and the NuPlan/MTGS scene suites are different;
their reference data is intentionally kept separate.

### Published reference data

`data/pai/` ships with precomputed PAI reference runs on the 441-scene
`nurec_curated_val` split; see "How the PAI reference runs were produced" below
for what each subject is. The bundle layout is:

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
It also supplies the two named anchor subjects and their target scores used to
scale the reported Policy Capability Score (see "Anchor scale").

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

For a track with no installed reference bundle, `--without-references` allows an
experimental comparison of two or more local runs:

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

### How the PAI reference runs were produced

Every PAI reference subject is a full `nurec_curated_val` run (441 scenes x 3
rollouts) under the competition simulation contract, so a locally produced run
is directly comparable. Reproduce one with:

```bash
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge=dev +nurec_scenes=curated_val \
  runtime.simulation_config.n_rollouts=3 \
  wizard.log_dir=./runs/my-subject
```

`+e2e_challenge=dev` is what pins the contract: the competition sensor list at
10 Hz, 200 simulation steps and 1.7 s of force-GT. Selecting a driver alone is
not enough, because a driver config can carry its own camera and timing
overrides (`driver/vavam_configs.yaml` replaces them with a 1-camera 2 Hz
shape). `n_rollouts=3` matches the references.

The controller is a deliberate axis here, not an incidental setting. The base
config selects `controller: nonlinear`; `controller=default` is the linear MPC.
Be aware that this option is not available in the actual competition, but the
linear MPC does not have as strong coupling between the lateral and longitudinal
axes, which allows for better robustness to some of the infeasible trajectories
that VaVAM produces.

| subject | driver | controller | provenance |
|---|---|---|---|
| `alpamayo1` | in-repo Alpamayo 1 | nonlinear | reproducible with the command above |
| `vavam-nonlinear` | VAVAM submission image | nonlinear | reproducible with the command above |
| `vavam-linear` | VAVAM submission image | linear (`controller=default`) | reproducible with the command above |
| `alternative_<n>` | not recorded | not recorded | generated with other policies; no further details available |

### Anchor scale

`data/pai/reference_manifest.json` maps the Policy Capability Score (PCS, the
`policy_capability_score` column) onto a fixed affine scale defined by two
anchor subjects:

| role | subject | target |
|---|---:|---:|
| low | `alternative_2` | 600 |
| high | `alpamayo1` | 2000 |

The two anchors reproduce their targets exactly; every other subject, including
a locally supplied one, is placed on the line through them. A subject weaker
than the low anchor extrapolates below 1000.

**These numbers are an arbitrary local scale.** The anchor subjects and their
targets were chosen for this bundle alone, so a local PCS is not comparable to
the competition leaderboard and is not expected to match it.

### Subject count and the zoib sufficiency guard

The `zoib` algorithm needs `S + 5N` observations for `S` subjects and `N`
scenes. On the 441-scene split each subject contributes 441 canonical scores,
so `441S >= S + 2205`, which means **six subjects minimum**. Below that the tool
records a warning in `manifest.json` and silently falls back to an arithmetic
average, which produces no posterior rank interval and no `rank_spread`.

### Build the local leaderboard end to end

```bash
# 1. run your driver on the same split as the references
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge=dev +nurec_scenes=curated_val \
  runtime.simulation_config.n_rollouts=3 \
  wizard.log_dir=./runs/my-model

# 2. copy it into the bundle and add a "runs" entry for it in
#    data/pai/reference_manifest.json:
#      {"subject_id": "my-model",
#       "summary_path": "my-model/aggregate/results-summary.json"}
mkdir -p e2e_challenge/local_evaluation/data/pai/my-model/aggregate
cp ./runs/my-model/aggregate/results-summary.json \
   e2e_challenge/local_evaluation/data/pai/my-model/aggregate/

# 3. rank every subject in the bundle
uv run --extra local-evaluation \
  python e2e_challenge/local_evaluation/evaluate.py \
  --track pai --output-dir ./runs/my-model/local-evaluation

# 4. confirm the fit was not the fallback
python -c "import json;print(json.load(open('./runs/my-model/local-evaluation/manifest.json')).get('warnings') or 'zoib fit used')"
```

Subjects listed in the manifest are picked up automatically, so no `--run` is
needed. Use `--run MODEL_ID=PATH` only for a run you want to keep outside the
bundle. `--track` is required because the challenge aggregator is likewise
invoked per track.
