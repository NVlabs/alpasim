# Curated NuRec Train/Val Splits

An 80/20 train/validation split curated from the public NuRec 26.01 and 26.04 releases
to locally train and evaluate models before leaderboard submissions. 

| suite | scenes | |
|---|---|---|
| `nurec_curated_train` | 1761 | 80% |
| `nurec_curated_val` | 441 | 20% |

## Usage

These are ordinary test suites, so nothing about running a driver changes.
Follow your driver's own workflow and add the "nurec_scenes" flag:

```bash
uv run alpasim_wizard +e2e_challenge=dev +nurec_scenes=curated_train ...
uv run alpasim_wizard +e2e_challenge=dev +nurec_scenes=curated_val ...
```

## What was excluded

2202 of 2523 publicly available scenes are included.

The 26.04 catalog carries 1448 of the release's 1607 clips. The other 159 are
re-renders of clips the upstream 26.01 catalog already holds under the same
`scene_id`. The curated splits use the 26.01 render for the 153/159 they include.

Most of the other missing scenes are excluded due to issues in the underlying 
HD-Maps used for scoring.

## Location leakage

Scenes recorded in the same place can drive the same road, and a model evaluated
on the same road can score optimistically due to overfitting during training.
This is mainly an issue due to the reduced size of the dataset.

Road-sharing scenes are grouped and assigned to the same split to avoid leakage.
Same location but different road/driving direction is allowed between train and eval.
