# public_2601_stop_sign_v0

`public_2601_stop_sign_v0` is a 26-scene stop-sign subset of the
`public_2601` public scene set. It is intended for training and evaluating
stop-and-go behavior around genuine stop signs.

## Selection

The candidate clips were selected from `public_2601` by screening
`rig_trajectories.json` for a recorded ego stop, then reviewed manually. Only
scenes with a genuine stop sign were retained.

The recorded stop starts at the first sample of the first contiguous interval
where ground-plane ego speed stays below 0.5 m/s for at least 0.5 seconds. Each
CSV row sets:

```text
event_t0_us = recorded_stop_start_us
```

The suite therefore identifies the detected near-stop itself. Consumers that
need context before the stop can apply their own T0 offset.

## Artifacts

The suite pins each selected clip to an exact USDZ artifact UUID. All 26
`(clip_id, uuid)` pairs are rows of `../public_2601.csv`, which provides the
checked-in artifact version for this subset.

## Usage

Select the suite by its filename stem:

```bash
scenes.suite_ids=[public_2601_stop_sign_v0]
```

## Validation

- 26 unique clips and 26 exact artifact UUID pins
- numeric near-stop T0 for every row
- exact `(clip_id, uuid)` subset of `public_2601`
- checked-in scene-suite validation and Wizard scene-suite tests
