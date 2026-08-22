# SimScale DiffusionDrive NAVHARD AlpaSim E2E Driver

This standalone submission serves the official SimScale DiffusionDrive NAVHARD
checkpoint through AlpaSim's `EgodriverService`. It consumes `CAM_L0`, `CAM_F0`,
and `CAM_R0` at 1920x1080 and returns an eight-pose model prediction as a
41-point, 10 Hz controller trajectory.

The image contains the runtime code, checkpoint, and gRPC package it needs. It
does not import NAVSIM, NuPlan, `reference/SimScale`, or another submission at
runtime, and it does not use host mounts for source, weights, or datasets.

## Prepare The Checkpoint

Download the released checkpoint from the official
[`OpenDriveLab/SimScale`](https://huggingface.co/datasets/OpenDriveLab/SimScale)
Hugging Face dataset, then stage the verified file:

```bash
DOWNLOAD_DIR="${TMPDIR:-/tmp}/simscale-checkpoints"
uv run --with huggingface-hub hf download \
  OpenDriveLab/SimScale \
  SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navhard.ckpt \
  --repo-type dataset --local-dir "$DOWNLOAD_DIR"
bash e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/scripts/prepare_assets.sh \
  "$DOWNLOAD_DIR/SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navhard.ckpt"
```

The checkpoint must be exactly 243,596,717 bytes with SHA-256
`8fdbdb3fdfa7b496e7d7a438efbb5c2022377e59cbfd7095270d89623c5d963f`.
`prepare_assets.sh` rejects a different file. The staged copy under
`assets/diffusiondrive/` is ignored by Git.

## Build

Build the image from the repository root:

```bash
bash e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/scripts/build_image.sh
docker image inspect alpasim-e2e-simscale-diffusiondrive:latest
```

The default build uses the pinned
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` base and includes the local
`src/grpc` package and pinned `requirements.txt` dependencies. It is
independent of the other SimScale submissions.

## Local Smoke And Probe

Start the hardened driver container:

```bash
ALPASIM_DRIVER_DETACH=1 \
  e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/run_local_container.sh
```

Then probe it from another terminal:

```bash
PYTHONPATH=e2e_challenge/sample_submission_simscale_navsim_diffusiondrive \
  uv run python \
  e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/scripts/probe_container.py \
  --address 127.0.0.1:6789 --timeout 180
```

The probe starts two sessions and requires finite, sorted 41-point trajectories
from model inference. Logs should contain `diffusiondrive_inference > 0` and
`inference_error = 0`; an all-fallback run is not a successful smoke test.

Model dynamics come from the same-timestamp RPC `DynamicState` supplied by the
corrected official runtime. Pose history is used only when that state is
missing, and each fallback increments `dynamic_state_fallback`. This requires
official runtime `4d352ba` or newer.

## NuPlan/MTGS Smoke

With the driver still running and a local NuPlan/MTGS data root prepared, run:

```bash
ALPASIM_DRIVER_HOST=localhost \
ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/alpasim-nuplan-track \
  uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.log_dir=./runs/e2e_challenge_navsim_diffusiondrive_navhard_smoke
```

Inspect `aggregate/results-summary.json` and the driver logs. The rollout must
complete with model inference and without inference errors or straight-line
fallbacks.
