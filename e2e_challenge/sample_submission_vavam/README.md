# VAVAM Sample Submission

This is a compact VAVAM-backed `egodriver.EgodriverService` example for the
AlpaSim e2e challenge. It is a richer starting point than the minimal
[`starter_kit`](../starter_kit/README.md).

The driver uses `camera_front_wide_120fov`, rectifies f-theta images to the
NuScenes-style pinhole view expected by VAVAM, runs model inference at 2 Hz, and
serves 10 Hz `Drive` requests from the latest cached plan.

Runtime writes are limited to stdout/stderr and cache directories under `/tmp`
or `/run`, matching the challenge container restrictions.

> Note: run commands from the repo root.

## Assets

Model weights are local build inputs and should not be committed. Download or
copy them locally, then stage them for the Docker build:

```bash
bash e2e_challenge/sample_submission_vavam/scripts/prepare_assets.sh /path/to/vavam_weights
```

The source directory must contain:

```text
VAM_width_1024_pretrained_139k.pt
VQ_ds16_16384_llamagen_encoder.jit
```

The script copies them into:

```text
e2e_challenge/sample_submission_vavam/assets/vavam/
```

You can also set `VAVAM_ASSET_SRC=/path/to/vavam_weights`.

## Build

```bash
bash e2e_challenge/sample_submission_vavam/scripts/build_image.sh
```

## Local Smoke Test

Start the driver container:

```bash
e2e_challenge/sample_submission_vavam/run_local_container.sh
```

Then run the PAI smoke test from another terminal:

```bash
source setup_local_env.sh
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge=dev \
  wizard.log_dir=./runs/e2e_challenge_vavam_smoke
```

For NuPlan, follow the data setup in the [starter kit](../starter_kit/README.md)
and run the `+e2e_challenge_nuplan=dev` smoke test with the VAVAM container
still running.
