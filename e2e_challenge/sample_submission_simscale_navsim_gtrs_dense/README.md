# SimScale GTRS-Dense AlpaSim Driver

This submission serves the SimScale GTRS-Dense NAVHARD checkpoint through
AlpaSim's `EgodriverService`. The default image uses the ResNet34 backbone,
the reward checkpoint, FP32 scoring, and the verified `navsim_16384.npy`
trajectory vocabulary. Defaults are `GTRS_BACKBONE=resnet`,
`GTRS_METHOD=reward`, and `GTRS_SPEED_ENHANCEMENT=1`.

Set `GTRS_BACKBONE=vov` at build time to use the released V-99-eSE image
backbone. The method remains independently selectable with
`GTRS_METHOD=reward` or `GTRS_METHOD=expert`; a backbone/checkpoint change
requires rebuilding the image.

The speed-enhancement profile scores candidates with `NC * DAC * EP` using
`EP=3`, then applies longitudinal progress to the top `K=64` candidates with
`lambda=3`. Set `GTRS_SPEED_ENHANCEMENT=0` to use `EP=1`, `K=0`, and
`lambda=0`. Advanced overrides are available through
`GTRS_SCORER_MODE`, `GTRS_EP_EXPONENT`, `GTRS_SPEED_TOP_K`, and
`GTRS_SPEED_WEIGHT`. The 8,192 NAVHARD vocabulary remains available through
`GTRS_VOCAB_PATH`.

## Prepare The Checkpoint

For the default ResNet reward image, download and stage:

```bash
DOWNLOAD_DIR="${TMPDIR:-/tmp}/simscale-checkpoints"
uv run --with huggingface-hub hf download \
  OpenDriveLab/SimScale \
  SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_reward_navhard.ckpt \
  --repo-type dataset --local-dir "$DOWNLOAD_DIR"
bash e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/scripts/prepare_assets.sh \
  "$DOWNLOAD_DIR/SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_reward_navhard.ckpt"
```

The ResNet reward checkpoint must be exactly 269,095,388 bytes with SHA-256
`8dad0395332ccd844785cbfc7c9e24cb3f8d8dbf5cb9ca7f8f8dc75478fcf409`.
`prepare_assets.sh` validates the identity and stores the ignored copy under
`assets/gtrs_dense/`.

For a VoV reward image, stage the corresponding official checkpoint with the
selector set for both asset preparation and image build:

```bash
uv run --with huggingface-hub hf download \
  OpenDriveLab/SimScale \
  SimScale_ckpts/GTRS_Dense/gtrs_dense_vov_sim_reward_navhard.ckpt \
  --repo-type dataset --local-dir "$DOWNLOAD_DIR"
GTRS_BACKBONE=vov bash e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/scripts/prepare_assets.sh \
  "$DOWNLOAD_DIR/SimScale_ckpts/GTRS_Dense/gtrs_dense_vov_sim_reward_navhard.ckpt"
```

## Build

```bash
bash e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/scripts/build_image.sh
docker image inspect alpasim-e2e-simscale-gtrs-dense:latest
```

The published default starts from
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, installs pinned dependencies,
and copies the local `src/grpc` package. On an offline host, an explicitly
configured local base image may be used with `BASE_IMAGE` and
`INSTALL_DEPENDENCIES=0`; this is only a local build override.

## Local Smoke And Probe

```bash
ALPASIM_DRIVER_DETACH=1 \
  e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/run_local_container.sh
PYTHONPATH=e2e_challenge/sample_submission_simscale_navsim_gtrs_dense \
  uv run python \
  e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/scripts/probe_container.py \
  --address 127.0.0.1:6789 --timeout 180
```

The probe requires two concurrent sessions and finite 41-point trajectories.
Logs should show `gtrs_inference > 0` and `inference_error = 0`; inspect
`cached_plan`, `straight_fallback`, and `dynamic_state_fallback` to ensure the
run is not being served entirely by fallback trajectories.

The driver reads rig-frame velocity and acceleration from the same-timestamp
RPC `DynamicState`. Pose history is used only when that state is missing, and
each fallback increments `dynamic_state_fallback`. This requires official
runtime `4d352ba` or newer.

## NuPlan/MTGS Smoke

With a local NuPlan/MTGS data root prepared and the driver running:

```bash
ALPASIM_DRIVER_HOST=localhost \
ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/alpasim-nuplan-track \
  uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.log_dir=./runs/e2e_challenge_navsim_gtrs_dense_smoke
```

Inspect the aggregate result and driver logs. A valid smoke run must include
fresh GTRS inference and no inference errors or straight-line fallbacks.
