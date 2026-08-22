# SimScale LTF AlpaSim E2E Driver

This submission serves the SimScale Latent TransFuser checkpoint
`ltf_sim_navtest.ckpt` through AlpaSim's `EgodriverService`. It supports the
NuPlan/MTGS E2E track with 1920x1080 `CAM_L0`, `CAM_F0`, and `CAM_R0` RGB/JPEG
observations. Other NuPlan cameras are ignored, and the PAI track is not
supported because its camera contract differs from NAVSIM.

## Prepare The Checkpoint

Download the released checkpoint from the official
[`OpenDriveLab/SimScale`](https://huggingface.co/datasets/OpenDriveLab/SimScale)
Hugging Face dataset, then stage it:

```bash
DOWNLOAD_DIR="${TMPDIR:-/tmp}/simscale-checkpoints"
uv run --with huggingface-hub hf download \
  OpenDriveLab/SimScale \
  SimScale_ckpts/LTF/ltf_sim_navtest.ckpt \
  --repo-type dataset --local-dir "$DOWNLOAD_DIR"
bash e2e_challenge/sample_submission_simscale_navsim_transfuser/scripts/prepare_assets.sh \
  "$DOWNLOAD_DIR/SimScale_ckpts/LTF/ltf_sim_navtest.ckpt"
```

The checkpoint must be exactly 224,560,669 bytes with SHA-256
`9c1a17651bb2cd8e2edf006ea45634432c38554a8f44e0714f64d11ea31f2c69`.
`prepare_assets.sh` rejects a different file. The staged copy under
`assets/ltf/` is ignored by Git.

## Build

```bash
bash e2e_challenge/sample_submission_simscale_navsim_transfuser/scripts/build_image.sh
docker image inspect alpasim-e2e-simscale-ltf:latest
```

The image build includes only this submission, the verified checkpoint, and
the local `src/grpc` package. The runtime does not install NAVSIM, NuPlan,
Lightning, or Hydra and does not need host mounts.

## Local Smoke And Probe

Start the driver container:

```bash
ALPASIM_DRIVER_DETACH=1 \
  e2e_challenge/sample_submission_simscale_navsim_transfuser/run_local_container.sh
```

Then probe it from another terminal:

```bash
PYTHONPATH=e2e_challenge/sample_submission_simscale_navsim_transfuser \
  uv run python \
  e2e_challenge/sample_submission_simscale_navsim_transfuser/scripts/probe_container.py \
  --address 127.0.0.1:6789 --timeout 180
```

The launcher uses a read-only root filesystem and bounded writable tmpfs
directories. The probe starts two sessions, submits the official camera
definitions, and calls `Drive` concurrently. Logs should show
`ltf_inference > 0` and `inference_error = 0`; an all-fallback run is not a
successful smoke test.

The main runtime settings are `ALPASIM_DRIVER_NAME_PREFIX`,
`ALPASIM_DRIVER_REPLICAS`,
`ALPASIM_DRIVER_BASE_PORT`, `ALPASIM_DOCKER_GPUS`, `LTF_MAX_BATCH_SIZE`,
`LTF_BATCH_WINDOW_MS`, and `LTF_DEVICE`.

Model dynamics come from the same-timestamp RPC `DynamicState` supplied by the
corrected official runtime. Pose history is used only when that state is
missing, and each fallback increments `dynamic_state_fallback`. This requires
official runtime `4d352ba` or newer.

## NuPlan/MTGS Smoke

With a local NuPlan/MTGS data root prepared and the driver running:

```bash
ALPASIM_DRIVER_HOST=localhost \
ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/alpasim-nuplan-track \
  uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.log_dir=./runs/e2e_challenge_navsim_transfuser_smoke
```

Inspect `aggregate/results-summary.json` and the driver logs. A valid smoke
run must include fresh TransFuser inference without inference errors or
straight-line fallbacks.
