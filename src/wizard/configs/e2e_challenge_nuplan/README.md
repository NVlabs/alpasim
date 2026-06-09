# E2E Challenge NuPlan Presets

## Dev

Run a one-scene closed-loop smoke test with a managed MTGS renderer and an
already-running contestant driver:

```bash
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/media/mwatson/4TB/worldengine \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.log_dir=./runs/e2e_challenge_nuplan_dev
```

The preset expects MTGS assets under:

```text
$ALPASIM_NUPLAN_ROOT/navtest/assets
```

It uses `runtime.renderer.kind=sensorsim` because MTGS exposes the standard
Sensorsim gRPC API. The `renderer` service command starts
`alpasim-mtgs-server` inside the AlpaSim base image.

The preset uses the standard AlpaSim base image tag. Rebuild it after Dockerfile
changes so the image contains the MTGS server extra:

```bash
docker build --network=host -t alpasim-base:0.89.0 .
```
