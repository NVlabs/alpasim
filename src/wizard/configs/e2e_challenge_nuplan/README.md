# E2E Challenge NuPlan Presets

## Dev

Run a one-scene closed-loop smoke test with a managed MTGS renderer and an
already-running contestant driver:

```bash
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/worldengine-root \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.log_dir=./runs/e2e_challenge_nuplan_dev
```

## Full Navtest Evaluation

Run the full navtest preset with the same trusted data root:

```bash
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/worldengine-root \
uv run alpasim_wizard +e2e_challenge_nuplan=full \
  wizard.log_dir=./runs/e2e_challenge_nuplan_full
```

Set `ALPASIM_NUPLAN_ROOT` to the World Engine data root containing
`navtest/configs`, `navtest/assets`, and `nuplan_test`.

## Preset Structure

`dev.yaml` and `full.yaml` both compose
`../e2e_challenge_nuplan_common/base.yaml` for the shared MTGS/trajdata runtime
setup. They only differ in run naming, scene selection, scene limit, and whether
videos are rendered.

To run the dev preset with the full scene set, override the scene group and
remove the dev scene limit:

```bash
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  nuplan_scenes=navtest_full \
  scenes.limit_to_first_n=0
```

## Full Scene List

The full navtest list is expanded from the navtest config YAML files. There are
1,491 config files, but the evaluation scene id is generated for every central
token:

```text
<central_log>-<central_token>
```

For the current navtest set this produces 12,146 unique scene ids in
`../nuplan_scenes/navtest_full.yaml`, which is composed by `full.yaml`. This
matches the scene naming expected by the trajdata-backed NuPlan scene provider.

## Validation

Before running a full evaluation, verify that Hydra can resolve the preset and
that the scene provider sees the full set:

```bash
uv run --extra wizard alpasim_check_config \
  +e2e_challenge_nuplan=full \
  defines.worldengine_root=/path/to/worldengine-root
```

Expected output:

```text
Found 12146 scenes.
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
