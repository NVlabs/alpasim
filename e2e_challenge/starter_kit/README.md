# Starter Kit

The starter kit provides a minimal Python example of `egodriver.EgodriverService`.
It can be smoke-tested against either of the two challenge tracks.

> Note: run all commands from the repo root.

## Common: Starter Kit Driver

Build the starter driver image:

```bash
docker build -f e2e_challenge/starter_kit/Dockerfile \
  -t alpasim-e2e-starter-driver:latest .
```

Start one hardened local driver container:

```bash
e2e_challenge/starter_kit/run_local_container.sh
```

Leave that container running while you start one of the smoke tests below in
another terminal.

## PAI Smoke Test

This smoke test allows for smoke testing of the PAI track.

```bash
source setup_local_env.sh  # if you haven't already
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge=dev \
  wizard.log_dir=./runs/e2e_challenge_smoke
```

Result:

```text
./runs/e2e_challenge_smoke/aggregate/results-summary.json
```

### Optional 8-GPU Multi-Replica Smoke

This mirrors the current official evaluation shape and requires an 8-GPU host.
Start 16 local driver containers:

```bash
ALPASIM_DRIVER_REPLICAS=16 \
  e2e_challenge/starter_kit/run_local_container.sh
```

Then run the simulator stack:

```bash
uv run alpasim_wizard +e2e_challenge=dev \
  topology=8gpu_32rollouts \
  runtime.endpoints.driver.n_concurrent_rollouts=2 \
  'wizard.external_services.driver=["localhost:6789","localhost:6790","localhost:6791","localhost:6792","localhost:6793","localhost:6794","localhost:6795","localhost:6796","localhost:6797","localhost:6798","localhost:6799","localhost:6800","localhost:6801","localhost:6802","localhost:6803","localhost:6804"]' \
  wizard.log_dir=./runs/e2e_challenge_multi_smoke
```

## NuPlan Smoke Test

This smoke test uses the NuPlan/MTGS preset. It needs a local NuPlan/MTGS data
root in addition to the starter driver container.

### Data Setup

Create a local data root and download the prebuilt navtest cache, scene configs,
and the first MTGS asset shard from the
[OpenDriveLab challenge dataset](https://huggingface.co/datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track).
The first asset shard is enough for the `dev` smoke test.

```bash
export ALPASIM_NUPLAN_ROOT=/path/to/alpasim-nuplan-track
export ALPASIM_NUPLAN_HF=/path/to/alpasim-nuplan-track-hf

mkdir -p "$ALPASIM_NUPLAN_ROOT" "$ALPASIM_NUPLAN_HF"

uv run --with huggingface-hub hf download \
  --repo-type dataset \
  --local-dir "$ALPASIM_NUPLAN_HF" \
  OpenDriveLab/AlpasimChallenge2026_nuplan_track \
  trajdata_cache/nuplan_test.tar.gz \
  MTGS_asset/navtest/configs.tar.gz \
  MTGS_asset/navtest/assets/part001.tar.gz

tar -xzf "$ALPASIM_NUPLAN_HF/trajdata_cache/nuplan_test.tar.gz" \
  -C "$ALPASIM_NUPLAN_ROOT"
tar -xzf "$ALPASIM_NUPLAN_HF/MTGS_asset/navtest/configs.tar.gz" \
  -C "$ALPASIM_NUPLAN_ROOT"
tar -xzf "$ALPASIM_NUPLAN_HF/MTGS_asset/navtest/assets/part001.tar.gz" \
  -C "$ALPASIM_NUPLAN_ROOT"
```

The resulting layout should include:

```text
/path/to/alpasim-nuplan-track/
  navtest/
    configs/
    assets/
  nuplan_test/
```

### Run the Smoke Test

With the starter driver container still running:

```bash
source setup_local_env.sh  # if you haven't already
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/alpasim-nuplan-track \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.log_dir=./runs/e2e_challenge_nuplan_smoke
```

Result:

```text
./runs/e2e_challenge_nuplan_smoke/aggregate/results-summary.json
```

The dev preset runs one scene. To smoke-test the full NuPlan scene set,
download and extract all `MTGS_asset/navtest/assets/part*.tar.gz` shards, then
override the scene group and remove the dev scene limit:

```bash
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/alpasim-nuplan-track \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  nuplan_scenes=navtest_full \
  scenes.limit_to_first_n=0 \
  wizard.log_dir=./runs/e2e_challenge_nuplan_full_smoke
```

## Notes

The local smoke tests use the official container restrictions except outbound
network blocking. See the [challenge README](../README.md) for the submission
contract and the [Challenge CLI README](../competitor_cli/README.md) for upload
and submission commands.
