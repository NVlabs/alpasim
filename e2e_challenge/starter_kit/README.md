# Starter Kit

The starter kit provides a minimal Python example of `egodriver.EgodriverService`.

> Note: run all commands from the repo root.

## Build the Image:

```bash
docker build -f e2e_challenge/starter_kit/Dockerfile \
  -t alpasim-e2e-starter-driver:latest .
```

## Run a Local Smoke Test

Start one hardened local driver container:

```bash
e2e_challenge/starter_kit/run_local_container.sh
```

Then, in another terminal, bring up the evaluation framework with the wizard:

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

## Optional 8-GPU Multi-Replica Smoke

This mirrors the current official evaluation shape and requires an 8-GPU host.
Start 12 local driver containers:

```bash
ALPASIM_DRIVER_REPLICAS=12 \
  e2e_challenge/starter_kit/run_local_container.sh
```

```bash
uv run alpasim_wizard +e2e_challenge=dev \
  topology=8gpu_36rollouts \
  'wizard.external_services.driver=["localhost:6789","localhost:6790","localhost:6791","localhost:6792","localhost:6793","localhost:6794","localhost:6795","localhost:6796","localhost:6797","localhost:6798","localhost:6799","localhost:6800"]' \
  runtime.endpoints.driver.n_concurrent_rollouts=3 \
  wizard.log_dir=./runs/e2e_challenge_multi_smoke
```

## Notes

The local smoke test uses the official container restrictions except outbound network
blocking. See the [challenge README](../README.md) for the submission contract
and the [Challenge CLI README](../competitor_cli/README.md) for upload and
submission commands.
