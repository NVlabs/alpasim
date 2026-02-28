# Running the Simulation for Local Development

This guide covers approaches for running and debugging simulations locally, each suited for different workflows.

## Prerequisites

Before running simulations, ensure your environment is set up:

```bash
# Setup environment (if not already done)
source setup_local_env.sh

# Compile protobufs (required after any .proto changes)
cd src/grpc && uv run compile-protos
```

> **Note**: Always recompile protobufs after pulling changes that modify `.proto` files in `src/grpc/v0/`.

---

## Approach 1: Full Docker Compose

Run all services in Docker containers on a local workstation with GPU:

```bash
# From src/wizard directory
uv run alpasim_wizard +deploy=local_oss wizard.log_dir=./my_run

# Or with +deploy=local for internal models
uv run alpasim_wizard +deploy=local wizard.log_dir=./my_run
```

**Deploy configs** (in `src/wizard/configs/deploy/`):
- `local_oss.yaml` - Single GPU, OSS models
- `local.yaml` - Single GPU, internal models
- `local_oss_2gpus.yaml` - Two GPUs, higher parallelism

---

## Approach 2: Hybrid - Runtime + Service "Bare Metal"

For developing/debugging a specific service (e.g., trafficsim), run runtime and that service on the host with other services in Docker:

```bash
# Step 1: Generate configs without starting containers
uv run alpasim_wizard +deploy=local \
  wizard.log_dir=./my_run \
  wizard.debug_flags.use_localhost=True \
  wizard.dry_run=True \
  runtime.endpoints.do_shutdown=False \
  runtime.save_dir=/path/to/asl/output

# Step 2: Start the other services in Docker
cd ./my_run
docker compose -f generated-docker-compose.yaml up physics-0 sensorsim-0 driver-0

# Step 3: Start runtime and your service on the host (e.g., from VSCode with debugger)
```

**Key wizard flags for hybrid debugging:**

| Flag | Purpose |
|------|---------|
| `wizard.debug_flags.use_localhost=True` | Use `localhost:port` addresses in generated configs |
| `wizard.dry_run=True` | Generate configs without starting containers |
| `runtime.endpoints.do_shutdown=False` | Don't shutdown containers after simulation |
| `runtime.save_dir=<path>` | Override save directory for host execution |

---

## Approach 3: VSCode Debugging

Multi-step workflow using `.vscode/launch.json` configurations:

1. **Generate configs only** (Run wizard with `wizard.run_method=NONE`):
   - Use "Run wizard" launch config to generate configs in `.wizard/`

2. **Start containers** from generated Docker Compose:

   ```bash
   cd .wizard && docker compose -f generated-docker-compose.yaml up
   ```

3. **Attach debugger to runtime/services**:
   - Use "Run alpasim_runtime.simulate" config to debug the runtime
   - Use "Run vam driver" config to debug the driver locally

**Example launch.json for wizard (generates configs only):**

```json
{
  "name": "Run wizard",
  "type": "debugpy",
  "module": "alpasim_wizard",
  "args": [
    "wizard.log_dir=${workspaceFolder}/.wizard",
    "wizard.run_method=NONE",
    "+deploy=local_oss"
  ],
  "cwd": "${workspaceFolder}/src/wizard"
}
```

**Example launch.json for runtime debugging:**

```json
{
  "name": "Run alpasim_runtime.simulate",
  "type": "debugpy",
  "module": "alpasim_runtime.simulate",
  "args": [
    "--user-config=${workspaceFolder}/.wizard/generated-user-config-0.yaml",
    "--network-config=${workspaceFolder}/.wizard/generated-network-config.yaml",
    "--usdz-glob=${workspaceFolder}/data/nre-artifacts/all-usdzs/*.usdz",
    "--log-dir=${workspaceFolder}/.wizard"
  ],
  "cwd": "${workspaceFolder}/src/runtime"
}
```

---

## Debugging Tips

**Single-Worker Mode** - Set `nr_workers=1` for easier debugging (inline mode, single process):

```bash
# Via command line
uv run alpasim_wizard +deploy=local runtime.nr_workers=1 ...
```

**Skip Services** - Skip specific services to isolate issues (returns synthetic data):

```bash
runtime.endpoints.physics.skip=True
runtime.endpoints.trafficsim.skip=True
```

**Log Replay Mode** - Replay from logs without traffic/physics simulation:

```bash
runtime.endpoints.physics.skip=True \
runtime.endpoints.trafficsim.skip=True \
runtime.default_scenario_parameters.physics_update_mode=NONE \
runtime.default_scenario_parameters.force_gt_duration_us=20000000
```

**Inspecting ASL Logs** - Use `print_asl` utility to inspect simulation logs:

```bash
uv run python -m alpasim_grpc.utils.print_asl rollout.asl --message-types rollout_metadata
uv run python -m alpasim_grpc.utils.print_asl rollout.asl --message-types actor_poses
```

For detailed scene-specific debugging workflows, see [SCENE_DEBUGGING.md](SCENE_DEBUGGING.md).
