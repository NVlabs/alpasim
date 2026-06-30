# Changelog
This document lists major updates which change UX and require adaptation.
It should be sorted by date (more recent on top) and link to MRs which introduce the changes.

## Runtime telemetry with Prometheus and Grafana (30.06.26)
AlpaSim runs now start Prometheus telemetry by default. The wizard allocates
metrics ports, starts Prometheus support services, writes scrape configuration,
and preserves the Prometheus TSDB under the run directory. Runs still generate
`metrics_plot.png`.

Grafana dashboard resources and a helper script are available for inspecting
local or shared Prometheus file-SD targets:

```bash
src/tools/scripts/start-prometheus-grafana.sh <file-sd-dir>
```

Eval aggregation no longer reads Prometheus runtime metric summaries or adds
runtime performance fields to driving metric outputs. Runs no longer generate
`prometheus/runtime_metrics_summary.json`; query Prometheus or use Grafana for
runtime performance analysis.

## SMART/CATK trafficsim integration (29.06.26)
Added an integrated CATK-backed traffic simulation service. Use
`trafficsim=catk` to run the SMART/CATK traffic predictor from the AlpaSim base
image with model weights stored under `data/trafficsim-models`.

## Upgrade OSS renderer to NRE-GA 26.04 (11.06.26)
Bumped the `base_config.yaml` renderer image to `nvcr.io/nvidia/nre/nre-ga:26.04` and repointed `defines.renderer_entrypoint` to the `/app/run` symlink (the old path was renamed in 26.04).

## Eval Modifications (05.06.26)
Eval aggregation now writes `results-summary.json` with per-rollout pass/fail status,
scene scores, failed rollout rows, and driver `drive` RPC latency from telemetry.
Ground-truth scoring adds `progress_rel_to_total` and `gt_dist_traveled_m`; tune or
disable scoring with `eval.scene_score`, and set `eval.allow_aggregation_with_failed_rollouts=true`
to aggregate successful rollouts while preserving failed rollout records.


## May/June 2026 public sync updates (02.06.26)

This sync focuses on renderer unification, video-model support, scene catalog
updates, and runtime robustness improvements.

### Renderer and video-model workflow

Wizard-managed renderer services now use the `renderer` service key for both
the default NRE-backed renderer and video-model renderers. The runtime selects
the active renderer with `runtime.renderer.kind`, and generated configs now use
`runtime.endpoints.renderer` / `network.renderer` instead of `sensorsim`.

Video-model renderer support is now built into the main runtime and wizard
configuration. New public configs include `deploy=external_video_model`,
`deploy=managed_flashdreams`, `+chunking=<8frame|12frame|16frame>`, and
`docs/VIDEO_MODEL.md`. Docker Compose deployments now stop backing services when
`runtime-0` exits.

**Migration**: Replace `services.sensorsim` and
`wizard.run_sim_services=["sensorsim", ...]` with `services.renderer` and
`wizard.run_sim_services=["renderer", ...]`. Replace
`physics/implemented_in_sensorsim.yaml` with
`physics/implemented_in_renderer.yaml`. Replace custom
`runtime.endpoints.sensorsim` / `network.sensorsim` entries with `renderer`.
For one-shot Compose runs, use `docker compose up --exit-code-from runtime-0`.

### Scene catalog and scene loading

The default public Hugging Face scene catalog now uses the 26.01 release and the
`public_2601` suite. The previous 25.05 catalog remains available through
`sim_scenes_2505.csv` / `sim_suites_2505.csv`.

Runtime scene discovery now uses `scene_provider` configuration instead of
passing artifact globs through the runtime CLI.

**Migration**: Existing 25.05 cached artifacts are not reused by the new default.
Pin `scenes.scenes_csv` / `scenes.suites_csv` to the legacy CSVs if you need the
old suite. For direct `python -m alpasim_runtime.simulate` usage, remove
`--usdz-glob` and configure `scene_provider.kind` plus
`scene_provider.usdz.data_dir`.

### Runtime, daemon, and evaluation robustness

The runtime daemon now exposes `RuntimeService.get_runtime_info` for client-side
capacity and scene discovery. Startup logs now report connection progress while
waiting for services.

Closed-loop and RL workflows also gained `DriveResponse.terminate_session`,
single-rollout `RolloutSpec.session_uuid`, explicit driver precondition errors,
spawned evaluation workers, empty driver-response handling, and continuous
min-distance eval scorers.

**Migration**: Existing clients remain compatible. Daemon clients can call
`get_runtime_info`, set `RolloutSpec.session_uuid` for precise single-rollout
abort handling, or set `DriveResponse.terminate_session=true` for early episode
termination. Eval configs can opt into `min_distance_to_obstacle_m` and
`min_distance_to_lane_boundary_m`.

### Simulation timing and logging

Rollout timing now starts from the first GT camera frame timestamp. Force-GT
startup blends recorded GT trajectories into physics-derived trajectories, and
`runtime.simulation_config.skip_driver_during_force_gt` can skip expensive driver
policy queries during force-GT warmup.

`alpasim_utils.asl_to_frames` now exports video-model RGB / HD-map streams and
names output frames by end-of-frame timestamp. `alpasim_utils.print_asl` redacts
large video-model image payloads.

**Migration**: Remove custom `time_start_offset_us` and camera
`first_frame_offset_us` settings. Update downstream scripts that parse frame
filenames to expect end-of-frame timestamps.

### Developer workflow

The `src/grpc` package now builds generated protobuf artifacts during package
builds, so downstream Git installs no longer need a manual `compile-protos`
step. `trajdata-alpasim` is pinned to a specific Git commit for consistent
workspace resolution.

**Migration**: Local proto development can still use `uv run compile-protos`.
Re-run dependency sync after pulling this change.

## Wizard runtime server mode and run mode rename (28.04.26)
Wizard run modes now distinguish one-shot runtime execution from long-running runtime daemon deployment. `wizard.run_mode=BATCH` has been renamed to `wizard.run_mode=ONESHOT`, and `wizard.run_mode=SERVER` starts the runtime as a gRPC server for request-scoped simulations.

**Migration**: If you explicitly set `wizard.run_mode=BATCH`, replace it with `wizard.run_mode=ONESHOT`. The default behavior remains unchanged for standard one-shot execution workflows.

### External driver configuration changes

External driver ownership is now expressed through `driver_source` config groups rather than deployment targets:
* Default managed driver behavior remains unchanged.
* `driver_source=external_static` uses configured external driver addresses, replacing the removed `deploy=local_external_driver` deployment target.
* `driver_source=external_dynamic` supports per-request drivers via `SimulationRequest.available_drivers`.

**Migration**: Replace `deploy=local_external_driver` with `deploy=local driver_source=external_static driver=manual`.

## Move evaluation to a separate thread (03.04.26)
Runtime evaluation now runs in its own thread instead of inline in the simulation loop. This decouples eval latency from the simulation step, improving throughput when evaluation is expensive.

## Dependency fix: override `torchmetrics` pin (03.04.26)
Added `torchmetrics>=1.8.2` to `override-dependencies` in the root `pyproject.toml` to resolve a conflict between upstream driver dependencies.

## Duplicate config detection across providers (01.04.26)
The Hydra config discovery plugin now detects YAML files that exist at the same relative path in multiple config providers (e.g. both `wizard` and an installed plugin). Duplicate paths raise a `ValueError` at startup, preventing silent config shadowing.

## Rename driver configs: ar1 → alpamayo1, a15 → alpamayo1_5 (31.03.26)

Driver config names, entry points, and `model_type` values now use explicit names instead of abbreviations:

| Before | After |
|--------|-------|
| `driver=ar1` | `driver=alpamayo1` |
| `driver=a15` | `driver=alpamayo1_5` |

**Migration**: Replace `driver=ar1` with `driver=alpamayo1` and `driver=a15` with `driver=alpamayo1_5` in CLI invocations, SLURM scripts, and any custom configs that reference these drivers.

## Upgrade OSS sensorsim to NRE-GA 26.02 and unify entrypoint (30.03.26)
The OSS sensorsim image has been upgraded from `docker.io/carlasimulator/nvidia-nurec-grpc:0.2.0` to `nvcr.io/nvidia/nre/nre-ga:26.02`.

* The sensorsim entrypoint (`/app/run serve-grpc`) and all shared flags are now defined once in `base_config.yaml`.
* New flag `--enable-editing-actors` added to the base sensorsim command, required by NRE 26.3 for render requests that include dynamic object updates.

**Migration**: If you override `services.sensorsim.command` in a custom manifest, add `--enable-editing-actors` to the argument list.

## Config refactoring: three-axis composition, per-service images, unified exp/ group (30.03.26)

### Three-axis config model

The wizard config is now composed from three required, independent axes instead of monolithic deploy configs:

```bash
uv run alpasim_wizard deploy=local topology=1gpu driver=vavam wizard.log_dir=./out
```

| Group | Purpose | Examples |
|-------|---------|----------|
| `deploy=` | Where to run (filesystem, run method) | `local`, `local_external_driver` |
| `topology=` | GPU layout, replicas, concurrency | `1gpu`, `2gpu`, `8gpu_64rollouts` |
| `driver=` | Which driving model | `vavam`, `ar1`, `a15`, `manual` |

All three are required. Omitting any prints a helpful error listing available options.

### Driver configs simplified

Each driver config now includes its own runtime settings via the Hydra defaults list. Specify a single config instead of a list:

| Before | After |
|--------|-------|
| `driver=[vavam,vavam_runtime_configs]` | `driver=vavam` |
| `driver=[ar1,alpamayo_runtime_configs]` | `driver=ar1` |
| `driver=[a15,alpamayo_runtime_configs]` | `driver=a15` |

### stable_manifest removed, images derived from pyproject.toml

The `stable_manifest` config group (`oss.yaml`, `oss_gitlab.yaml`) has been removed. Its content has been merged into `base_config.yaml`:

* Services built from the repo (driver, physics, controller, trafficsim, runtime) use `${defines.base_image}`, which reads the version from `pyproject.toml` at runtime via a `repo-version:` OmegaConf resolver.
* The external sensorsim image (`nvcr.io/nvidia/nre/nre-ga:26.02`) is set directly in `base_config.yaml`.
* A default OSS scene ID is now in `base_config.yaml`, so new users can run without specifying scenes.

### Runtime endpoint config moved to topology

`runtime.nr_workers` and all `runtime.endpoints.*.n_concurrent_rollouts` values are now set by topology configs instead of `base_config.yaml`. Each topology preset defines capacity to match its GPU layout. `base_config.yaml` retains only behavioral settings (`do_shutdown`, `enable_autoresume`, etc.).

### Unified exp/ config group

The scattered `model/`, `experiment/`, `sim/`, and `exp/` config directories have been consolidated under a single `exp/` group. Presets (e.g., `vavam_4hz`) moved to `exp/presets/`.

### New optional config groups

New optional groups in `base_config.yaml` defaults allow overriding service-specific settings:
* `controller=` — override controller config
* `sensorsim=` — override NRE image
* `trafficsim=` — override trafficsim config

### SLURM submit.sh changes

* `submit.sh` no longer defaults to any deploy target. All three axes (`deploy=`, `topology=`, `driver=`) must be specified.
* Early sanity check rejects submissions with missing required config groups before allocating SLURM resources.
* Example: `sbatch submit.sh deploy=ord topology=8gpu_64rollouts driver=vavam`

### Breaking changes summary

* `+deploy=` syntax is now `deploy=` (no `+` prefix). Same for `topology` and `driver`.
* `driver=[<model>,<runtime_configs>]` list syntax is now just `driver=<model>`.
* `cameras/wide_only_cam.yaml` removed (use `cameras/1cam.yaml`).
* `stable_manifest` config group removed entirely.
* Deleted monolithic deploy configs: `iad_oss`, `ord_oss`, `ord_oss_single`, `local_2gpus`, `iad` (OSS). Use `deploy=<target> topology=<layout>` instead.
* `runtime.nr_workers` and `runtime.endpoints.*` defaults removed from `base_config.yaml` (set by topology).
* `defines.nre_cache_size` removed from `base_config.yaml` (set by topology).

## Alpamayo 1.5 driver support (24.03.26)
[Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5) is now available as a driver (`model_type: a15`). Use `driver=a15` to run with the 10B model.

* New `A15Model` driver with camera-index-aware inference and optional classifier-free guidance navigation (`use_classifier_free_guidance_nav: true`, ~60 GB VRAM).
* AR1 and A1.5 now share a common `AlpamayoBaseModel` base class, reducing code duplication.
* `planner_delay_us` now defaults to `0` everywhere; the legacy `alpamayo_runtime_configs` file (which set 200ms delay) has been removed.

## Make ~/.netrc optional for public users (17.03.26)
References to `~/.netrc` in the Dockerfile and wizard's Docker Compose generation were unconditional, requiring all users to have the file. The Dockerfile now conditionally sets `NETRC` only when the secret is provided, and the wizard only includes the `netrc` secret in the compose config when `~/.netrc` exists on the host.

## Composable dependency management (12.03.26)
The root `pyproject.toml` now exposes every workspace member as a named optional-dependency extra, enabling composable installs from the repo root. A bare `uv sync` installs nothing (avoiding heavy deps like torch by default).

* `uv sync --extra wizard` — wizard and its transitive deps only
* `uv sync --extra all` — all core packages
* `source setup_local_env.sh` still works and installs all core packages (plugins must be added separately).

See [Onboarding — Dependency management](docs/ONBOARDING.md#dependency-management) for details.

## Overridable Hydra config groups (12.03.26)
Wizard config groups (e.g. `driver`, `deploy`) can now be extended by any installed package. Packages register an `alpasim.configs` entry point pointing to a Python package that contains YAML files, and the wizard automatically adds it to Hydra's search path at startup via `SearchPathPlugin`.

* `model_type` in driver config is now a plain string (e.g. `"ar1"`, `"manual"`) instead of an enum.
* The transfuser driver configs have been moved out of the wizard into the transfuser plugin — when installed, `driver=[transfuser,transfuser_runtime_configs]` resolves automatically.

## Plugin system (12.03.26)
Alpasim is now extensible via Python [entry points](https://packaging.python.org/en/latest/specifications/entry-points/). Any installed package can register models, controllers, configs, or tools without modifying the core codebase.

* New `alpasim-plugins` package (`src/plugins`) provides a `PluginRegistry` that discovers entry points lazily at runtime.
* Driver models (ar1, transfuser, vam, manual) and controller MPCs (linear, nonlinear) are registered as entry points and resolved by name.
* Run `uv run alpasim-info` to list all installed plugins.

See [Plugin System](docs/PLUGIN_SYSTEM.md) for the full architecture, entry-point groups, and how to create new plugins.

## Runtime event-based simulation loop and config cleanup (10.03.26)
- The runtime simulation loop is now event-based instead of a fixed sequential control-step loop.
- `pose_reporting_interval_us` is the active pose-reporting setting; older `egopose_*` configuration
  naming has been removed from the active runtime path.
- The active egomotion noise model path was removed, so configs and tooling should no longer expect
  `egomotion_noise` behavior in standard runtime execution.

## Runtime daemon mode for on-demand simulation (10.03.26)
- The runtime can now run as a long-lived gRPC daemon that accepts simulation requests on demand.
- The gRPC API changed: `RolloutSpec.random_seed` was replaced by `nr_rollouts`, structured rollout
  results are returned, and a `shut_down` RPC was added for graceful shutdown.
- One-shot CLI execution still works, but now routes through the same daemon engine internally.

## NRE 26.02 update, compatibility matrix removal, and sensorsim worker scaling (10.03.26)
- The manual scene artifact compatibility matrix was removed. Scene selection now treats newer NRE
  versions as backwards-compatible and chooses the newest available artifact per scene.
- Sensorsim/NRE scaling now relies on internal workers (`--max-workers`) rather than multiple
  replicas per container in the common OSS deploy configs.
- If you tune throughput, update your expectations for sensorsim capacity: `replicas_per_container`
  alone no longer tells the full story.

## Add Higher Frequecy Reporting (18.02.26)
Added higher frequency pose/state information for when model updates are more sparse.
Additionally, changed the way that the `HF_HOME` environment variable is handled to be more like the public repo.

## ARM64 support and unified SLURM submit script (17.02.26)
* **ARM64 support**: AlpaSim can now run on aarch64 (DGX Spark, DGX Station, IPP5 GB300).
  Build with `docker build --secret id=netrc,src=$HOME/.netrc -t alpasim-base:arm64 .`
  and deploy with `+deploy=local_arm` (Docker Compose) or `+deploy=ipp5` (SLURM).
* **Unified SLURM script**: `src/tools/run-on-slurm/` is the single entry point; previous per-site directories have been consolidated into `src/tools/run-on-slurm/submit.sh`.

**Migration**: Update SLURM submit commands:
- `cd src/tools/run-on-slurm && sbatch --account=<acct> --partition=<part> submit.sh +deploy=ord_oss`
- `cd src/tools/run-on-ipp5 && sbatch submit.sh` → `cd src/tools/run-on-slurm && sbatch --account=<acct> --partition=<part> --gpus-per-node=4 submit.sh +deploy=ipp5`

## Output directory structure changes (03.02.26)
The wizard output directory structure has been reorganized for clarity:
* `./asl/` directory renamed to `./rollouts/` - contains rollout logs organized by scene and session
* `0.asl` and `0.rclog` files renamed to `rollout.asl` and `rollout.rclog`
* `./metrics/` directory renamed to `./telemetry/` - contains Prometheus telemetry data (not to be confused with evaluation metrics stored in rollouts)
* Videos are now saved next to ASL files: `rollouts/<scene_id>/<rollout_uuid>/<video>.mp4`
* Metrics parquet files are saved next to ASL files: `rollouts/<scene_id>/<rollout_uuid>/metrics.parquet`
* `aggregate/videos/all` now uses symlinks instead of hard copies for space efficiency

**Migration**: If you have scripts that reference the old paths, update them to use the new structure:
- `asl/` → `rollouts/`
- `0.asl` → `rollout.asl`
- `0.rclog` → `rollout.rclog`
- `metrics/` → `telemetry/`
- `eval/videos/` or `videos/` → `rollouts/<scene_id>/<rollout_uuid>/<video>.mp4`

## Evaluation now runs in-runtime by default (03.02.26)
* Evaluation metrics are now computed during simulation (in-runtime) by default, eliminating the need for separate eval containers.
* The previous behavior (running evaluation in separate containers after simulation) can be restored with `+eval=eval_in_separate_job`.
* This change simplifies the default workflow and reduces resource usage for most use cases.
* Videos are now saved next to ASL files in `rollouts/<scene_id>/<rollout_uuid>/` (unified path for both modes).
* TODO: Image-based metrics are not yet supported in this workflow (e.g. is_camera_black)

## Remove Maglev Dependency (27.01.26)
Removed `maglev.av` dependency  from the base image to better align with the public-facing
repository. The dependency was required to produce roadcast logs, and this functionality has been
moved to a separate tool (along with the buildauth script) in `src/tools/asl_to_roadcast`. See the
README there for instructions on how to use it to generate roadcast logs going forward and how to
view the produced roadcast logs in DDB. Additionally, ddb and avmf have been removed since these
depended on having roadcast logs and weren't being used.

## Updates to Controller (26.01.26)
Added a new controller implementation in the OSS controller which is faster than the previous one
and allow the choice at runtime between the two implementations. The new (linear) implementation is
the default, and the nonlinear one can be selected using the `defines.mpc_implementation` wizard
configuration parameter.

## Update to Local USDZ support (12.12.25)
Local directory support was recently dropped in one of the larger refactorings. This has been
restored with a slightly different interface. Now, for users to run Alpasim with local USDZ files,
they can use the `scenes.local_usdz_dir` configuration parameter. For example:
``` bash
# to run all scenes in the local_usdz_dir directory:
alpasim_wizard +deploy=local wizard.log_dir=<output_dir> scenes.local_usdz_dir=<abs or rel path to directory> scenes.test_suite_id=local
# to run a subset  of the scenes:
alpasim_wizard +deploy=local wizard.log_dir=<output_dir> scenes.local_usdz_dir=<abs or rel path to directory> scenes.scene_ids=[<your scene ids>]
```

## Autoresume Support for SLURM array jobs (14.04.25)
* A helper script `src/tools/run-on-slurm/resume_slurm_job.sh` is provided to simplify resuming failed array job tasks.

## Autoresume Support (21.03.25)
* Adds the ability for users to restart failed jobs in a batch by setting `runtime.enable_autoresume=true`.

## Deprecation of old repos (24.03.25)
Three alpasim repositories are deprecated in favor of this one, to unify the development process more.

## Breaking change: wizard using `uv tool` (14.03.25)
Using `uv` allows us to automatically updated wizard dependencies without future
action from the user, while currently users have to re-install the wizard.
To migrate:
1. Install `uv` if not yet done: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   Alternatively, run `uv self update` as older versions have been reported to
   not work.
2. Install wizard: `uv tool install -e src/wizard/`
3. `conda` is no longer used with Alpasim, the `alpasim` env can be deleted.

*For developers only:* For debugging using vscode I did the following:
* `uv sync` in `src/wizard/` creates a venv under `src/wizard/.venv`
* In `launch.json`, use `"module": "alpasim_wizard"`
* Use the command "Python: Select Interpreter" to manually pick the python
  interpreter unter `.venv` (you might need to enter the path as the venv wasn't
  picked up automatically for me).

## Removal of batching from the runtime (13.03.25)
* User facing:
    * `runtime.endpoints.*.n_concurrent_batches` is now called `runtime.endpoints.*.n_concurrent_rollouts`.
    * `runtime.batch_size` no longer exists.
* Developer:
    * The concept of batch size has been removed from the runtime.
        * Instead of `Bound/UnboundBatch` and `Rollout` we have `Bound/UnboundRollout`.
    * gRPC API changes.
        * Fields like `batch_size` can be assumed to always be equal to 1 and `rollout_index` equal to 0. They are deprecated.
        * Fields which are `repeated` to support multiple rollouts are deprecated. New fields (with single rollout per message semantics) are added.
        * Runtime falls back to deprecated fields - no breaking change for now.

## Wizard USDZ management changes (24.02.25)

* Scene selection is performed via `scenes.{scene_ids,test_suite_id}` instead of `wizard.nre_sceneset`.
    * The options are mutually exclusive.
    * Specific artifacts will be automatically selected to match the configured NRE version.
    If impossible, an error is thrown.
* `usdz` files are now cached by their `uuid` rather than path.
* `python -m alpasim_wizard.check_config <hydra args...>` is a new command which can be ran **on login node** to quickly sanity-check if the run configuration is valid in terms of syntax and scene settings.
