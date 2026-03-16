# Changelog
This document lists major updates which change UX and require adaptation.
It should be sorted by date (more recent on top) and link to MRs which introduce the changes.

## Composable dependency management (12.03.26)
The root `pyproject.toml` now exposes every workspace member as a named optional-dependency extra, enabling composable installs from the repo root. A bare `uv sync` installs nothing (avoiding heavy deps like torch by default).

* `uv sync --extra wizard` — wizard and its transitive deps only
* `uv sync --extra all` — all core packages
* `uv sync --extra all --extra internal` — core + internal plugin
* `source setup_local_env.sh` still works and installs all core packages (plugins must be added separately).

Internal-only content (CI/CD, configs, tools, docs) has been moved to an optional package at `plugins/internal/`.

The default stable manifest (`src/wizard/configs/stable_manifest/oss.yaml`) now references public / locally-built Docker images. A new `plugins/internal/configs/stable_manifest/oss_gitlab.yaml` provides the internal-CI equivalent with `nvcr.io`-hosted images; the CI auto-bump scripts update versions there.

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
- `cd src/tools/run-on-ipp5 && sbatch submit.sh` → `cd src/tools/run-on-slurm && sbatch --account=<acct> --partition=<part> --gpus=4 submit.sh +deploy=ipp5`

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
