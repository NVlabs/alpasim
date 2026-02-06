# Changelog

This document lists major updates which change UX and require adaptation. It should be sorted by
date (more recent on top) and link to MRs which introduce the changes.

## Output directory structure changes (2.3.26)

## ARM64 support and unified SLURM submit script (17.02.26)
* **ARM64 support**: AlpaSim can now run on aarch64.
  Build with `docker build --secret id=netrc,src=$HOME/.netrc -t alpasim-base:arm64 .`
  and deploy with `+deploy=local_oss_arm` (Docker Compose) or `+deploy=ipp5` (SLURM).
* **Unified SLURM script**: To run on slurm use `src/tools/run-on-slurm/submit.sh`.
  All arguments are forwarded to the wizard as Hydra overrides.

**Example**: Example SLURM submit command usage:
- `cd src/tools/run-on-slurm && sbatch --account=<acct> --partition=<part> --gpus=4 submit.sh +deploy=ipp5`

## Output directory structure changes (03.02.26)
The wizard output directory structure has been reorganized for clarity:

- `./asl/` directory renamed to `./rollouts/` - contains rollout logs organized by scene and session
- `0.asl` and `0.rclog` files renamed to `rollout.asl` and `rollout.rclog`
- `./metrics/` directory renamed to `./telemetry/` - contains Prometheus telemetry data (not to be
  confused with evaluation metrics stored in rollouts)
- Videos are now saved next to ASL files: `rollouts/<scene_id>/<rollout_uuid>/<video>.mp4`
- Metrics parquet files are saved next to ASL files:
  `rollouts/<scene_id>/<rollout_uuid>/metrics.parquet`
- `aggregate/videos/all` now uses symlinks instead of hard copies for space efficiency

**Migration**: If you have scripts that reference the old paths, update them to use the new
structure:

- `asl/` → `rollouts/`
- `0.asl` → `rollout.asl`
- `0.rclog` → `rollout.rclog`
- `metrics/` → `telemetry/`
- `eval/videos/` or `videos/` → `rollouts/<scene_id>/<rollout_uuid>/<video>.mp4`

## Evaluation now runs in-runtime by default (2.3.26)

- Evaluation metrics are now computed during simulation (in-runtime) by default, eliminating the
  need for separate eval containers.
- The previous behavior (running evaluation in separate containers after simulation) can be restored
  with `+eval=eval_in_separate_job`.
- This change simplifies the default workflow and reduces resource usage for most use cases.
- Videos are now saved next to ASL files in `rollouts/<scene_id>/<rollout_uuid>/` (unified path for
  both modes).
- TODO: Image-based metrics are not yet supported in this workflow (e.g. is_camera_black)

## Updates to Controller (1.26.26)

Added a new controller implementation in the OSS controller which is faster than the previous one
and allow the choice at runtime between the two implementations. The new (linear) implementation is
the default, and the nonlinear one can be selected using the `defines.mpc_implementation` wizard
configuration parameter.

## AR1 (1.5.25)

AR1 support has been added to Alpasim. Users can now run AR1 as described in the tutorial.

## Update to OSS tutorial definitions / Driver Abstraction (12.17.25)

The tutorials have been updated to reflect the changed scene storage model. Manual downloadds are no
longer required, although users must provide the `HF_TOKEN` env var to support automatic downloads.

Additionally, and abstraction has been made in the driver code in the hopes that this will make it
easier to add new driver types in the future. As a result, the location of driver code has moved to
`data/drivers`.

## Update to Local USDZ support (12.12.25)

Local directory support was recently dropped in one of the larger refactorings. This has been
restored with a slightly different interface. Now, for users to run Alpasim with local USDZ files,
they can use the `scenes.local_usdz_dir` configuration parameter. For example:

```bash
# to run all scenes in the local_usdz_dir directory:
alpasim_wizard +deploy=local wizard.log_dir=<output_dir> scenes.local_usdz_dir=<abs or rel path to directory> scenes.test_suite_id=local
# to run a subset  of the scenes:
alpasim_wizard +deploy=local wizard.log_dir=<output_dir> scenes.local_usdz_dir=<abs or rel path to directory> scenes.scene_ids=[<your scene ids>]
```
