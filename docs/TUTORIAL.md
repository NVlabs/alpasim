# AlpaSim tutorial: introduction

This tutorial makes three assumptions

1. It targets an AlpaSim user rather than an AlpaSim developer
1. It treats docker compose as the primary execution environment.
1. It focuses on letting the user do simple things quick and leaves detail for later. This is
   reflected in subdivision into three levels of complexity.

# Level 1

In level 1 we run a default simulation with the VaVAM driver policy, learn how to interpret the
results, and perform basic debugging.

## Architecture of AlpaSim

AlpaSim consists of multiple networked microservices (renderer, physics simulation, runtime,
controller, driver, traffic simulation). The AlpaSim runtime requests observed video frames from the
renderer and egomotion history from the controller, communicates with the physics microservice to
constrain actors to the road surface, and provides the information to the driver, with the
expectation of receiving driving decisions in return to close the loop.

This repository contains the implementations of a subset of the services needed to execute the
simulation as well as config files and infra code necessary to bring the microservices up via
docker/enroot.

## Running with docker compose

Let's start by executing a run with default settings.

1. Follow [onboarding](ONBOARDING.md) to ensure necessary dependencies have
   been installed
1. Set up your environment with:
   - `source setup_local_env.sh`
   - This will compile protos, download an example driver model, ensure you have a valid Hugging
     Face token, and install the `alpasim_wizard` command line tool.
1. Run the following one-time setup steps required by the default Docker workflow:
   ```bash
   # 1) Ensure HF env is present (needed for scene/model downloads)
   export HF_TOKEN="<your_hf_token>"
   # 2) Download VaVAM assets required by default driver
   bash data/download_vavam_assets.sh --model vavam-b
   ```
   If you need to create a Hugging Face token, see the Hugging Face access section in
   [onboarding](ONBOARDING.md).
1. Run the wizard to create the necessary config files, download the scene (if necessary), and run a
   simulation _ `uv run alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial` _ This will create a
   `tutorial/` directory with all necessary config files and run the simulation

## Results structure

The simulation logs/output will be in the created `tutorial` directory. For a visualization of the
results, an `mp4` file is created in `tutorial/eval/videos/clipgt-026d..._0.mp4`. The full results
should looks something like:

```
tutorial/
├── aggregate
│   ├── metrics_results.png
│   ├── metrics_results.txt
│   ├── metrics_unprocessed.parquet
│   └── videos
│       ├── all
│       │   └── clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3_814f3c22-bb78-11f0-a5f3-2f64b47b8685_0.mp4
│       └── violations
│           ├── collision_at_fault
│           ├── collision_rear
│           ├── dist_to_gt_trajectory
│           │   └── clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3_814f3c22-bb78-11f0-a5f3-2f64b47b8685_0.mp4 -> ../../all/clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3_814f3c22-bb78-11f0-a5f3-2f64b47b8685_0.mp4
│           └── offroad
├── rollouts
│   ├── clipgt-f7020b3e-3d61-4cb6-b157-2f4aac1a7d8d
│   │   └── 86513b18-96c5-11ef-8b6f-b83fd26d88f0
│   │       ├── rollout.asl
│   │       ├── rollout_indexed
│   │       │   ├── camera_front_tele_30fov.mp4
│   │       │   ├── camera_front_wide_120fov.mp4
│   │       │   ├── manifest.json
│   │       │   ├── rclog-all.index
│   │       │   └── rclog-all-indexed.log
│   │       ├── rollout.rclog
│   │       ├── metrics.parquet
│   │       ├── {clipgt_id}_{batch_id}_{rollout_id}.mp4
│   │       └── _complete
│   ├── clipgt-fa369408-2787-41cb-b629-a7885d7c46e2
│   │   └── 8656ecb6-96c5-11ef-8b6f-b83fd26d88f0
│   │       ├── rollout.asl
│   │       ├── rollout_indexed
│   │       │   ├── camera_front_tele_30fov.mp4
│   │       │   ├── camera_front_wide_120fov.mp4
│   │       │   ├── manifest.json
│   │       │   ├── rclog-all.index
│   │       │   └── rclog-all-indexed.log
│   │       ├── rollout.rclog
│   │       ├── metrics.parquet
│   │       ├── {clipgt_id}_{batch_id}_{rollout_id}.mp4
│   │       └── _complete
│   └── clipgt-fe127c3f-8b06-4c4f-9933-1e5089a1a731
│       └── 864da3ea-96c5-11ef-8b6f-b83fd26d88f0
│           ├── rollout.asl
│           ├── rollout_indexed
│           │   ├── camera_front_tele_30fov.mp4
│           │   ├── camera_front_wide_120fov.mp4
│           │   ├── manifest.json
│           │   ├── rclog-all.index
│           │   └── rclog-all-indexed.log
│           ├── rollout.rclog
│           ├── metrics.parquet
│           ├── {clipgt_id}_{batch_id}_{rollout_id}.mp4
│           └── _complete
├── driver
│   └── vam-driver.yaml
├── driver-config.yaml
├── eval
│   ├── metrics_unprocessed.parquet
│   └── videos
│       └── clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3_814f3c22-bb78-11f0-a5f3-2f64b47b8685_0.mp4
├── eval-config.yaml
├── generated-network-config.yaml
├── generated-user-config-0.yaml
├── metrics
├── run_metadata.yaml
├── run.sh
├── trafficsim-config.yaml
├── txt-logs
├── wizard-config-loadable.yaml
└── wizard-config.yaml
```

Some noteworthy files and directories:

- `rollouts` contains logs of simulation behavior in each rollout, used to analyze AV behavior and
  calculate metrics. The logs are organized into
  `rollouts/{scenario.scene_id}/{batch_uuid}/rollout.*` - in this case we have 3 scenes with one
  batch of a single rollout each. The subdivision into batches is historical and can be ignored for
  most purposes.
  - `.asl` files which record the messages exchanged within the simulation. These are useful for
    debugging the simulator behavior and replaying events. More in
    [asl log format](#asl-log-format).
  - `metrics.parquet` contains per-rollout evaluation metrics.
  - `{clipgt_id}_{batch_id}_{rollout_id}.mp4` evaluation videos (when video rendering is enabled),
    where `clipgt_id=f"clipgt-{scene_id}"`, `batch_id="0"`, `rollout_id=rollout_uuid`.
  - `_complete` is a marker file created when a rollout finishes successfully. Used by the
    autoresume feature to track which rollouts completed and to remove incomplete rollout
    directories on restart.

* `aggregate/` contains aggregated results across all rollouts:
  - `metrics_results.txt` - Formatted table of driving scores (mean, std, quantiles)
  - `metrics_results.png` - Visual summary of driving quality metrics
  - `metrics_unprocessed.parquet` - Combined metrics from all rollouts
  - `videos/` - Videos organized by violation type (collision_at_fault, offroad, etc.)
* `metrics/` contains performance profiling data (see
  [OPERATIONS.md](OPERATIONS.md#how-do-i-view-performance-metrics) for details):
  - `metrics.prom` - Prometheus metrics from simulation
  - `metrics_plot.png` - Performance visualization (CPU/GPU/RPC metrics)
* `driver` is a directory with logs written by the driver service, useful to debug policy-internal
  problems.
* `wizard-config.yaml` contains the config the wizard used for this run **after applying the
  inheritance of hydra**. This is useful for debugging configuration issues.
* `generated-user-config-{ARRAY_ID}.yaml` contains an expanded version of the simulation config
  provided by the user, possibly split into chunks when simulating on multiple nodes.
* `trafficsim-config.yaml`. A copy of the traffic simulation config used for simulation, useful for
  debugging traffic simulation.
* `generated-network-config.yaml` describes which services listen on which ports during simulation.
  Not useful unless debugging the simulator itself.

If everything went correctly `rollouts` and `aggregate` are usually the only results of interest.
For understanding driving quality metrics and performance tuning, see the
[Operations Guide](OPERATIONS.md).

## Basic debugging

> :warning: This section is about debugging the _configuration_ of the simulator itself (not of
> vehicle behavior within simulation)

The console contains logs from all microservices, and is the first place one should look when
something goes wrong. When an error happens (for example the `rollouts` directory does not appear),
it's best to consult that log to see where the first errors occurred. The microservices may produce
additional logs that can be useful for debugging, but that is not covered here.


# Level 2

In level 2 we learn to customize the simulation (i.e. change the driver policy, change simulated
scenes, etc.) and understand the architecture in more depth.

## AlpaSim Wizard Configuration

AlpaSim wizard is configured via [hydra](https://hydra.cc/docs/intro/) and takes in a `.yaml`
configuration file and arbitrary command line overrides. Example config files are in
`src/wizard/configs/`. We suggest reading [base_config.yaml](/src/wizard/configs/base_config.yaml),
which has detailed comments on the configuration fields.

### Runtime specification

Under the top-level `runtime` item in the `base_config.yaml`, we describe the details of the
simulation to be performed (as opposed to deployment settings under `wizard.*` and `services.*`).

The important configurable fields of `runtime` are:

- `save_dir` - the name of the directory where to save `asl` logs. It needs to be kept in sync with
  wizard mount points. certain modules
- `endpoints` - used to configure simulator scaling properties
- `simulation_config` - specify all the simulation parameters (e.g. timing, cameras,
  vehicle configuration, etc.).

For example, one might change the number of rollouts per scene generated in the configuration files
by running the wizard as follows:

```bash
uv run alpasim_wizard +deploy=local wizard.log_dir=<dir> runtime.simulation_config.n_rollouts=8
```

### Evaluation video layouts
You can choose which video layouts to render via `eval.video.video_layouts`. Available layouts are `DEFAULT` (BEV map, camera, metrics) and `REASONING_OVERLAY` (first-person camera with reasoning text overlay and trajectory chart). To generate reasoning-overlay videos only, override when invoking the wizard, for example:
```bash
uv run alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial driver=[ar1,ar1_runtime_configs] eval.video.video_layouts=[REASONING_OVERLAY]
```
You can also set `eval.video.video_layouts=[DEFAULT,REASONING_OVERLAY]` to render both layouts per rollout.

## Driver

The driver in AlpaSim is a policy for the ego vechicle that takes in sensor inputs and optional
navigation commands, and outputs a trajectory for the ego vehicle to follow, along with other
optional outputs, such as chain-of-causation reasoning text.

The driver is specfied by a pair of config files under `src/wizard/configs/`, one for the driver
service itself, and one for the runtime (so that it provides the inputs required for the specific
driver).

### VaVAM

The wizard uses [VaVAM](https://github.com/valeoai/VideoActionModel) as the default driver. To
explicitly define the driver config, one can use:

```bash
uv run alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_alpamayo driver=[vavam,vavam_runtime_configs]
```

### Alpamayo-R1

To run with the [Alpamayo-R1](https://github.com/NVlabs/alpamayo) 10b model use
`driver=[ar1,ar1_runtime_configs]`.

First, one may download the model weights from HuggingFace:

```bash
huggingface-cli download nvidia/Alpamayo-R1-10B
```

The wizard will use the `HF_HOME` environment variable to find the system HuggingFace cache
(`~/.cache/huggingface` by default). If the model weights do not exists locally, the driver service
will automatiocally download them, but the download may timeout, requiring you to re-run.
Alternatively, you can specify the path to the model directory by setting the
`model.checkpoint_path` configuration field.

Then run the wizard with the following command:

```bash
uv run alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_alpamayo driver=[ar1,ar1_runtime_configs]
```

> :warning: The Alpamayo R1 model is large (10b parameters)--please ensure that your GPU has the
> capacity to run it.

To visualize the predicted chain-of-causation reaoning you can change the generated video layout

```bash
uv run alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_alpamayo driver=[ar1,ar1_runtime_configs] eval.video.video_layouts=[REASONING_OVERLAY]
```

### Transfuser (provisional)

As an example for how to integrate a different driver model, we provide a provisional integration
for the
[Transfuser](https://github.com/autonomousvision/lead?tab=readme-ov-file#beyond-carla-cross-benchmark-deployment)
policy, specifically the Latent TransFuser v6
([LTFv6](<(https://huggingface.co/ln2697/tfv6_navsim)>)) model developed for
[NAVSIM](https://github.com/autonomousvision/navsim).

To run with the Transfuser model use `driver=[transfuser,transfuser_runtime_configs]`.

First, one must download the Transfuser model weights/config from HuggingFace:

```bash
huggingface-cli download longpollehn/tfv6_navsim model_0060.pth --local-dir=data/drivers/transfuser/
huggingface-cli download longpollehn/tfv6_navsim config.json --local-dir=data/drivers/transfuser/
```

Then, run the wizard with the following command:

```bash
uv run alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_transfuser driver=[transfuser,transfuser_runtime_configs]
```

### Log replay driver

If you would like to force the ego vehicle to follow its recorded trajectory, instead of following
the predictions of a policy, you can set
`runtime.endpoints.{physics,trafficsim,controller}.skip: true`,
`runtime.simulation_config.physics_update_mode: NONE` and
`runtime.simulation_config.force_gt_duration_us` to a very high value (20s+).

## Data Preparation

AlpaSim uses [trajdata](https://github.com/NVlabs/trajdata/tree/alpasim) (custom `alpasim` branch)
for unified data loading across different autonomous driving datasets. The trajdata library is
automatically installed via `uv` when you run `setup_local_env.sh`.

Before running simulations, you need to prepare a trajdata cache from your scene data.

### Preparing USDZ Scene Cache

The wizard automatically handles data preparation for downloaded USDZ scenes. However, if you need to
manually prepare or rebuild the cache, use the `prepare_data` tool:

```bash
# Prepare cache from USDZ scenes
uv run python -m alpasim_runtime.prepare_data \
    --desired-data=usdz \
    --data-dir=./data/nre-artifacts/all-usdzs \
    --cache-location=./cache/trajdata_usdz \
    --smooth_trajectories=true \
    --log-level=INFO
```

This command:
- Scans USDZ files in the specified directory
- Extracts trajectory and map data
- Creates a trajdata cache for fast scene loading
- The cache is reused across runs to improve startup time

### Using Configuration Files

For more complex setups or repeated use, create a configuration file (e.g.,
`user_config/config_prepare_usdz.yaml`):

```yaml
data_source:
  desired_data: ["usdz"]
  data_dirs:
    usdz: "./data/nre-artifacts/all-usdzs"
  cache_location: "./cache/trajdata_usdz"
  incl_vector_map: true
  rebuild_cache: false  # Set to true to force rebuild
  rebuild_maps: false   # Set to true to force rebuild
  desired_dt: 0.1       # 10 Hz sampling rate
  num_workers: 4        # Parallel workers for cache creation

smooth_trajectories: true
```

Then run:

```bash
uv run python -m alpasim_runtime.prepare_data \
    --user-config=user_config/config_prepare_usdz.yaml
```

### Cache Location

The trajdata cache contains:
- Preprocessed scene metadata and indices
- Trajectory data in a unified format
- Vector map data (when enabled)
- Scene lookup tables for fast access

You can share the cache directory across machines to avoid redundant preprocessing.

### Rebuilding the Cache

If you encounter data inconsistencies or add new scenes, rebuild the cache:

```bash
# Command line
uv run python -m alpasim_runtime.prepare_data \
    --desired-data=usdz \
    --data-dir=./data/nre-artifacts/all-usdzs \
    --cache-location=./cache/trajdata_usdz \
    --rebuild-cache

# Or in config file, set: rebuild_cache: true
```

> :green_book: The wizard uses configuration files from `user_config/` that include data source
> settings. These configs are automatically used during simulation runs.

## Scenes

The scene in AlpaSim is a NuRec reconstruction of a real-world driving log.

Publicly available NuRec scenes are stored on
[Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/main/sample_set/25.07_release)
and, once downloaded, are placed under `data/nre-artifacts/all-usdzs`. The scenes are identified by
their uuid, rather than their filenames, to prevent versioning issues. The list of currently
available scenes exists in [scenes set](/data/scenes/sim_scenes.csv) and the set of available suites
exists in [scene suites](/data/scenes/sim_suites.csv).

#### Selecting Individual Scenes

For custom scene selection, you can specify scenes manually using `scenes.scene_ids`:

```bash
uv run alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_2 scenes.scene_ids=['clipgt-02eadd92-02f1-46d8-86fe-a9e338fed0b6']
```

If necessary, the scene will automatically be downloaded from Hugging Face to your local
`data/nre-artifacts/all-usdzs` directory. If the download is necessary, ensure you have set your
Hugging Face token in the `HF_TOKEN` environment variable as described in the onboarding
instructions.

> :green_book: Scene ids are defined/viewable in `data/scenes/sim_scenes.csv` :warning: A scene id
> does not uniquely identify the `usdz` file as the scene id comes from the `metadata.yaml` file
> inside the `usdz` zip file. The proper artifact file will be chosen to satisfy the NRE version
> requirements.

#### Using Scene Suites

Scene suites provide pre-validated collections of scenes for testing. To use the public sceneset
with 916 validated scenes (:warning: this will download all the scenes):

```bash
uv run alpasim_wizard +deploy=local scenes.test_suite_id=public_2602 wizard.log_dir=$PWD/tutorial_suite
```

This will run simulations across all 916 scenes in the `public_2602` suite from the 26.02 release dataset.

## Custom components

### Code changes

Code changes in the repo are automatically mounted into the docker containers at runtime, with the
exception that the virtual environment of the container is not synced, so changes that rely on new
dependencies will require rebuilding the container image. To try this out, one can add some logging
statements to the driver code in `src/driver/src/alpasim_driver/` and rerun the wizard.

### Custom container images

The simulation is split into multiple microservices, each running in its own docker container. The
primary requirement for a custom container image is that it exposes a gRPC endpoint compatible with
the expected service interface. The default images used for each service are specified in
[`stable_manifest`](/src/wizard/configs/stable_manifest/oss.yaml); however, these can be overridden
by setting `services.<service>.image` to the desired image name and updating the relevant service
command `services.<service>.command`. For more information about the service interfaces, please see
the [protocol buffer definitions](/src/grpc/alpasim_grpc/v0/).

## Asl log format

`asl` contains most of messages exchanged in the course of a batch simulation as size-delimited
protobuf messages. These files can be read to access detailed information about the course of the
simulation. Aside from being used for evaluation, they can also be useful for debugging model or
simulation behavior. [This notebook](/src/runtime/notebooks/replay_logs_alpamodel.ipynb) shows an
example of reading an `asl` log and "replaying the stimuli" on a driver instance, allowing for
reproducing behavior with your favorite debugger attached.


# Level 3

In level 3 we show how to circumvent the `alpasim_wizard` defined components: this enables use cases
such as enabling breakpoint debugging in components or even replacing components entirely. The basic
idea behind the approach is to:

- Use the `alpasim_wizard` to generate config files without actually running the simulation
- Manually start the desired components with the generated config files
- Use the `alpasim_wizard` generated config files to run the rest of the simulation as normal.

## Manual Driver (Interactive Control)

For interactive control of the ego vehicle with keyboard input, see
[MANUAL_DRIVER.md](MANUAL_DRIVER.md). This allows you to drive through scenarios manually while
visualizing the camera feed in real-time.

## Breakpoint debugging: example with the controller

The following steps might be used to show how to debug the controller component with breakpoints in
the context of a full simulation.

1. (Terminal 1) Run the wizard to generate config files without running the simulation:

   ```bash
   uv run alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_dbg wizard.run_method=NONE  wizard.debug_flags.use_localhost=True
   ```

1. (Terminal 1) `cd` to the generated directory (`tutorial_dbg`) and note the command/port of the
   component to be replaced in `docker-compose.yaml`. For the simulation case, we are looking for
   components in the `sim` profile, which includes `controller-0`, `driver-0`, `physics-0`,
   `runtime-0`, and `sensorsim-0`. Here we will replace `controller-0`, which in this case has been
   allocated port 6003.

1. (Terminal 2) `cd` into the the controller src directory (`<repo_root>/src/controller/`) and
   prepare to start the controller. Note that there are various ways to accomplish this, including
   through an IDE. Add breakpoints as desired in the controller code and then start the controller
   with:

   ```bash
   cd <repo_root>/src/controller/
   mkdir my_controller_log_dir
   # Note: port (6003 in this case) must match the port allocated in docker-compose.yaml
   uv run python -m alpasim_controller.server --port=6003 --log_dir=my_controller_log_dir --log-level=INFO
   ```

1. (Terminal 1) Start the rest of the simulation with docker compose:

   ```bash
   docker compose -f docker-compose.yaml --profile sim up runtime-0 driver-0 physics-0 sensorsim-0
   ```

### Using VSCode Debugger (Optional)

For VSCode users, instead of running the controller from the command line (step 3), you can use the
built-in debugger:

1. Create or update `.vscode/launch.json` with:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Controller (Level 3 Tutorial)",
      "type": "debugpy",
      "request": "launch",
      "module": "alpasim_controller.server",
      "justMyCode": false,
      "cwd": "${workspaceFolder}/src/controller",
      "args": ["--port=6003", "--log_dir=my_controller_logdir", "--log-level=INFO"],
      "console": "integratedTerminal"
    }
  ]
}
```

2. Set breakpoints in the controller code
1. Press F5 (or go to Run and Debug → "Debug Controller")
1. Your breakpoints will hit as the simulation runs!

**Note:** Make sure the `--port` argument matches the port allocated in `docker-compose.yaml`.

## Breakpoint debugging: example with the runtime

If the `runtime` is the service being debugged, there are a few things that change. For one, it is
expected that the other services are up and running before the `runtime` is brought up, so the
ordering of steps will change. Additionally, one can speed up iteration by preventing the simulation
from shutting down the docker containers after each simulation by setting
`runtime.endpoints.do_shutdown=False` in the wizard command line.

1. (Terminal 1) Run the wizard to generate config files without running the simulation:
   ```bash
   uv run alpasim_wizard +deploy=local \
   wizard.log_dir=$PWD/tutorial_dbg_runtime \
   wizard.run_method=NONE  \
   wizard.debug_flags.use_localhost=True \
   runtime.endpoints.do_shutdown=False
   ```
1. (Terminal 1) `cd` to the generated directory (`tutorial_dbg_runtime`) and start the non-runtime
   services:
   `bash     docker compose -f docker-compose.yaml --profile sim up driver-0 controller-0 physics-0 sensorsim-0     `
1. (Terminal 2) `cd` into the the runtime src directory (`<repo_root>/src/runtime/`) and prepare to
   start the runtime. The exact command paths will vary, but, to use the configuration generated
   from the earlier steps, an example command would be:
   ```bash
   cd <repo_root>/src/runtime/
   # Following command is based on the docker-compose.yaml generated by the wizard
   # Ensure the user config contains the data_source configuration
   uv run python -m alpasim_runtime.simulate \
     --user-config=../../tutorial_dbg_runtime/generated-user-config-0.yaml \
     --network-config=../../tutorial_dbg_runtime/generated-network-config.yaml \
     --log-dir=../../tutorial_dbg_runtime \
     --eval-config=../../tutorial_dbg_runtime/eval-config.yaml \
     --log-level=INFO
   ```

### Using VSCode Debugger (Optional)

For VSCode users, instead of running the runtime from the command line (step 3), you can use the
built-in debugger:

1. Add this configuration to `.vscode/launch.json`:

```json
{
  "name": "Debug Runtime (Level 3 Tutorial)",
  "type": "debugpy",
  "request": "launch",
  "module": "alpasim_runtime.simulate",
  "justMyCode": false,
  "cwd": "${workspaceFolder}/src/runtime",
  "args": [
    "--user-config=../../tutorial_dbg_runtime/generated-user-config-0.yaml",
    "--network-config=../../tutorial_dbg_runtime/generated-network-config.yaml",
    "--eval-config=../../tutorial_dbg_runtime/eval-config.yaml",
    "--log-dir=../../tutorial_dbg_runtime",
    "--log-level=INFO"
  ],
  "console": "integratedTerminal",
  "env": {
    "PYTHONPATH": "${workspaceFolder}/src/grpc:${workspaceFolder}/src/eval/src:${workspaceFolder}/src/utils:${workspaceFolder}/src/runtime:${env:PYTHONPATH}"
  }
}
```

2. Set breakpoints in the runtime code
1. Press F5 (or go to Run and Debug → "Debug Runtime")
1. Your breakpoints will hit as the simulation runs!
