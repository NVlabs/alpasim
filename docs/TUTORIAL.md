# AlpaSim tutorial: introduction
This tutorial makes three assumptions
1. It targets an AlpaSim user rather than an AlpaSim developer
2. It treats docker compose` as the primary execution environment.
3. It focuses on letting the user do simple things quick and leaves detail for later. This is
reflected in subdivision into three levels of complexity.

# Level 1
In level 1 we focus we run a default simulation, learn how to interpret the results, and perform
basic debugging.

## Architecture of AlpaSim
AlpaSim consists of multiple networked microservices (renderer, physics simulation, runtime,
controller, driver, traffic simulation). The AlpaSim runtime requests observed video frames from
the renderer and egomotion history from the controller, communicates with the physics microservice
to constrain actors to the road surface, and provides the information to the driver, with the
expectation of receiving driving decisions in return to close the loop.

This repository contains the implementations of a subset of the services needed to execute the
simulation as well as config files and infra code necessary to bring the microservices up via
docker/enroot.

## Running with docker compose
Let's start by executing a run with default settings.
1. Follow [instructions in onboarding](/docs/ONBOARDING.md) to ensure necessary dependencies have
been installed
2. Set up your environment with:
    * `source setup_local_env.sh`
    * This will compile protos, download an example driver model, ensure you have a valid Hugging
    Face token, and install the `alpasim_wizard` command line tool.
3. Run the wizard to create the necessary config files, download the scene (if necessary), and run
a simulation
    * `alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial`
    * This will create a `tutorial/` directory with all necessary config files and run the simulation

## Results structure
The simulation logs/output will be in the created `tutorial` directory. For a visualization of
the results, an `mp4` file is created in `tutorial/eval/videos/clipgt-026d..._0.mp4`. The full results
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
├── asl
│   └── clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3
│       └── 814f3c22-bb78-11f0-a5f3-2f64b47b8685
│           ├── 0.asl
│           └── _complete
├── avmf-config.yaml
├── controller
│   └── alpasim_controller_814f3c22-bb78-11f0-a5f3-2f64b47b8685.csv
├── docker-compose.yaml
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
* `asl` contains logs of simulation messages between components in each rollout and can be used to analyze AV behavior and
calculate metrics. The logs are organized into
`asl/{scenario.scene_id}/{rollout_id}.*` - in this case we have 1 scenes with one
batch of a single rollout.
  * `.asl` files which record the messages exchanged within the simulation. These are useful for debugging the simulator behavior and replaying events.
* `eval/` contains per-rollout evaluation results:
  * `metrics_unprocessed.parquet` - Raw driving quality metrics for each rollout
  * `videos/` - Video recordings of each rollout
* `aggregate/` contains aggregated results across all rollouts:
  * `metrics_results.txt` - Formatted table of driving scores (mean, std, quantiles)
  * `metrics_results.png` - Visual summary of driving quality metrics
  * `metrics_unprocessed.parquet` - Combined metrics from all rollouts
  * `videos/` - Videos organized by violation type (collision_at_fault, offroad, etc.)
* `metrics/` contains performance profiling data (see [OPERATIONS.md](OPERATIONS.md#how-do-i-view-performance-metrics) for details):
  * `metrics.prom` - Prometheus metrics from simulation
  * `metrics_plot.png` - Performance visualization (CPU/GPU/RPC metrics)
* `driver` is a directory with logs written by the driver service, useful to debug policy-internal problems.
* `wizard-config.yaml` contains the config the wizard used for this run **after applying the inheritance of hydra**. This is useful for debugging configuration issues.
* `generated-user-config-{ARRAY_ID}.yaml` contains an expanded version of the simulation config provided by the user, possibly split into chunks when simulating on multiple nodes.
* `trafficsim-config.yaml`. A copy of the traffic simulation config used for simulation, useful for debugging traffic simulation.
* `generated-network-config.yaml` describes which services listen on which ports during simulation.
Not useful unless debugging the simulator itself.

If everything went correctly `asl` and `eval` are usually the only results of interest. For understanding driving quality metrics and performance tuning, see the [Operations Guide](OPERATIONS.md).


## Basic debugging

> :warning: This section is about debugging the _configuration_ of the simulator itself (not of
vehicle behavior within simulation)

The console contains logs from all microservices, and is the first place one should look when
something goes wrong. When an error happens (for example the `asl` directory does not appear), it's
best to consult that log to see where the first errors occurred. The microservices may produce
additional logs that can be useful for debugging, but that is not covered here.

# Level 2
In level 2 we learn to customize the simulation (i.e. run our own code, change simulated scenes,
change the driver model, etc.) and understand the architecture in more depth.

## AlpaSim Wizard Configuration
At *level 1* the wizard was run using some default arguments, but here we learn to interact with
it to:
* Change scenario configuration parameters
* Make changes to the code
* Scale the simulator (i.e. replicate services under heavy load)

## Wizard config files
AlpaSim wizard is configured via [hydra](https://hydra.cc/docs/intro/) and takes in a `.yaml`
configuration file and arbitrary command line overrides. Example config files are in
`src/wizard/configs/`. We suggest reading
[base_config.yaml](/src/wizard/configs/base_config.yaml), which has detailed comments on the
configuration fields.

### Runtime specification
Under the top-level `runtime` item in the `base_config.yaml`, we describe the details of the
simulation to be performed (as opposed to deployment settings under `wizard.*` and `services.*`).

The important configurable fields of `runtime` are:
* `save_dir` - the name of the directory where to save `asl` logs. It needs to be kept in sync with
wizard mount points.
certain modules
* `endpoints` - used to configure simulator scaling properties
* `default_scenario_parameters` - specify all the simulation parameters (e.g.
  timing, cameras, vehicle configuration, etc.).

For example, one might change the number of rollouts per scene generated in the configuration files
by running the wizard as follows:
```bash
alpasim_wizard +deploy=local wizard.log_dir=<dir> runtime.default_scenario_parameters.n_rollouts=8
```

### Run with the Alpamayo R1 Driver Model
To run with the Alpamayo R1 driver model, one must apply two additional configuration changes: one
for the driver, and one for the runtime. The driver service will automatiocally download the model
weights if they do not exist locally.
> :warning: The Alpamayo R1 model is large--please ensure that your GPU has the capacity to run it.
> :warning: The Alpamayo R1 model is large--the download may timeout, requiring you to re-run.

```bash
alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_alpamayo driver=[ar1,ar1_runtime_configs]
```

If you have already downloaded the Alpamayo R1 model weights, you can specify the path to the model
directory by setting the `model.checkpoint_path` configuration field.

### Code Changes
Code changes in the repo are automatically mounted into the docker containers at runtime, with the
exception that the virtual environment of the container is not synced, so changes that rely on new
dependencies will require rebuilding the container image. To try this out, one can add some logging
statements to the driver code in `src/driver/src/alpasim_driver/` and rerun the wizard.


### Scenes
Publicly available scenes are stored on
[Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/main/sample_set/25.07_release)
and, once downloaded, are placed under `data/nre-artifacts/all-usdzs`. The scenes are identified by
their uuid, rather than their filenames, to prevent versioning issues. The list of currently
available scenes exists in [scenes set](/data/scenes/sim_scenes.csv) and the set of available
suites exists in [scene suites](/data/scenes/sim_suites.csv).

#### Selecting Individual Scenes
For custom scene selection, you can specify scenes manually using `scenes.scene_ids`:

```bash
alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_2 scenes.scene_ids=['clipgt-02eadd92-02f1-46d8-86fe-a9e338fed0b6']
```

If necessary, the scene will automatically be downloaded from Hugging Face to your local
`data/nre-artifacts/all-usdzs` directory. If the download is necessary, ensure you have set your
Hugging Face token in the `HF_TOKEN` environment variable as described in the onboarding
instructions.

> :green_book: Scene ids are defined/viewable in `data/scenes/sim_scenes.csv`
> :warning: A scene id does not uniquely identify the `usdz` file as the scene id comes from
the `metadata.yaml` file inside the `usdz` zip file. The proper artifact file will be chosen to
satisfy the NRE version requirements.

#### Using Scene Suites
Scene suites provide pre-validated collections of scenes for testing. To use the public sceneset
with 901 validated scenes (:warning: this will download all the scenes):

```bash
alpasim_wizard +deploy=local scenes.test_suite_id=public_2507_ex_failures wizard.log_dir=$PWD/tutorial_suite
```

This will run simulations across all 910 scenes in the `public_2507_ex_failures` suite, which
excludes problematic scenes from the full 25.07 release dataset.


### Log replay
You can set `runtime.endpoints.{physics,trafficsim,controller}.skip: true`,
`runtime.default_scenario_parameters.physics_update_mode: NONE` and
`runtime.default_scenario_parameters.force_gt_duration_us` to a very high value (20s+) to obtain
log-replay behavior (with NRE-rendered images).

## Custom container images
The simulation is split into multiple microservices, each running in its own docker container. The
primary requirement for a custom container image is that it exposes a gRPC endpoint compatible with
the expected service interface. The default images used for each service are specified in
[`stable_manifest`](/src/wizard/configs/stable_manifest/oss.yaml); however, these can be overridden
by setting `services.<service>.image` to the desired image name and updating the relevant service
command `services.<service>.command`. For more information about the service interfaces, please
see the [protocol buffer definitions](/src/grpc/alpasim_grpc/v0/).


## Asl log format
`asl` contains most of messages exchanged in the course of a batch simulation as size-delimited
protobuf messages. These files can be read to access detailed information about the course of the
simulation. Aside from being used for evaluation, they can also be useful for debugging model
or simulation behavior.
[This notebook](/src/runtime/notebooks/replay_logs_alpamodel.ipynb) shows an example of reading an
`asl` log and "replaying the stimuli" on a driver instance, allowing for reproducing behavior with
your favorite debugger attached.

# Level 3
In level 3 we show how to circumvent the `alpasim_wizard` defined components: this enables
use cases such as enabling breakpoint debugging in components or even replacing components entirely.
The basic idea behind the approach is to:

* Use the `alpasim_wizard` to generate config files without actually running the simulation
* Manually start the desired components with the generated config files
* Use the `alpasim_wizard` generated config files to run the rest of the simulation as normal.

## Breakpoint debugging: example with the controller
The following steps might be used to show how to debug the controller component with breakpoints in
the context of a full simulation.
1. (Terminal 1) Run the wizard to generate config files without running the simulation:
    ```bash
    alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_dbg wizard.run_method=NONE  wizard.debug_flags.use_localhost=True
    ```
2. (Terminal 1) `cd` to the generated directory (`tutorial_dbg`) and note the command/port of the
component to be replaced in `docker-compose.yaml`. For the simulation case, we are looking for
components in the `sim` profile, which includes `controller-0`, `driver-0`, `physics-0`,
`runtime-0`, and `sensorsim-0`. Here we will replace `controller-0`, which in this case has been
allocated port 6003.
3. (Terminal 2) `cd` into the the controller src directory (`<repo_root>/src/controller/`)
and prepare to start the controller. Note that there are various ways to accomplish this, including
through an IDE. Add breakpoints as desired in the controller code and then start the controller
with:

    ```bash
    cd <repo_root>/src/controller/
    mkdir my_controller_log_dir
    # Note: port (6003 in this case) must match the port allocated in docker-compose.yaml
    uv run python -m alpasim_controller.server --port=6003 --log_dir=my_controller_log_dir --log-level=INFO
    ```
4. (Terminal 1) Start the rest of the simulation with docker compose:
    ```bash
    docker compose -f docker-compose.yaml --profile sim up runtime-0 driver-0 physics-0 sensorsim-0
    ```

### Using VSCode Debugger (Optional)

For VSCode users, instead of running the controller from the command line (step 3), you can use the built-in debugger:

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
            "args": [
                "--port=6003",
                "--log_dir=my_controller_logdir",
                "--log-level=INFO"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

2. Set breakpoints in the controller code
3. Press F5 (or go to Run and Debug → "Debug Controller")
4. Your breakpoints will hit as the simulation runs!

**Note:** Make sure the `--port` argument matches the port allocated in `docker-compose.yaml`.

## Breakpoint debugging: example with the runtime
If the `runtime` is the service being debugged, there are a few things that change. For one, it is
expected that the other services are up and running before the `runtime` is brought up, so the
ordering of steps will change. Additionally, one can speed up iteration by preventing the
simulation from shutting down the docker containers after each simulation by setting
`runtime.endpoints.do_shutdown=False` in the wizard command line.

1. (Terminal 1) Run the wizard to generate config files without running the simulation:
    ```bash
    alpasim_wizard +deploy=local \
    wizard.log_dir=$PWD/tutorial_dbg_runtime \
    wizard.run_method=NONE  \
    wizard.debug_flags.use_localhost=True \
    runtime.endpoints.do_shutdown=False
    ```
2. (Terminal 1) `cd` to the generated directory (`tutorial_dbg_runtime`) and start the non-runtime
services:
    ```bash
    docker compose -f docker-compose.yaml --profile sim up driver-0 controller-0 physics-0 sensorsim-0
    ```
3. (Terminal 2) `cd` into the the runtime src directory (`<repo_root>/src/runtime/`)
and prepare to start the runtime. The exact command paths will vary, but, to use the configuration
generated from the earlier steps, an example command would be:
    ```bash
    cd <repo_root>/src/runtime/
    # Following command is based on the docker-compose.yaml generated by the wizard
    uv run python -m alpasim_runtime.simulate \
    --usdz-glob=../../data/nre-artifacts/all-usdzs/**/*.usdz \
    --user-config=../../tutorial_dbg_runtime/generated-user-config-0.yaml \
    --network-config=../../tutorial_dbg_runtime/generated-network-config.yaml \
    --log-dir=../../tutorial_dbg_runtime \
    --log-level=INFO
    ```

### Using VSCode Debugger (Optional)

For VSCode users, instead of running the runtime from the command line (step 3), you can use the built-in debugger:

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
        "--usdz-glob=../../data/nre-artifacts/all-usdzs/**/*.usdz",
        "--user-config=../../tutorial_dbg_runtime/generated-user-config-0.yaml",
        "--network-config=../../tutorial_dbg_runtime/generated-network-config.yaml",
        "--log-dir=../../tutorial_dbg_runtime",
        "--log-level=INFO"
    ],
    "console": "integratedTerminal"
}
```

2. Set breakpoints in the runtime code
3. Press F5 (or go to Run and Debug → "Debug Runtime")
4. Your breakpoints will hit as the simulation runs!
