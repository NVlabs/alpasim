# AlpaSim tutorial: introduction
This tutorial makes three assumptions
1. It targets an AlpaSim user rather than an AlpaSim developer
2. It treats docker compose` as the primary execution environment.
3. It focuses on letting the user do simple things quick and leaves detail for later. This is
reflected in subdivision into three levels of complexity.

[TOC]

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
1. Ensure that you have the sample LFS artifact (`git lfs pull`)
2. Download and install `uv` if not yet done: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
   Alternatively, run `uv self update` as older versions have been reported to not work.
3. Set up your environment with:
    * `source setup_local_env.sh`
4. Run the wizard to create the necessary config files and run a simulation
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
│           ├── dist_to_gt_trajectory
│           │   └── clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3_814f3c22-bb78-11f0-a5f3-2f64b47b8685_0.mp4 -> ../../all/clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3_814f3c22-bb78-11f0-a5f3-2f64b47b8685_0.mp4
│           └── offroad
├── alpasim-runtime.prom
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
* `driver` is a directory with logs written by the driver service, useful to debug policy-internal problems.
* `wizard-config.yaml` contains the config the wizard used for this run **after applying the inheritance of hydra**. This is useful for debugging configuration issues.
* `generated-user-config-{ARRAY_ID}.yaml` contains an expanded version of the simulation config provided by the user, possibly split into chunks when simulating on multiple nodes.
* `trafficsim-config.yaml`. A copy of the traffic simulation config used for simulation, useful for debugging traffic simulation.
* `generated-network-config.yaml` describes which services listen on which ports during simulation.
Not useful unless debugging the simulator itself.

If everything went correctly `asl` and `eval` are usually the only results of interest.


## Basic debugging

> :warning: This section is about debugging the _configuration_ of the simulator itself (not of
vehicle behavior within simulation)

The console contains logs from all microservices, and is the first place one should look when
something goes wrong. When an error happens (for example the `asl` directory does not appear), it's
best to consult that log to see where the first errors occurred. The microservices may produce
additional logs that can be useful for debugging, but that is not covered here.

# Level 2
In level 2 we learn to customize the simulation (i.e. run our own code, change simulated scenes)
and understand the architecture in more depth.

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

### Code Changes
Code changes in the repo are automatically mounted into the docker containers at runtime, with the
exception that the virtual environment of the container is not synced, so changes that rely on new
dependencies will require rebuilding the container image. To try this out, one can add some logging
statements to the driver code in `src/driver/src/alpasim_driver/` and rerun the wizard.


### Scenes
Additional scenes are stored on
[Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/main/sample_set/25.07_release)
and, once downloaded, should be placed somewhere under `data/nre-artifacts/all-usdzs`.

#### Downloading Scenes
The huggingface cli can be used to download additional scenes. For instance:
```bash
hf download --repo-type=dataset \
    --local-dir=data/nre-artifacts/all-usdzs \
    nvidia/PhysicalAI-Autonomous-Vehicles-NuRec \
    sample_set/25.07_release/Batch0001/02eadd92-02f1-46d8-86fe-a9e338fed0b6/02eadd92-02f1-46d8-86fe-a9e338fed0b6.usdz
```

#### Using Scene Suites
Scene suites provide pre-validated collections of scenes for testing.

> :information_source: **Prerequisite:** Scene suites assume you have already downloaded the required scenes from [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/main/sample_set/25.07_release) to your local `data/nre-artifacts/` directory. See the [Downloading Scenes](#downloading-scenes) section above for details.

To use the public sceneset with 901 validated scenes:

```bash
alpasim_wizard +deploy=local +experiment/scenes=public_2507_all_ex19 wizard.log_dir=$PWD/tutorial_suite
```

This will run simulations across all 901 scenes in the `public_2507_all_ex19` suite, which excludes 19 problematic scenes from the full 25.07 release dataset.

#### Selecting Individual Scenes
For custom scene selection, you can specify scenes manually using `scenes.scene_ids`:

```bash
alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial_2 scenes.scene_ids=['clipgt-02eadd92-02f1-46d8-86fe-a9e338fed0b6']
```

> :green_book: A scene id is typically (but not necessarily) `clipgt-<filename without .usdz>`
> :warning: A scene id does not uniquely identify the `usdz` file as the scene id comes from
the `metadata.yaml` file inside the `usdz` zip file. The proper artifact file will be chosen to
satisfy the NRE version requirements.

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
