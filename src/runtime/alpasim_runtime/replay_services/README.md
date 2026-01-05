# Replay Services

Services for replaying recorded simulation data in Alpasim.
Primary use case at the moment is the `test_runtime_integration_replay.py` test
which checks that the runtime sends the same requests as in the logs.
This is helpful to make sure nothing changed after refactoring.
**Note that this test is manual as we don't want to burden developers to update
the recorded ASL logs every time they make an intentional change to runtime**

## Components

- **ASLReplay**: Replays ASL (Alpasim Scene Language) files containing actor trajectories and scene data
- **ReplayServiceBase**: Base class for implementing custom replay services

## Generating replay files

To generate the needed files to be replayed, run, from src/wizard:

```
RUN_DIR=/home/migl/workspace/alpasim/.wizard \
uv run python -m alpasim_wizard \
    wizard.log_dir=${RUN_DIR} \
    +deploy=local \
    runtime.endpoints.trafficsim.skip=False \
    scenes.scene_ids="[clipgt-c14c031a-8c17-4d08-aa4d-23c020a6871e]" \
    runtime.default_scenario_parameters.n_sim_steps=60
```

And copy the asl log, network-config and user-config files to
src/runtime/tests/data/integration.

## Changing the scene

If you change the scene_id, make sure to also add the updated usdz file to the
folder - it's recommended to modify the contents of the usdz file and remove
all the large rendering artifacts we don't need.
You can zip it back up with (from within the unziped usdz folder)
```
zip -r -0 6ea1c7a3-98b7-4adc-b774-4d9526371a0b.usdz ./*
```

To upload it to git, use `git lfs track "path/to/file.(usdz|asl)"`
