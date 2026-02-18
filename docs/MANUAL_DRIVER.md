# Manual Driver

The manual driver allows interactive control of the ego vehicle using keyboard input. This is useful
for:

- Debugging and exploring simulation scenarios interactively
- Testing specific driving behaviors manually
- Visualizing camera feeds in real-time

## Requirements

- A display (X11 or Wayland) - the manual driver requires a GUI for keyboard input

## Controls

| Key       | Action                                   |
| --------- | ---------------------------------------- |
| W / UP    | Accelerate (increase target speed)       |
| S / DOWN  | Brake/Decelerate (decrease target speed) |
| A / LEFT  | Steer left                               |
| D / RIGHT | Steer right                              |
| SPACE     | Emergency stop (zero speed)              |
| ESC / Q   | Quit                                     |

The driver generates constant-curvature arc trajectories based on the current steering angle and
speed, which are then tracked by the MPC controller.

## Running the Manual Driver

While the bulk of Alpasim runs inside docker-compose, the manual driver runs as an external service
(python script) on your machine - hence the term "external driver" later on.

### Step 1: Start the Manual Driver

From the repository root, start the driver:

```bash
uv run --project src/driver python -m alpasim_driver.main \
  --config-path=configs --config-name=manual
```

The driver will start and display a pygame window showing the camera feed. It binds to
`0.0.0.0:6789` by default (all network interfaces).

### Step 2: Launch the Simulator

Run the wizard with the `local_external_driver` deploy config:

```bash
uv run --project src/wizard alpasim_wizard \
  +deploy=local_external_driver \
  wizard.log_dir=$PWD/manual_run \
  scenes.scene_ids='["your-scene-id"]'
```

The runtime will connect to your driver at `localhost:6789`.

### Running Driver on a Separate Machine

If your driver runs on a different host or port, override the address with the driver's external IP
(reported in its logs upon launch):

```bash
uv run --project src/wizard alpasim_wizard \
  +deploy=local_external_driver \
  wizard.log_dir=$PWD/manual_run \
  scenes.scene_ids='["your-scene-id"]' \
  wizard.external_services.driver='["192.168.1.100:6789"]'
```

## Configuration

The manual driver configuration is in `src/driver/configs/manual.yaml`. Key settings:

| Setting                 | Default                        | Description                           |
| ----------------------- | ------------------------------ | ------------------------------------- |
| `host`                  | `0.0.0.0`                      | Interface to bind to (all interfaces) |
| `port`                  | `6789`                         | Port to listen on                     |
| `inference.use_cameras` | `["camera_front_wide_120fov"]` | Camera to display                     |
| `log_level`             | `INFO`                         | Logging verbosity                     |

## Simulation Settings

The `local_external_driver` deploy config includes settings optimized for interactive use:

- **10 Hz control rate** (vs. 2 Hz for batch evaluation) for smoother response
- **1920x1080 camera resolution** for better visualization
- **Localhost networking** so containers can reach the external driver

These can be customized by overriding `runtime.default_scenario_parameters` on the command line.
