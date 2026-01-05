# Alpasim Operations Guide

This guide covers common operational tasks for tuning, optimizing, and troubleshooting Alpasim.

## Performance Tuning

### How do I change replica counts and GPU distribution?

The number of service replicas and their GPU assignments are configured in deployment configs located in `src/wizard/configs/deploy/`:

- **Local workstation**: `local_oss.yaml`

#### Understanding the Configuration

Each service has two key parameters:

```yaml
services:
  sensorsim:
    replicas_per_container: 4          # Number of service replicas per container
    gpus: [0, 1, 2, 3]                  # GPUs to create containers on
```

**How it works**:
- **One container per GPU** (or one container total if `gpus: null`)
- Each container runs `replicas_per_container` service instances
- Total replicas = `nr_gpus * replicas_per_container`

Example:
- `gpus: [0, 1, 2, 3]` --> 4 containers (one per GPU)
- `replicas_per_container: 4` --> 4 replicas per container
- **Total**: 4 * 4 = 16 service replicas

#### Balancing Replicas and Concurrent Rollouts

Total simulation throughput capacity is determined by:
```
Total capacity = nr_gpus * replicas_per_container * n_concurrent_rollouts
```
where **`n_concurrent_rollouts`** is the number of rollouts (simulation episodes) each service replica can process simultaneously. This controls how many scenes can be simulated in parallel.

All services must have equal total capacity to avoid bottlenecks. Example from `local_oss.yaml` scaled up:

```yaml
services:
  sensorsim:
    replicas_per_container: 4
    gpus: [0, 1]

  driver:
    replicas_per_container: 8
    gpus: [2, 3]

  controller:
    replicas_per_container: 16
    gpus: null  # CPU-only: 1 container

runtime:
  endpoints:
    sensorsim:
      n_concurrent_rollouts: 4   # 2 GPUs * 4 replicas * 4 concurrent = 32

    driver:
      n_concurrent_rollouts: 2   # 2 GPUs * 8 replicas * 2 concurrent = 32

    controller:
      n_concurrent_rollouts: 2   # 1 CPU * 16 replicas * 2 concurrent = 32
```

### How do I change the model?

By default, the VaVam driver and model are used. The model weights are downloaded using `data/download_vavam_assets.sh` and stored in `data/vavam-driver/`.

#### Using a Different Model

To use a custom model, mount a custom vavam-driver directory:
```bash
uv run alpasim_wizard +deploy=local_oss \
    wizard.log_dir=runs/{DATETIME} \
    defines.vavam_driver=/path/to/custom/vavam-driver
```
**Default location**: `data/vavam-driver/` (in repository root)
The wizard mounts `defines.vavam_driver` as `/mnt/vavam_driver` in the container and the driver loads the model from that path.

#### Using a Different Driver/Inference Code

To use a custom driver container image:
```bash
uv run alpasim_wizard +deploy=local_oss \
    wizard.log_dir=runs/{DATETIME} \
    services.driver.image=<your-registry>/<your-driver-image>:<tag>
```
Your custom image must expose a gRPC endpoint compatible with the driver service interface (see [protocol buffer definitions](/src/grpc/alpasim_grpc/v0/)).

For development of driver code within this repository, changes to `src/driver/` are automatically mounted into containers at runtime (see [Code Changes](TUTORIAL.md#code-changes) in TUTORIAL.md).

### How do I change inference frequency?

Changing inference frequency is complex and requires coordinating multiple timing parameters.

#### Understanding the Parameters

The simulator has multiple synchronized "clocks":
1. **Driver inference** (`control_timestep_us`) - How often the model makes decisions
2. **Camera frames** (`frame_interval_us`) - How often cameras capture images
3. **GPS/Pose updates** (`egopose_interval_us`) - How often position is updated
4. **Simulation start** (`time_start_offset_us`) - Initial offset to avoid artifacts

For correct operation, these must be mathematically aligned.

#### Step-by-Step Walkthrough

**Scenario 1: Simple frequency change (matching camera and inference rates)**

To change to 5Hz inference (200ms between decisions):

1. **Set inference frequency** (`control_timestep_us`):
   ```bash
   runtime.default_scenario_parameters.control_timestep_us=200000  # 200ms = 5Hz
   ```

2. **Match GPS update rate** (`egopose_interval_us` must equal `control_timestep_us`):
   ```bash
   runtime.default_scenario_parameters.egopose_interval_us=200000
   ```

3. **Set time offset** (must be a multiple of `control_timestep_us`):
   ```bash
   runtime.default_scenario_parameters.time_start_offset_us=600000  # 3 * 200ms
   ```

4. **Match camera frame rate** (VaVam default has 1 camera):
   ```bash
   runtime.default_scenario_parameters.cameras.0.frame_interval_us=200000
   ```

   For configs with 2 cameras (e.g., `+cameras=2cam`), also set:
   ```bash
   runtime.default_scenario_parameters.cameras.1.frame_interval_us=200000
   ```

**Full command**:
```bash
uv run alpasim_wizard +deploy=local_oss \
    wizard.log_dir=runs/{DATETIME} \
    runtime.default_scenario_parameters.control_timestep_us=200000 \
    runtime.default_scenario_parameters.egopose_interval_us=200000 \
    runtime.default_scenario_parameters.time_start_offset_us=600000 \
    runtime.default_scenario_parameters.cameras.0.frame_interval_us=200000
```

Note: Add `cameras.1.frame_interval_us=200000` if using 2-camera configs

**Scenario 2: High-rate camera with lower inference rate**

To use 30Hz cameras (33.3ms) but 10Hz inference (100ms):

1. **Camera captures at 30Hz**: `frame_interval_us=33334` (33.3ms)
2. **Inference runs at 10Hz**: `control_timestep_us=100002` (must be 3 × 33334)
3. **Subsample frames**: `driver.inference.Cframes_subsample=3` (use every 3rd frame)
4. **Egopose matches inference**: `egopose_interval_us=100002`
5. **Time offset aligns**: `time_start_offset_us=300006` (3 × 100002)

**Full command** (based on `sim/20s_at_30Hz.yaml`):
```bash
uv run alpasim_wizard +deploy=local_oss \
    wizard.log_dir=runs/{DATETIME} \
    runtime.default_scenario_parameters.control_timestep_us=100002 \
    runtime.default_scenario_parameters.egopose_interval_us=100002 \
    runtime.default_scenario_parameters.time_start_offset_us=300006 \
    runtime.default_scenario_parameters.cameras.0.frame_interval_us=33334 \
    ++driver.inference.Cframes_subsample=3
```

Note: Add `cameras.1.frame_interval_us=33334` if using 2-camera configs.

#### Validation

The `assert_zero_decision_delay` flag (enabled by default in OSS configs) validates timing synchronization at runtime. It checks that:
- Camera frames complete exactly at decision time
- Egopose updates complete exactly at decision time

If misconfigured, the simulator will error with messages like:
```
Camera camera_front_wide_120fov out of sync with planning.
Last started frame finishes at X which is Y microseconds away from decision time Z.
```

**What it does**: At each control step, before calling the driver, the runtime verifies that the last camera frame and egopose update completed exactly at `now_us` (zero delay). This ensures the model receives perfectly synchronized data.

**Testing your configuration**:
```bash
# The flag is true by default, but you can explicitly set it:
runtime.default_scenario_parameters.assert_zero_decision_delay=true
```

#### Common Frequencies

Based on actual config files in `src/wizard/configs/`:

| Frequency | `control_timestep_us` | `egopose_interval_us` | `time_start_offset_us` | Notes |
|-----------|----------------------|----------------------|----------------------|-------|
| 2Hz | 500000 (500ms) | 500000 | 500000 (1×) or 1500000 (3×) | VaVam default |
| 5Hz | 200000 (200ms) | 200000 | 600000 (3×) | Example config |
| 10Hz | 100000 (100ms) | 100000 | 300000 (3×) | Base config default |
| 30Hz | 33334 (33.3ms) | 33334 | 100002 (3×) | High frequency |

**Pattern**: Most configs use `time_start_offset_us = 3 × control_timestep_us` to avoid artifacts at scene start.

**See also**:
- [src/runtime/README.md - Zero delay mode](../src/runtime/README.md#zero-delay-mode) for synchronization requirements
- `src/wizard/configs/driver/vavam_runtime_configs.yaml` for a 2Hz example

## Viewing Results and Metrics

### Where are simulation results stored?

After a run completes, results are in `wizard.log_dir` (e.g., `runs/{RUN_DIR}/`):

- **`asl/`** - Simulation logs (`.asl` files for debugging)
- **`eval/`** - Per-rollout driving quality metrics (`metrics_unprocessed.parquet`) and videos
- **`aggregate/`** - Aggregated results across all rollouts:
  - `metrics_results.txt` - Formatted table of driving scores
  - `metrics_results.png` - Visual summary of driving quality metrics
  - `metrics_unprocessed.parquet` - Combined metrics from all rollouts
  - `videos/` - Organized by violation types
- **`metrics/`** - Performance profiling data:
  - `metrics.prom` - Prometheus metrics from simulation
  - `metrics_plot.png` - Performance visualization (CPU/GPU/RPC metrics)
- **`txt-logs/`** - Service logs for debugging
- **`wizard-config.yaml`** - Resolved configuration used for this run

See [TUTORIAL.md - Results Structure](TUTORIAL.md#results-structure) for detailed breakdown.

### Understanding Driving Quality Metrics

The simulation evaluates driving quality across multiple dimensions. Results are in `aggregate/metrics_results.txt` and visualized in `aggregate/metrics_results.png`.

#### Key Metrics

**Safety Metrics** (binary: 0 = pass, 1 = fail):
- **`collision_at_fault`**: Driver caused a collision (front/lateral impact)
- **`collision_rear`**: Rear-end collision (not at fault)
- **`offroad`**: Vehicle drove off the road

**Performance Metrics** (continuous):
- **`dist_to_gt_trajectory`**: Maximum distance from ground truth path (meters)
  - Lower is better; indicates how closely the driver follows expected routes
  - Aggregated using MAX over time (worst deviation during the drive)
- **`duration_frac_20s`**: Fraction of 20s drive completed before any failure
  - 1.0 = completed full 20s without issues
  - <1.0 = failed early (collision, off-road, or excessive deviation)

**Distance Between Incidents**:
- **`avg_dist_between_incidents`**: Average km traveled per incident (collision or offroad)
  - Higher is better; measures safety over distance
- **`avg_dist_between_incidents_at_fault`**: Average km traveled per at-fault incident
  - Higher is better; excludes rear-end collisions not caused by the driver

#### Interpreting the Results

The `aggregate/metrics_results.txt` file shows statistics (mean, std, min, max, quantiles) for each metric across all rollouts. For example:

```
collision_at_fault: mean=0.05 → 5% of rollouts had at-fault collisions
dist_to_gt_trajectory: mean=2.3 → Average 2.3m deviation from GT path
duration_frac_20s: mean=0.95 → Average 95% of 20s completed
```

Videos in `aggregate/videos/violations/` are organized by failure type for easy review of problematic scenarios.

### How do I view performance metrics?

#### Metrics Plot (Automatically Generated)

After each simulation run, Alpasim automatically generates a comprehensive performance visualization:

**Location**: `runs/{RUN_DIR}/metrics/metrics_plot.png`

This 3×3 grid plot includes:

**Row 1: RPC Performance**
- RPC Duration histogram - Total time from call start to coroutine resumption
- RPC Blocking histogram - Event loop scheduler delay (time from gRPC I/O completion to coroutine resumption)
- RPC Queue Depth histogram - Service saturation levels

**Row 2: Simulation Timing**
- Rollout Duration histogram - Total time per rollout
- Step Duration histogram - Time per simulation step
- Service Configuration table - Shows replica counts and capacity

**Row 3: Resource Utilization**
- CPU Utilization boxplots - Per-service CPU usage
- GPU Utilization boxplots - GPU compute usage
- GPU Memory boxplots - Memory usage with capacity line

**Summary header** shows:
- Async worker idle percentage - How much time runtime spent idle
- Sim seconds per rollout - Wallclock time per simulation

#### Interpreting the Metrics Plot

**Identifying Bottlenecks**:
- **High queue depth** on a service → Increase replicas_per_container or n_concurrent_rollouts
- **High RPC duration** → Service is slow, consider optimization or scaling
- **Low GPU utilization** (<50%) → Underutilized, can increase load
- **High GPU utilization** (>90%) → May be saturated, check for throttling
- **Unbalanced service config** → Total capacity should match across all services

**Performance Indicators**:
- **Low idle percentage** (<20%) → Runtime is busy, good utilization
- **High idle percentage** (>80%) → Lots of waiting, check for bottlenecks
- **Consistent rollout times** → Good stability
- **Wide rollout time variance** → Investigate outliers in logs

## Simulation Configuration

### How do I enable/disable specific services?

Use `runtime.endpoints.<service>.skip` to disable services:

```bash
# Disable traffic simulation
uv run alpasim_wizard +deploy=local_oss \
    wizard.log_dir=runs/{DATETIME} \
    runtime.endpoints.trafficsim.skip=true

# Disable physics (log replay mode)
uv run alpasim_wizard +deploy=local_oss \
    wizard.log_dir=runs/{DATETIME} \
    runtime.endpoints.physics.skip=true \
    runtime.default_scenario_parameters.physics_update_mode=NONE \
    runtime.default_scenario_parameters.force_gt_duration_us=20000000
```
