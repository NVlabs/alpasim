# Controller Benchmark Suite

Benchmark suite for measuring and comparing controller performance.

## Table of Contents

- [Controller Benchmark Suite](#controller-benchmark-suite)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Commands](#commands)
    - [`run` - Run Benchmark Suite](#run---run-benchmark-suite)
    - [`compare` - Compare Two Runs](#compare---compare-two-runs)
    - [`list` - List Available Trajectories](#list---list-available-trajectories)
    - [`select-egomotion` - Select a K-Disk Ego-Motion Cover](#select-egomotion---select-a-k-disk-ego-motion-cover)
  - [Real-Data Ego-Motion Catalogs](#real-data-ego-motion-catalogs)
  - [Trajectory Types](#trajectory-types)
  - [Output Format](#output-format)
  - [Comparison Report](#comparison-report)

## Quick Start

```bash
# From src/controller directory:
cd src/controller

# Run quick benchmark (14 trajectories, ~2-3 minutes) with default ControllerConfig
uv run python -m benchmark run --quick -o results/baseline.json -d "Baseline MPC"

# Run full benchmark (120 trajectories, ~20-30 minutes)
uv run python -m benchmark run -o results/baseline.json -d "Baseline MPC"

# Run benchmark with a custom controller config
uv run python -m benchmark run --quick --config path/to/controller.yaml \
    -o results/optimized.json -d "Custom MPC"

# Run a real-data ego-motion benchmark selected with k-disk coverage
uv run python -m benchmark run --trajectory-catalog results/egomotion_k200.json \
    --config path/to/controller.yaml -o results/sqp_egomotion_k200.json

# Compare two runs
uv run python -m benchmark compare results/baseline.json results/optimized.json \
    -o comparison_report/
```

The `--config` YAML is parsed directly against the `ControllerConfig` dataclass in
`alpasim_controller/mpc_controller.py`. Any subset of its fields can be set; unset
fields fall back to the dataclass defaults. A minimal override looks like:

```yaml
mpc_implementation: nonlinear
n_horizon: 15
```

## Commands

### `run` - Run Benchmark Suite

```bash
uv run python -m benchmark run [OPTIONS]

Options:
  --quick, -q              Run 14 trajectories instead of 120 (faster iteration)
  --output, -o FILE        Output JSON file path
  --description, -d        Description of this benchmark run
  --config FILE            Path to controller YAML config (default: ControllerConfig defaults)
  --trajectory-catalog FILE
                           Ego-motion catalog JSON; selected records override synthetic trajectories
```

When `--trajectory-catalog` is set, each catalog record runs for its full ego-motion duration.
The benchmark remains receding horizon: each controller tick receives the normal finite planning
horizon sliced from the full reference trajectory.

### `compare` - Compare Two Runs

```bash
uv run python -m benchmark compare BASELINE COMPARISON [OPTIONS]

Arguments:
  BASELINE             Path to baseline results JSON
  COMPARISON           Path to comparison results JSON

Options:
  --output-dir, -o     Directory for comparison report (plots + summary)
  --interactive, -i    Show plots interactively instead of saving
  --baseline-name      Display name for baseline
  --comparison-name    Display name for comparison
```

### `list` - List Available Trajectories

```bash
uv run python -m benchmark list [--quick] [--trajectory-catalog FILE]
```

### `select-egomotion` - Select a K-Disk Ego-Motion Cover

```bash
uv run python -m benchmark select-egomotion CATALOG --k 200 -o results/egomotion_k200.json
```

The selector computes ego-motion features such as path length, speed range, acceleration, yaw
rate, curvature, and lateral displacement, normalizes them across the catalog, then greedily adds
the sample farthest from the current selected set. The reported covering radius is the maximum
normalized feature distance from any catalog record to the selected set.

## Real-Data Ego-Motion Catalogs

Compile a ClipGT ego-motion-only catalog from the full dataset with:

```bash
uv run python tools/compile_ego_motion_catalog.py \
    --metadb-path /path/to/clipgt_metadb.parquet \
    --url-prefix /path/to/clipgt/url/root \
    --output results/egomotion_full.json \
    --selected-output results/egomotion_k200.json \
    --k 200
```

The compiler decodes only egomotion (`ego_history_xyz`, `ego_future_xyz`, rotations, and
timestamps). It uses a dummy video cache and does not decode camera images. By default it requests
300 future steps at 0.1 s, producing 30 s records; tune `--num-future-steps` and `--time-step` if
the dataset slice uses a different duration.

## Trajectory Types

The benchmark includes 120 reference trajectories covering:

- **Straight driving** (24 trajectories): 1-25 m/s
- **Left turns** (20 trajectories): radius 20-500m
- **Right turns** (20 trajectories): radius 20-500m
- **Acceleration** (15 trajectories): various speed ramps
- **Deceleration** (15 trajectories): various speed ramps
- **Combined maneuvers** (10 trajectories): turns with speed changes
- **Stop-then-go with turn** (8 trajectories): start from stop, accelerate into turn
- **Turn-to-stop** (8 trajectories): decelerate from turn to stop

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "description": "Baseline MPC",
  "summary": {
    "n_simulations": 100,
    "total_iterations": 10000,
    "total_time_s": 1234.5,
    "solve_time_mean_ms": 45.2,
    "solve_time_median_ms": 42.1,
    "solve_time_min_ms": 30.5,
    "solve_time_max_ms": 120.3,
    "tracking_rmse_m": 0.42,
    "tracking_max_error_m": 1.8,
    "control_aggressiveness_score": 0.35,
    "total_control_violation_count": 0
  },
  "simulations": [
    {
      "trajectory_name": "straight_v10_0",
      "trajectory_description": "Straight at 10 m/s",
      "total_time_ms": 12345.6,
      "control_summary": {
        "tracking_rmse_m": 0.42,
        "steering_cmd_abs_max_rad": 0.12,
        "steering_cmd_abs_mean_rad": 0.03,
        "acceleration_cmd_abs_max_mps2": 0.3,
        "acceleration_cmd_abs_mean_mps2": 0.1,
        "acceleration_cmd_min_mps2": -0.2,
        "acceleration_cmd_max_mps2": 0.3,
        "jerk_cmd_abs_max_mps3": 2.5,
        "jerk_cmd_abs_mean_mps3": 0.7,
        "curvature_abs_max_1pm": 0.01,
        "curvature_abs_mean_1pm": 0.003,
        "control_aggressiveness_score": 0.08,
        "total_violation_count": 0
      },
      "iterations": [
        {
          "timestamp_us": 0,
          "solve_time_ms": 45.2,
          "x": 0.0,
          "y": 0.0,
          "yaw": 0.0,
          "vx": 10.0,
          "ref_x": 0.0,
          "ref_y": 0.0,
          "ref_yaw": 0.0,
          "steering_cmd": 0.0,
          "acceleration_cmd": 0.0,
          "jerk_cmd": 0.0,
          "curvature": 0.0,
          "curvature_rate": 0.0
        }
      ]
    }
  ]
}
```

Control summaries report both command bounds and smoothness/aggressiveness metrics. The default
benchmark bounds are:

- steering command: `abs(u_steer) <= 2.0 rad`
- acceleration command: `-9.0 <= u_accel <= 6.0 m/s²`
- jerk command: `abs(du_accel/dt) <= 20.0 m/s³`
- realized curvature: `abs(yaw_rate / vx) <= 0.35 1/m`
- curvature rate: `abs(dcurvature/dt) <= 2.0 1/(m*s)`

The `control_aggressiveness_score` is the RMS of normalized steering, acceleration, jerk,
curvature, and curvature-rate usage. Lower is smoother/less aggressive; use it alongside tracking
error rather than as a replacement.

## Comparison Report

The comparison report includes:

- **summary.txt**: Text summary with timing comparisons
- **tracking/control summary**: RMSE, maximum error, control effort, jerk, curvature, and bound
  violations
- **timing_comparison.png**: Timing histograms and bar charts
- **trajectories/**: Per-trajectory plots showing:
  - X-Y path (bird's eye view)
  - X position vs time
  - Y position vs time
  - Solve time vs simulation time
