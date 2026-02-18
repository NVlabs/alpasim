# Controller Benchmark Suite

Benchmark suite for measuring and comparing controller performance.

## Quick Start

```bash
# From src/controller directory:
cd src/controller

# Run quick benchmark (14 trajectories, ~2-3 minutes)
uv run python -m benchmark run --quick -o results/baseline.json -d "Baseline MPC"

# Run full benchmark (120 trajectories, ~20-30 minutes)
uv run python -m benchmark run -o results/baseline.json -d "Baseline MPC"

# After making changes, run another benchmark
uv run python -m benchmark run --quick -o results/optimized.json -d "Reduced horizon"

# Compare different implementations
uv run python -m benchmark run --quick -m nonlinear -o results/nonlinear.json -d "Nonlinear MPC"
uv run python -m benchmark run --quick -m linear -o results/linear.json -d "Linear MPC (OSQP)"

# Compare the two runs
uv run python -m benchmark compare results/nonlinear.json results/linear.json -o comparison_report/
```

## Commands

### `run` - Run Benchmark Suite

```bash
uv run python -m benchmark run [OPTIONS]

Options:
  --quick, -q              Run 14 trajectories instead of 120 (faster iteration)
  --output, -o FILE        Output JSON file path
  --description, -d        Description of this benchmark run
  --mpc-implementation, -m MPC implementation: "linear" (default) or "nonlinear"
```

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
uv run python -m benchmark list [--quick]
```

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
    "solve_time_max_ms": 120.3
  },
  "simulations": [
    {
      "trajectory_name": "straight_v10_0",
      "trajectory_description": "Straight at 10 m/s",
      "total_time_ms": 12345.6,
      "iterations": [
        {
          "timestamp_us": 0,
          "solve_time_ms": 45.2,
          "x": 0.0,
          "y": 0.0,
          "yaw": 0.0,
          "ref_x": 0.0,
          "ref_y": 0.0,
          "ref_yaw": 0.0
        }
      ]
    }
  ]
}
```

## Comparison Report

The comparison report includes:

- **summary.txt**: Text summary with timing comparisons
- **timing_comparison.png**: Timing histograms and bar charts
- **trajectories/**: Per-trajectory plots showing:
  - X-Y path (bird's eye view)
  - X position vs time
  - Y position vs time
  - Solve time vs simulation time
