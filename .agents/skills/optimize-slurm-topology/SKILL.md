---
name: optimize-slurm-topology
description: Optimize AlpaSim Slurm topology throughput using persistent local Prometheus/Grafana telemetry and run artifacts. Use when tuning service GPU placement, replicas_per_container, runtime.nr_workers, endpoint n_concurrent_rollouts, NRE/physics cache sizes, or Slurm experiment batches for full-duration rollout throughput.
---

# Optimize Slurm Topology

Use this workflow to iteratively improve AlpaSim rollout throughput on Slurm. Optimize for full 20s rollout throughput, not startup-only behavior.

## Inputs

When this skill is invoked, the user must specify a full run command as starting
point for the optimization. This command provides a base topology as starting
point, as well as the target configuration (e.g. driver, sceneset, n_rollouts,
cluster, and any other parameters such as number of cameras, simulation
frequency, etc.).

If the user doesn't specify a full run command, ask them!

## Telemetry and Memory Setup

Use one persistent local Prometheus/Grafana instance for the whole experiment.

1. Start local telemetry once with
   `src/tools/scripts/start-prometheus-grafana.sh <file-sd-dir-or-ssh-path>
   --grafana-port 3003 --prometheus-port 9093`. Use the non-default ports to
   avoid conflicts with user-started telemetry stacks.
2. The `<file-sd-dir-or-ssh-path>` argument is provided by the experiment logs.
   For example, the default value on IAD is
   `<iad-ssh-alias>:/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/data/av_alpamayo_sim/.cache/prometheus/file-sd`
3. Keep the local telemetry stack running until all experiment candidates have
   been evaluated. Then stop it with `src/tools/scripts/start-prometheus-grafana.sh stop`
4. Create a repo-local experiment record from `references/experiment-record.md`.
   Default path:
   `docs/experiments/topology-opt-<driver>-<cluster>-<YYYYMMDD>.md`. This file will be your experiment log and memory. It should contain all necessary information to understand why a topology was tried and what the results were.

## Experiment Loop

0. Start from a known topology and run a baseline experiment.
1. If you already have a baseline, start one or multiple candidate experiments
   in parallel (at most 3).
2. The experiments have an initial startup time of a couple of about 5 min
   before they appear in Prometheus. After that, they start producing rollouts.
   However, because all rollouts are initially started simultenously, there's
   significant congestion in the first 10-15 minutes. Wait until you can see
   this congestion has cleared and the system has reached a steady state. Use
   the 5m `seconds_per_rollout` only as an early diagnostic. Once the full-run
   `seconds_per_rollout` has stabilized, typically after 30-45 minutes, use it
   as the primary optimization target.
3. Reject candidates that OOM or crash. Analyze the reason for failure and avoid
   repeating the same mistake. A typical reason is insufficient GPU memory.
4. Once a candidate reaches steady state, analyze it carefully and document (see
   `references/metrics.md` and `references/topology-knobs.md`):
  * Its stabilized full-run `seconds_per_rollout`, using the 5m value only to
    diagnose recent behavior and confirm that the run remains healthy.
  * Its bottlenecks, using
    `alpasim:rpc_queue_depth_at_start_latest:max` as the primary bottleneck
    signal and `alpasim:rpc_queue_depth_at_start_latest:min` to detect workers
    that are starving or receiving uneven load.
  * Its used and available resources, including per-GPU utilization, memory
    consumption, memory pressure, and memory headroom.
  * Opportunities for improvement.
5. Skill improvement reflections: Did you learn something new about the system
   that was not yet covered in the skill? This can include, for example:
  * How to run experiments or query results.
  * Which Prometheus queries are useful.
  * How the topology knobs affect throughput and memory.
  * Are there additional metrics that we should introduce to better understand
    the system?
  * Any additional scripts that you wrote to help with the experiments that
    would be useful to add to the skill.
  * Or anything else that you think is useful to remember for future
    experiments.
5. Keep two record sections current (see `references/experiment-record.md`).
   These records should include the output of both step 4 and step 5, and should
   be updated after every candidate experiment:
   - a short Markdown progress document (including table) for humans and quick
     parsing;
   - a more detailed JSON memory block with fixed inputs, runs, metrics, GPU
     utilization, GPU memory consumption and headroom, decisions, and links to
     run artifacts.
6. Decide on the next topology changes and go back to step 1.
7. You should stop running experiments as soon as you have enough data to
   support your reasoning and decision. It is not required to let them run until
   the end. However, do not compare or stop a healthy candidate before its
   full-run `seconds_per_rollout` has stabilized, normally 30-45 minutes after
   rollout production begins. Note that "steady state" can still contain cyclic
   behavior.
8. Stop when you can't make progress over multiple iterations or when you don't
   believe there is more enough free resources to improve throughput.

## Supporting documents

* Read `references/experiment-record.md` for how to keep a record of the experiment and its candidates.
* Read `references/metrics.md` for current metric names, PromQL queries, and interpretation.
* Read `references/topology-knobs.md` for guidelines on how to change topology and what to expect from each change.

## Final Reporting

At the end of the optimization, re-read the experiment record and summarize the
results in a DETAILED final report, including:

1. Baseline topology and initial speed (primary stabilized full-run
   `seconds_per_rollout`, plus 5m `seconds_per_rollout` for recent-behavior
   context), including per-GPU utilization and memory consumption/headroom.
2. All tried topologies, their reasoning, expected effect, measured result,
   resource usage, and decision.
3. Best topology found, why it won, remaining bottlenecks or constraints, and
   how much GPU utilization and memory headroom remains for further tuning.
4. Any advice on how the skill or the instructions can be improved.
5. Link the repo-local experiment record.
