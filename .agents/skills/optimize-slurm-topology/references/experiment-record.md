# Experiment Record Template

This file contains templates for both a markdown experiment record and a JSON run memory. The experiment record is intended to be human-readable and editable, while the run memory is intended to be machine-readable and used to resume the optimization process.

Generate these files in the repo for each optimization pass. Default path:

`docs/experiments/topology-opt-<driver>-<cluster>-<YYYYMMDD>.md`
`docs/experiments/topology-opt-<driver>-<cluster>-<YYYYMMDD>.json`

## Json Record

The top-level structure of the JSON record is as follows:

```json
{
  "experiment_id": "topology-opt-<driver>-<cluster>-<YYYYMMDD>",
  "objective": "minimize stabilized full-run seconds_per_rollout",
  "fixed_inputs": {
    "deploy": "<cluster>",
    "driver": "<driver>",
    "base_topology": "topology=<name>",
    "scenes": {
      "scene_ids": null,
      "test_suite_id": "public_2601"
    },
    "simulation_duration_s": 20,
    "rollouts_per_scene": 1,
    "git_commit": "<sha>",
    "remote_checkout": "<path>",
    "telemetry": {
      "prometheus_url": "http://localhost:<port>",
      "grafana_url": "http://localhost:<port>",
      "file_sd_source": "<ssh-or-path>"
    },
    "slurm": {
      "account": "<account>",
      "partition": "<partition>",
      "gpus_per_node": 8,
      "walltime": "04:00:00"
    }
  },
  "current_best": {
    "topology": null,
    "stabilized_seconds_per_rollout_full_run": null,
    "decision_reasoning": null
  },
  "runs": []
}
```

The run entries in the `runs` array have the following structure:

```json
{
  "runs": [
    {
      "name": "<baseline-or-candidate-name>",
      "job_id": "<slurm_job_id>",
      "run_dir": "<remote-run-dir>",
      "topology": "topology=<name>",
      "status": "pending|running|accepted|rejected|failed|stopped",
      "hypothesis": "<why this run exists>",
      "change": {
        "hydra_or_yaml_path": "<old -> new>"
      },
      "stable_metrics": {
        "warmup_excluded_s": null,
        "completed_rollouts": null,
        "seconds_per_rollout_5m": null,
        "seconds_per_rollout_full_run": null,
        "rollout_duration_p95_s": null,
        "step_duration_p95_s": null,
        "rpc_queue_depth_at_start_latest_by_service": {
          "sensorsim": {"worker_min": null, "worker_max": null},
          "driver": {"worker_min": null, "worker_max": null},
          "physics": {"worker_min": null, "worker_max": null},
          "controller": {"worker_min": null, "worker_max": null}
        },
        "rpc_duration_p95_s_by_method": {
          "run_controller_and_vehicle": null,
          "drive": null,
          "submit_egomotion_observation": null,
          "submit_image_observation": null,
          "submit_route": null,
          "ground_intersection": null,
          "render_rgb": null
        },
        "runtime_idle_fraction": null,
        "resource_usage": {
          "process_cpu_utilization_percent_by_group": {
            "<process-group>": {"min": null, "mean": null, "max": null}
          },
          "gpu_by_index": {
            "<gpu-index>": {
              "topology_services": [],
              "utilization_percent": {
                "min": null,
                "mean": null,
                "max": null
              },
              "memory_used_gb": {
                "min": null,
                "mean": null,
                "max": null
              },
              "memory_total_gb": null,
              "memory_pressure_percent": {
                "mean": null,
                "max": null
              },
              "memory_headroom_gb_min": null
            }
          },
          "host_memory": {
            "available_gb_min": null,
            "total_gb": null
          }
        },
        "bottleneck": {
          "service": null,
          "evidence": [],
          "confidence": "low|medium|high"
        },
        "telemetry_quality": {
          "missing_metrics": [],
          "gaps": [],
          "notes": null
        }
      },
      "failures": [],
      "decision": {
        "outcome": "pending|accept|reject",
        "reason": null,
        "next": null
      },
      "skill_reflections": [
        {
          "topic": "metrics|queries|topology_knobs|run_workflow|failure_mode|other",
          "learning": null,
          "suggested_skill_update": null
        }
      ]
    }
  ]
}
```


## Markdown Record

Keep a markdown record of the experiment in the repo for human readability. It
should contain a separate entry for each candidate. This entry should include
important findings and decisions on what to try next and why. Also include some
key metrics, including rollouts_per_second, queue-depth (min and max), GPU
utilization and memory and which services are on this GPU and if there is more
headroom (and why it wasn't used).

At the top of the file, also include a table.

| Step | Job | Topology | Change | Observation | Decision | Next | Skill update |
|---:|---|---|---|---|---|---|---|
| 1 | `<job_id>` | `topology=<base>` | baseline | `<stabilized full-run and recent 5m seconds_per_rollout; latest RPC queue worker min/max; per-GPU utilization, memory pressure/headroom; bottleneck>` | baseline | `<next knob>` | `<new learning or none>` |
