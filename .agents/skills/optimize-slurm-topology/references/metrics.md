# Metrics Reference

## Source Of Truth

1. Raw runtime metrics are defined in
   `src/runtime/alpasim_runtime/telemetry/telemetry_context.py`.
2. Slurm process metrics are defined in
   `src/wizard/alpasim_wizard/telemetry/slurm_process_exporter.py`.
3. Prometheus scrape targets and rule installation are generated in
   `src/wizard/alpasim_wizard/telemetry/prometheus.py`.
4. Recording rules live in
   `src/utils/alpasim_utils/telemetry/metrics_plot_recording_rules.yml`.
5. The wizard copies those rules into each run as
   `<run_dir>/prometheus/rules/alpasim-recording-rules.yml`.
6. Prometheus loads copied rules through generated
   `<run_dir>/prometheus/prometheus.yml`.

Prometheus file-SD targets are written to
`<run_dir>/prometheus/targets/alpasim.json`. Central file-SD publication uses
`wizard.telemetry.file_sd_dir` when configured.

## Query Surface

Use labels such as `run_uuid`, `run_name`, `node`, `user`, and
`slurm_job_id` to separate candidates. Scrape jobs are:

1. `alpasim-runtime-worker`: per-worker AlpaSim runtime metrics.
2. `alpasim-node`: node exporter metrics.
3. `alpasim-process`: process CPU and memory metrics.
4. `alpasim-dcgm`: DCGM GPU metrics, when `dcgm-exporter` is available.

## Interpretation Rules

1. Lower stabilized full-run `seconds_per_rollout` is the primary optimization
   target. It normally becomes useful after 30-45 minutes of rollout production.
2. Use the 5m `seconds_per_rollout` only as an early and recent-behavior
   diagnostic. Compare candidates only after startup congestion clears and the
   full-run metric stops materially drifting. Confirm stabilization across
   multiple observations; elapsed time alone is insufficient.
3. High `alpasim:rpc_queue_depth_at_start_latest:max` identifies the bottleneck
   and the service to scale or isolate. This gauge is set directly to the
   observed depth and does not saturate at 50.
4. Low `alpasim:rpc_queue_depth_at_start_latest:min` means at least one worker
   may be starving. Low min and high max might indicate some load imbalance (either across workers or time). Low min and only slighlty higher max indicates that the overall capacity should be increased to avoid service starvation.
5. High RPC blocking p95 and low runtime idle points to runtime/event-loop contention.
6. GPU memory above 90% is a hard constraint signal.
7. Low GPU utilization with memory headroom suggests more load or co-location.
8. High per-process CPU means process-level scaling can help even if node CPU average looks fine.

## How to find the bottleneck

Use the primary throughput metric to compare candidates. Use supporting signals
(see `topology-knobs.md`), e.g. latest queue depth, RPC p95, runtime idle, CPU,
GPU util, and GPU memory only to explain the bottleneck and choose the next
topology change.

## Prometheus-Attached Labels

The wizard attaches these labels to every scrape target in file-SD:

| Label | Value | Meaning |
|---|---|---|
| `run_uuid` | `run_metadata["run_uuid"]` | Stable identity for one run. |
| `run_name` | `run_metadata["run_name"]` | Human-readable run name. |
| `user` | `$USER` or `unknownUser` | User who launched the run. |
| `node` | `socket.gethostname()` | Node that generated the telemetry config. |
| `slurm_job_id` | `cfg.wizard.slurm_job_id` or empty string | Slurm job ID when running under Slurm. |
| `job` | one of the scrape jobs above | Target type: runtime worker, node, process, or DCGM. |

Prometheus also attaches `instance=<host>:<port>` from the scrape target. The
metric name is available as pseudo-label `__name__` in queries. Histogram bucket
series also include `le`; histogram `_sum` and `_count` series do not.

## Raw AlpaSim Metrics

Histogram metrics expose `_bucket`, `_sum`, and `_count` series. The labels in
this table are metric-emitted labels; raw scraped series also carry the
Prometheus-attached labels above. Use the recording rules below instead of raw
histogram math unless an ad-hoc query needs the raw distribution.

| Metric family | Type | Labels | Measures | Interpret |
|---|---|---|---|---|
| `alpasim_rpc_duration_seconds` | histogram | `service`, `method`, `tag`, `worker_id` | End-to-end RPC call duration. | High p95 means a service method is slow or queued internally. Check with queue depth and blocking. |
| `alpasim_rpc_blocking_seconds` | histogram | `service`, `method`, `tag`, `worker_id` | Time between gRPC I/O completion and coroutine resumption. | High values mean the worker event loop is not resuming promptly, often due Python/runtime contention. |
| `alpasim_rpc_queue_depth_at_start_latest` | gauge | `service`, `tag`, `worker_id` | Latest sampled in-flight RPC count when an RPC starts. | Primary queue signal for bottleneck and capacity decisions. It is set to the exact observed depth and has no bucket ceiling. A worker's value persists until its next RPC sample. |
| `alpasim_rollout_duration_seconds` | histogram | `worker_id` | Full rollout wall time. | High p95 shows tail rollout latency; compare across workers for imbalance. |
| `alpasim_step_duration_seconds` | histogram | `worker_id` | Per-step wall time. | High p95 shows step-level latency.|
| `alpasim_simulation_total_seconds` | gauge | `worker_id` | Total elapsed simulation time accumulated by a worker. | Input for 5m and full-run seconds per rollout. |
| `alpasim_simulation_rollout_count` | gauge | `worker_id` | Completed rollout count. | Used as the 5m denominator. Must increase before judging steady state. |
| `alpasim_simulation_seconds_per_rollout` | gauge | `worker_id` | Full-run average seconds per completed rollout. | Primary optimization input once its run-level average has stabilized. Includes startup/congestion, so allow 30-45 minutes and verify that drift has subsided. |
| `alpasim_event_loop_idle_seconds_total` | gauge | `worker_id` | Time the event loop spent waiting for I/O. | Used with poll/work time to estimate runtime idle fraction. High idle usually means workers wait on services. |
| `alpasim_event_loop_poll_seconds_total` | gauge | `worker_id` | Time spent checking non-blocking I/O. | Large values can indicate event-loop overhead. Interpret with idle/work. |
| `alpasim_event_loop_work_seconds_total` | gauge | `worker_id` | Time spent executing Python work. | High work fraction can mean runtime CPU contention. |
| `alpasim_gc_total_duration_seconds` | gauge | `worker_id` | Total time spent in garbage collection. | High values indicate GC pressure may affect runtime progress. |
| `alpasim_gc_max_duration_seconds` | gauge | `worker_id` | Longest observed GC pause. | Large pauses can explain tail latency spikes. |
| `alpasim_gc_collection_count_total` | gauge | `worker_id` | Number of GC collections. | Use with GC duration; count alone is not a bottleneck signal. |

RPC recording rules currently focus on these methods:
`run_controller_and_vehicle`, `drive`, `submit_egomotion_observation`,
`submit_image_observation`, `submit_route`, `ground_intersection`, and
`render_rgb`.

## External Metrics Used

Prometheus scrapes all metrics exposed by node exporter, process exporter, and
DCGM exporter. The repo explicitly uses the following external metric names in
recording rules or dashboards:

Exporter binaries may expose additional raw metrics that are not enumerated in
this repo. For a live run, query Prometheus
`/api/v1/label/__name__/values` or open the exporter `/metrics` endpoint if
those raw exporter internals matter.

| Metric | Source job | Labels | Measures | Interpret |
|---|---|---|---|---|
| `namedprocess_namegroup_cpu_seconds_total` | `alpasim-process` | `groupname` | Cumulative CPU seconds per process group. | Rate converts to CPU utilization percent. High per-group CPU can motivate more replicas, lower concurrency, or more CPU. |
| `namedprocess_namegroup_memory_bytes` | `alpasim-process` | `groupname`, `memtype` | Process memory by process group. | Use resident memory to identify host memory pressure or unexpected process growth. |
| `alpasim_slurm_process_exporter_scrape_duration_seconds` | `alpasim-process` | none | Time spent collecting Slurm process metrics. | High values mean process metrics may be stale or expensive to collect. |
| `DCGM_FI_DEV_GPU_UTIL` | `alpasim-dcgm` | `gpu` | GPU utilization percent by GPU. | Low utilization with memory headroom suggests more load or co-location; high utilization plus queues suggests GPU capacity pressure. |
| `DCGM_FI_DEV_FB_TOTAL` | `alpasim-dcgm` | `gpu` | Total physical GPU framebuffer memory in MiB. | Static GPU capacity; use this instead of summing independently sampled used, free, and reserved gauges. |
| `DCGM_FI_DEV_FB_USED` | `alpasim-dcgm` | `gpu` | GPU framebuffer memory used in MiB. | Above 90% is a hard constraint signal. Reduce cache, concurrency, replicas, or co-location. |
| `DCGM_FI_DEV_FB_FREE` | `alpasim-dcgm` | `gpu` | Free GPU framebuffer memory in MiB. | Use directly as allocatable memory headroom; add used and reserved memory to derive physical total memory. |
| `DCGM_FI_DEV_FB_RESERVED` | `alpasim-dcgm` | `gpu` | Driver-reserved GPU framebuffer memory in MiB. | Add to used and free memory to derive total physical framebuffer memory. |
| `node_cpu_seconds_total` | `alpasim-node` | `mode` | Node CPU seconds by mode. | Dashboard converts idle rate to node CPU utilization. If node CPU is saturated, service scaling may not help. |
| `node_memory_MemAvailable_bytes` | `alpasim-node` | none | Available host memory. | Low available memory indicates node memory pressure. |
| `node_memory_MemTotal_bytes` | `alpasim-node` | none | Total host memory. | Used with available memory to compute host memory utilization. |

Process groups are configured as `runtime`, `driver`, `renderer`, `physics`,
`trafficsim`, and `controller`.

The labels in the external metrics table are exporter-emitted labels. Scraped
series also carry the Prometheus-attached labels above.

## Recording Rules

All rules are in group `alpasim_metrics_plot` in
`src/utils/alpasim_utils/telemetry/metrics_plot_recording_rules.yml`.

| Rule | Output labels | Expression summary | Measures | Interpret |
|---|---|---|---|---|
| `alpasim:rpc_duration_seconds_bucket:sum` | `run_uuid`, `run_name`, `method`, `le` | Sum raw RPC duration buckets by run, method, and bucket. | Aggregated RPC duration histogram. | Base input for RPC duration p95; use when inspecting distributions. |
| `alpasim:rpc_blocking_seconds_bucket:sum` | `run_uuid`, `run_name`, `method`, `le` | Sum raw RPC blocking buckets by run, method, and bucket. | Aggregated event-loop blocking histogram after RPC I/O completion. | Base input for blocking p95; high tails point to runtime scheduling contention. |
| `alpasim:rpc_queue_depth_at_start_latest:max` | `run_uuid`, `run_name`, `service` | Max current latest queue-depth-at-start gauge across worker series by run and service. | Exact latest queue-depth samples, aggregated across workers. | Primary bottleneck locator for service scaling or isolation. It has no 50-value ceiling, but is not a historical window maximum. |
| `alpasim:rpc_queue_depth_at_start_latest:min` | `run_uuid`, `run_name`, `service` | Min current latest queue-depth-at-start gauge across worker series by run and service. | Lowest exact latest queue-depth sample across workers. | Worker starvation and load-imbalance signal. A persistent low min with high max means workers are fed unevenly. |
| `alpasim:rollout_duration_seconds_bucket:sum` | `run_uuid`, `run_name`, `le` | Sum raw rollout duration buckets by run and bucket. | Aggregated rollout duration histogram. | Base input for rollout p95; use to inspect tail shape. |
| `alpasim:step_duration_seconds_bucket:sum` | `run_uuid`, `run_name`, `le` | Sum raw step duration buckets by run and bucket. | Aggregated step duration histogram. | Base input for step p95; use when per-step latency dominates. |
| `alpasim:driver_drive_rpc_duration_seconds_bucket:sum` | `run_uuid`, `run_name`, `le` | Sum raw driver `drive` duration buckets by run and bucket. | Driver `drive` latency distribution. | Base input for driver latency distribution panels. |
| `alpasim:driver_drive_rpc_duration_seconds_bucket:rate1m` | `run_uuid`, `run_name`, `le` | 1m rate of raw driver `drive` duration buckets by run and bucket. | Moving driver `drive` latency distribution. | Use in Grafana heatmaps to see the current driver tail shape. |
| `alpasim:nre_render_rpc_duration_seconds_bucket:sum` | `run_uuid`, `run_name`, `method`, `le` | Sum raw sensorsim render duration buckets for `render_rgb`, `batch_render_rgb`, and `render_aggregated`. | NRE render latency distribution by method. | Base input for NRE latency distribution panels. |
| `alpasim:nre_render_rpc_duration_seconds_bucket:rate1m` | `run_uuid`, `run_name`, `method`, `le` | 1m rate of sensorsim render duration buckets for `render_rgb`, `batch_render_rgb`, and `render_aggregated`. | Moving NRE render latency distribution by method. | Use in Grafana heatmaps; select one method at a time with `nre_render_method`. |
| `alpasim:rpc_duration_seconds:p95` | `run_uuid`, `run_name`, `method` | p95 over `rate(alpasim:rpc_duration_seconds_bucket:sum[1m])` by method. | RPC method tail latency. | High method p95 confirms a slow service call. Pair with queue depth. |
| `alpasim:rpc_blocking_seconds:p95` | `run_uuid`, `run_name`, `method` | p95 over `rate(alpasim:rpc_blocking_seconds_bucket:sum[1m])` by method. | RPC blocking tail latency. | High values indicate runtime event-loop contention rather than only remote service latency. |
| `alpasim:rollout_duration_seconds:p95` | `run_uuid`, `run_name` | p95 over aggregated rollout duration buckets. | Run-level rollout tail latency. | High values mean unstable or slow full rollouts. |
| `alpasim:rollout_duration_seconds:p95_by_worker` | `run_uuid`, `run_name`, `worker_id` | p95 over raw rollout duration buckets by worker. | Per-worker rollout tail latency. | Divergence across workers suggests imbalance, stragglers, or worker-specific faults. |
| `alpasim:step_duration_seconds:p95` | `run_uuid`, `run_name` | p95 over aggregated step duration buckets. | Step tail latency. | Useful when rollout latency is high but queue/RPC evidence is ambiguous. |
| `alpasim:driver_drive_rpc_duration_seconds:p95` | `run_uuid`, `run_name` | p95 over `alpasim:driver_drive_rpc_duration_seconds_bucket:rate1m`. | Driver `drive` tail latency. | Use with driver queue depth and driver heatmap to decide whether driver capacity is the bottleneck. |
| `alpasim:nre_render_rpc_duration_seconds:p95` | `run_uuid`, `run_name`, `method` | p95 over `alpasim:nre_render_rpc_duration_seconds_bucket:rate1m`. | NRE render tail latency by method. | Use with sensorsim queue depth and NRE heatmap to decide whether render capacity/cache is the bottleneck. |
| `alpasim:event_loop_idle_fraction:ratio` | `run_uuid`, `run_name` | idle / (idle + poll + work), summed by run. | Runtime event-loop idle fraction. | Very high idle means workers are waiting on services. Low idle with CPU headroom suggests increasing `runtime.nr_workers` may help. |
| `alpasim:simulation_seconds_per_rollout:rate5m` | `run_uuid`, `run_name` | `increase(alpasim_simulation_total_seconds[5m]) / increase(alpasim_simulation_rollout_count[5m])`, summed by run. | Moving 5m seconds per rollout. | Early and recent-behavior diagnostic. Do not use noisy short-window differences to rank stabilized candidates. |
| `alpasim:simulation_seconds_per_rollout:avg` | `run_uuid`, `run_name` | Average raw worker `alpasim_simulation_seconds_per_rollout` by run. | Full-run average seconds per rollout. | Primary optimization metric after it stops materially drifting, typically after 30-45 minutes of rollout production. Lower is better. |
| `alpasim:driver_drive_rpc_duration_seconds_sum:sum` | `run_uuid`, `run_name` | Sum raw driver `drive` RPC duration sums by run. | Total driver `drive` RPC seconds. | Divide by count for mean driver drive latency. |
| `alpasim:driver_drive_rpc_duration_seconds_count:sum` | `run_uuid`, `run_name` | Sum raw driver `drive` RPC counts by run. | Number of driver `drive` RPC observations. | Use as denominator; low count means weak latency evidence. |
| `alpasim:process_cpu_utilization_percent:rate30s` | `run_uuid`, `run_name`, `groupname` | `100 * rate(namedprocess_namegroup_cpu_seconds_total[30s])` by process group. | Process-group CPU utilization percent. | High group CPU means that process is CPU-bound or needs different replica/concurrency placement. |
| `alpasim:process_cpu_utilization_percent:max_by_group:rate30s` | `run_uuid`, `run_name`, `groupname` | Maximum per-process CPU utilization within each process group. | Hottest process CPU utilization percent. | Use with group totals to identify a saturated process hidden by aggregate CPU. |
| `alpasim:gpu_utilization_percent:avg` | `run_uuid`, `run_name`, `gpu` | Average `DCGM_FI_DEV_GPU_UTIL` by GPU. | GPU utilization percent. | Low utilization with memory headroom suggests underuse; high utilization with queues suggests GPU pressure. |
| `alpasim:gpu_memory_gb:avg` | `run_uuid`, `run_name`, `gpu` | Average `DCGM_FI_DEV_FB_USED / 1024` by GPU. | GPU memory used in GiB. | High memory limits cache/concurrency/co-location changes. |
| `alpasim:gpu_memory_total_gb:avg` | `run_uuid`, `run_name`, `gpu` | Average `DCGM_FI_DEV_FB_TOTAL / 1024` by GPU. | Total physical GPU memory in GiB. | Subtract used and reserved memory to compute available headroom. |
| `alpasim:gpu_memory_pressure_percent:avg` | `run_uuid`, `run_name`, `node`, `gpu` | `100 * used / (used + free)` by GPU. | GPU memory pressure percent. | Above 90% is a hard constraint; report both pressure and remaining GiB. |

## Common Queries

Primary throughput after stabilization:

```promql
alpasim:simulation_seconds_per_rollout:avg
```

Early and recent-behavior diagnostic:

```promql
alpasim:simulation_seconds_per_rollout:rate5m
```

Queue bottleneck and worker starvation by service (exact gauge values with no
50-value ceiling):

```promql
alpasim:rpc_queue_depth_at_start_latest:max
alpasim:rpc_queue_depth_at_start_latest:min
```

Raw fallback when the recording rule is unavailable:

```promql
max by (run_uuid, run_name, service) (
  alpasim_rpc_queue_depth_at_start_latest
)
min by (run_uuid, run_name, service) (
  alpasim_rpc_queue_depth_at_start_latest
)
```

RPC latency by method:

```promql
alpasim:rpc_duration_seconds:p95
```

Driver and NRE latency distributions:

```promql
alpasim:driver_drive_rpc_duration_seconds_bucket:rate1m
alpasim:nre_render_rpc_duration_seconds_bucket:rate1m
```

Driver and NRE tail latency:

```promql
alpasim:driver_drive_rpc_duration_seconds:p95
alpasim:nre_render_rpc_duration_seconds:p95
```

RPC blocking by method:

```promql
alpasim:rpc_blocking_seconds:p95
```

Runtime idle:

```promql
alpasim:event_loop_idle_fraction:ratio
```

Process CPU:

```promql
alpasim:process_cpu_utilization_percent:rate30s
```

GPU utilization and memory:

```promql
alpasim:gpu_utilization_percent:avg
alpasim:gpu_memory_gb:avg
alpasim:gpu_memory_total_gb:avg
alpasim:gpu_memory_pressure_percent:avg
```

Use `min_over_time`, `avg_over_time`, and `max_over_time` over the same stable
window used for throughput. For example:

```promql
min_over_time(alpasim:gpu_utilization_percent:avg[5m])
avg_over_time(alpasim:gpu_utilization_percent:avg[5m])
max_over_time(alpasim:gpu_utilization_percent:avg[5m])

min_over_time(alpasim:gpu_memory_gb:avg[5m])
avg_over_time(alpasim:gpu_memory_gb:avg[5m])
max_over_time(alpasim:gpu_memory_gb:avg[5m])

max_over_time(alpasim:gpu_memory_pressure_percent:avg[5m])
min_over_time(
  (DCGM_FI_DEV_FB_FREE / 1024)[5m:]
)
```
