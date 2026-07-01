# Performance telemetry

Every AlpaSim run starts Prometheus telemetry by default. No configuration is needed for the basic
setup. The wizard allocates ports, starts one Prometheus support service, starts runtime worker
`/metrics` endpoints, and writes the scrape configuration into the run directory.

At the end of the simulation, the runtime queries the Prometheus server and generates
`metrics_plot.png`.

The Prometheus data persists under `prometheus/data` and can be read by restarting a Prometheus
server later.

## Central Prometheus discovery

By default, AlpaSim publishes Prometheus file-SD targets under
`${defines.filesystem}/prometheus/file-sd`. For `deploy=local`, this resolves under the repo data
directory. For `deploy=iad`, it resolves under the shared IAD Lustre filesystem.

The wizard publishes one central discovery file:

```text
/shared/prometheus/alpasim/<run_uuid>.json
```

A central Prometheus can discover active AlpaSim runs with:

```yaml
scrape_configs:
  - job_name: alpasim
    file_sd_configs:
      - files:
          - /shared/prometheus/alpasim/*.json
        refresh_interval: 10s
```

For normal Docker Compose and Slurm runs, the wizard removes this file when the deployment exits.
If a run crashes before cleanup, later AlpaSim startups conservatively remove old discovery files
only when they are at least five hours old and all listed targets are unreachable.

To start a local Prometheus and Grafana against a local or mounted file-SD directory, run:

```bash
src/tools/scripts/start-prometheus-grafana.sh /shared/prometheus/alpasim
```

The file-SD directory can also be an SSH path. The script mounts it locally with `sshfs`:

```bash
src/tools/scripts/start-prometheus-grafana.sh \
  <iad-login>:/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/data/av_alpamayo_sim/.cache/prometheus/file-sd
```

The script prints the Prometheus and Grafana URLs.
