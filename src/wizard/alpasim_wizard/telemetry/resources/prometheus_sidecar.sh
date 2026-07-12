set -euo pipefail

START_PROMETHEUS={start_prometheus}
PROMETHEUS_BIN=
if [[ "$START_PROMETHEUS" == "true" ]]; then
  if command -v prometheus >/dev/null 2>&1; then
    PROMETHEUS_BIN=prometheus
  else
    echo "prometheus binary not found" >&2
    exit 1
  fi
fi

if command -v node_exporter >/dev/null 2>&1; then
  NODE_EXPORTER_BIN=node_exporter
elif command -v prometheus-node-exporter >/dev/null 2>&1; then
  NODE_EXPORTER_BIN=prometheus-node-exporter
else
  echo "node_exporter binary not found" >&2
  exit 1
fi

if command -v process-exporter >/dev/null 2>&1; then
  PROCESS_EXPORTER_BIN=process-exporter
elif command -v prometheus-process-exporter >/dev/null 2>&1; then
  PROCESS_EXPORTER_BIN=prometheus-process-exporter
else
  PROCESS_EXPORTER_BIN=
fi

start_slurm_process_exporter() {
  uv run --no-sync --project /repo/src/wizard \
    python -m alpasim_wizard.telemetry.slurm_process_exporter \
    --port="{prometheus_ports.process_exporter}" \
    --procfs=/host/proc \
    --cgroupfs=/host/sys/fs/cgroup &
  PROCESS_PID=$!
}

# hwmon/nvme sysfs reads issue live NVMe admin commands; a wedged controller
# blocks them in uninterruptible sleep and kills the whole /metrics endpoint.
$NODE_EXPORTER_BIN \
  --web.listen-address=0.0.0.0:{prometheus_ports.node_exporter} \
  --path.procfs=/host/proc \
  --path.sysfs=/host/sys \
  --path.rootfs=/rootfs \
  --no-collector.systemd \
  --no-collector.hwmon \
  --no-collector.nvme &
NODE_PID=$!

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  start_slurm_process_exporter
else
  if [[ -z "${PROCESS_EXPORTER_BIN}" ]]; then
    echo "process-exporter binary not found" >&2
    exit 1
  fi
  $PROCESS_EXPORTER_BIN \
    --web.listen-address=0.0.0.0:{prometheus_ports.process_exporter} \
    --procfs=/host/proc \
    --config.path=/mnt/log_dir/prometheus/process-exporter.yml &
  PROCESS_PID=$!
fi

if command -v dcgm-exporter >/dev/null 2>&1; then
  dcgm-exporter -f /mnt/log_dir/prometheus/dcgm-counters.csv -a :{prometheus_ports.dcgm_exporter} &
  DCGM_PID=$!
else
  echo "dcgm-exporter binary not found; GPU exporter disabled" >&2
  DCGM_PID=
fi

trap 'kill "$NODE_PID" "$PROCESS_PID" "$DCGM_PID" 2>/dev/null || true' TERM INT

if [[ "$START_PROMETHEUS" == "true" ]]; then
  exec $PROMETHEUS_BIN \
    --config.file=/mnt/log_dir/prometheus/prometheus.yml \
    --storage.tsdb.path=/mnt/log_dir/prometheus/data \
    --enable-feature=promql-at-modifier \
    --web.listen-address=0.0.0.0:{prometheus_ports.prometheus}
fi

wait
