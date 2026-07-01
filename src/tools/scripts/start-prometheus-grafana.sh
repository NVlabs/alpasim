#!/usr/bin/env bash
set -euo pipefail

readonly PROMETHEUS_IMAGE="prom/prometheus:v2.55.1"
readonly GRAFANA_IMAGE="grafana/grafana:11.5.2"

usage() {
  cat <<'USAGE'
Usage:
  start-prometheus-grafana.sh [start] <file-sd-dir> [options]
  start-prometheus-grafana.sh stop

Arguments:
  file-sd-dir                 Local path or SSH path.
                              Examples:
                                /tmp/alpasim-prometheus/file-sd
                                iad:/lustre/.../prometheus/file-sd
                                user@iad-login:/lustre/.../prometheus/file-sd

Options:
  --prometheus-port PORT      Host port for Prometheus. Default: first open >= 9090
  --grafana-port PORT         Host port for Grafana. Default: first open >= 3000
  -h, --help                  Show this help.
USAGE
}

find_open_port() {
  python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            port += 1
            continue
    print(port)
    break
PY
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

is_remote_path() {
  local path="$1"
  [[ "$path" == *:* && "$path" != /* ]]
}

shell_quote() {
  printf "%q" "$1"
}

ensure_remote_dir() {
  local remote="$1"
  local host="${remote%%:*}"
  local remote_path="${remote#*:}"
  ssh "$host" "mkdir -p -- $(shell_quote "$remote_path")"
}

unmount_sshfs() {
  local mount_dir="$1"
  if [[ ! -d "$mount_dir" ]] || ! mountpoint -q "$mount_dir"; then
    return
  fi

  if command -v fusermount3 >/dev/null 2>&1; then
    fusermount3 -u "$mount_dir"
  elif command -v fusermount >/dev/null 2>&1; then
    fusermount -u "$mount_dir"
  else
    umount "$mount_dir"
  fi
}

mount_remote_file_sd() {
  local remote="$1"
  local mount_dir="$2"

  require_command ssh
  require_command sshfs
  require_command mountpoint

  mkdir -p "$mount_dir"
  ensure_remote_dir "$remote"
  unmount_sshfs "$mount_dir"

  if ! sshfs "$remote" "$mount_dir" \
    -o ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,allow_other; then
    echo "Failed to mount remote file-SD directory with sshfs." >&2
    echo "Docker needs the SSHFS mount to use allow_other so Prometheus can read it." >&2
    echo "Enable it by setting 'user_allow_other' in /etc/fuse.conf, then rerun this script." >&2
    exit 1
  fi
}

make_world_readable() {
  local path
  for path in "$@"; do
    find "$path" -type d -exec chmod 755 {} +
    find "$path" -type f -exec chmod 644 {} +
  done
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

ACTION="start"
if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    start|up)
      ACTION="start"
      shift
      ;;
    stop)
      ACTION="stop"
      shift
      ;;
  esac
fi

FILE_SD_SOURCE=""
PROMETHEUS_PORT=""
GRAFANA_PORT=""
USER_NAME="${USER:-user}"
WORK_DIR="${TMPDIR:-/tmp}/alpasim-local-telemetry-${USER_NAME}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prometheus-port)
      PROMETHEUS_PORT="${2:?Missing value for --prometheus-port}"
      shift 2
      ;;
    --grafana-port)
      GRAFANA_PORT="${2:?Missing value for --grafana-port}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "$FILE_SD_SOURCE" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 1
      fi
      FILE_SD_SOURCE="$1"
      shift
      ;;
  esac
done

COMPOSE_FILE="$WORK_DIR/docker-compose.yaml"
SSHFS_FILE_SD_DIR="$WORK_DIR/file-sd-sshfs"

if [[ "$ACTION" == "stop" ]]; then
  if [[ -f "$COMPOSE_FILE" ]]; then
    require_command docker
    docker compose -f "$COMPOSE_FILE" down
  else
    echo "No local telemetry compose file found at $COMPOSE_FILE" >&2
  fi
  if command -v mountpoint >/dev/null 2>&1; then
    unmount_sshfs "$SSHFS_FILE_SD_DIR"
  fi
  exit 0
fi

if [[ -z "$FILE_SD_SOURCE" ]]; then
  echo "Missing required file-SD directory argument." >&2
  usage >&2
  exit 1
fi

require_command docker
require_command python3
docker compose version >/dev/null

if [[ -z "$PROMETHEUS_PORT" ]]; then
  PROMETHEUS_PORT="$(find_open_port 9090)"
fi
if [[ -z "$GRAFANA_PORT" ]]; then
  GRAFANA_PORT="$(find_open_port 3000)"
fi

PROMETHEUS_DIR="$WORK_DIR/prometheus"
GRAFANA_PROVISIONING_DIR="$WORK_DIR/grafana/provisioning"
GRAFANA_DASHBOARDS_DIR="$WORK_DIR/grafana/dashboards"
GRAFANA_PLUGIN_PROVISIONING_DIR="$GRAFANA_PROVISIONING_DIR/plugins"
GRAFANA_ALERTING_PROVISIONING_DIR="$GRAFANA_PROVISIONING_DIR/alerting"

mkdir -p \
  "$PROMETHEUS_DIR/rules" \
  "$GRAFANA_PROVISIONING_DIR/datasources" \
  "$GRAFANA_PROVISIONING_DIR/dashboards" \
  "$GRAFANA_PLUGIN_PROVISIONING_DIR" \
  "$GRAFANA_ALERTING_PROVISIONING_DIR" \
  "$GRAFANA_DASHBOARDS_DIR"

if is_remote_path "$FILE_SD_SOURCE"; then
  FILE_SD_DIR="$SSHFS_FILE_SD_DIR"
  mount_remote_file_sd "$FILE_SD_SOURCE" "$FILE_SD_DIR"
else
  FILE_SD_DIR="$FILE_SD_SOURCE"
  mkdir -p "$FILE_SD_DIR"
  if command -v mountpoint >/dev/null 2>&1; then
    unmount_sshfs "$SSHFS_FILE_SD_DIR"
  fi
fi

cp "$REPO_ROOT/src/utils/alpasim_utils/telemetry/metrics_plot_recording_rules.yml" \
  "$PROMETHEUS_DIR/rules/alpasim-recording-rules.yml"
cp "$REPO_ROOT/src/utils/alpasim_utils/telemetry/alpasim-runtime-dashboard.json" \
  "$GRAFANA_DASHBOARDS_DIR/alpasim-runtime.json"

cat >"$PROMETHEUS_DIR/prometheus.yml" <<'EOF'
global:
  scrape_interval: 5s
  evaluation_interval: 5s

rule_files:
  - /etc/prometheus/rules/*.yml

scrape_configs:
  - job_name: alpasim
    file_sd_configs:
      - files:
          - file-sd/*.json
        refresh_interval: 5s
EOF

cat >"$GRAFANA_PROVISIONING_DIR/datasources/prometheus.yaml" <<'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    uid: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

cat >"$GRAFANA_PROVISIONING_DIR/dashboards/alpasim.yaml" <<'EOF'
apiVersion: 1
providers:
  - name: AlpaSim
    type: file
    options:
      path: /var/lib/grafana/dashboards
EOF

make_world_readable "$PROMETHEUS_DIR" "$GRAFANA_PROVISIONING_DIR" "$GRAFANA_DASHBOARDS_DIR"

cat >"$COMPOSE_FILE" <<EOF
services:
  prometheus:
    image: ${PROMETHEUS_IMAGE}
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
      - --enable-feature=promql-at-modifier
      - --web.listen-address=0.0.0.0:9090
    ports:
      - "${PROMETHEUS_PORT}:9090"
    volumes:
      - "${PROMETHEUS_DIR}/prometheus.yml:/etc/prometheus/prometheus.yml:ro"
      - "${PROMETHEUS_DIR}/rules:/etc/prometheus/rules:ro"
      - "${FILE_SD_DIR}:/etc/prometheus/file-sd:ro"
      - prometheus-data:/prometheus

  grafana:
    image: ${GRAFANA_IMAGE}
    depends_on:
      - prometheus
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
      - GF_AUTH_DISABLE_LOGIN_FORM=true
      - GF_LOG_LEVEL=warn
    ports:
      - "${GRAFANA_PORT}:3000"
    volumes:
      - "${GRAFANA_PROVISIONING_DIR}:/etc/grafana/provisioning:ro"
      - "${GRAFANA_DASHBOARDS_DIR}:/var/lib/grafana/dashboards:ro"
      - grafana-data:/var/lib/grafana

volumes:
  prometheus-data:
  grafana-data:
EOF

docker compose -f "$COMPOSE_FILE" up -d

cat <<EOF
Prometheus UI: http://localhost:${PROMETHEUS_PORT}
Grafana UI: http://localhost:${GRAFANA_PORT}
File-SD source: ${FILE_SD_SOURCE}
File-SD directory watched by Prometheus: ${FILE_SD_DIR}
State directory: ${WORK_DIR}

Stop with:
  $0 stop
EOF
