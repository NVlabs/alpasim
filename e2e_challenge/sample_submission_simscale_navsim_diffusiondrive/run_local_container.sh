#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-simscale-diffusiondrive:latest}"
BASE_PORT="${ALPASIM_DRIVER_BASE_PORT:-${ALPASIM_DRIVER_PORT:-6789}}"
CONTAINER_PORT="${ALPASIM_DRIVER_CONTAINER_PORT:-6789}"
REPLICAS="${ALPASIM_DRIVER_REPLICAS:-1}"
GPUS="${ALPASIM_DOCKER_GPUS:-all}"
DETACH="${ALPASIM_DRIVER_DETACH:-0}"
PREFIX="${ALPASIM_DRIVER_NAME_PREFIX:-alpasim-e2e-simscale-diffusiondrive}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

decimal_in_range() {
  local name="$1"
  local value="$2"
  local minimum="$3"
  local maximum="$4"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a decimal integer"
  local numeric=$((10#${value}))
  ((numeric >= minimum && numeric <= maximum)) || \
    fail "${name} must be between ${minimum} and ${maximum}"
}

decimal_in_range ALPASIM_DRIVER_REPLICAS "${REPLICAS}" 1 1024
decimal_in_range ALPASIM_DRIVER_BASE_PORT "${BASE_PORT}" 1 65535
decimal_in_range ALPASIM_DRIVER_CONTAINER_PORT "${CONTAINER_PORT}" 1 65535
[[ "${DETACH}" == "0" || "${DETACH}" == "1" ]] || \
  fail "ALPASIM_DRIVER_DETACH must be 0 or 1"
[[ "${PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
  fail "ALPASIM_DRIVER_NAME_PREFIX must match ^[A-Za-z0-9][A-Za-z0-9_.-]*$"

REPLICAS_NUM=$((10#${REPLICAS}))
BASE_PORT_NUM=$((10#${BASE_PORT}))
CONTAINER_PORT_NUM=$((10#${CONTAINER_PORT}))
((BASE_PORT_NUM + REPLICAS_NUM - 1 <= 65535)) || \
  fail "replica host ports exceed 65535"

container_name() {
  printf '%s-%s' "${PREFIX}" "$1"
}

docker_args() {
  local idx="$1"
  local host_port="$2"
  local detached="$3"
  local name
  name="$(container_name "${idx}")"
  args=(
    docker run --rm --init --name "${name}"
    --cap-drop ALL
    --security-opt no-new-privileges:true
    --read-only
    --pids-limit 1024
    --memory 32g
    --cpus 8
    --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777
    --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755
    -p "127.0.0.1:${host_port}:${CONTAINER_PORT_NUM}"
    -e ALPASIM_DRIVER_HOST=0.0.0.0
    -e "ALPASIM_DRIVER_PORT=${CONTAINER_PORT_NUM}"
    -e "ALPASIM_CONTESTANT_REPLICA_INDEX=${idx}"
    -e "ALPASIM_CONTESTANT_REPLICAS=${REPLICAS_NUM}"
    -e "ALPASIM_DRIVER_GRPC_WORKERS=${ALPASIM_DRIVER_GRPC_WORKERS:-4}"
    -e "ALPASIM_DRIVER_LOG_LEVEL=${ALPASIM_DRIVER_LOG_LEVEL:-INFO}"
    -e "DIFFUSIONDRIVE_DEVICE=${DIFFUSIONDRIVE_DEVICE:-cuda}"
    -e "DIFFUSIONDRIVE_MAX_BATCH_SIZE=${DIFFUSIONDRIVE_MAX_BATCH_SIZE:-1}"
    -e "DIFFUSIONDRIVE_BATCH_WINDOW_MS=${DIFFUSIONDRIVE_BATCH_WINDOW_MS:-2}"
  )
  if [[ "${detached}" == "1" ]]; then
    args+=(--detach)
  fi
  if [[ -n "${GPUS}" && "${GPUS}" != "none" ]]; then
    args+=(--gpus "${GPUS}")
  fi
  args+=("${IMAGE}")
}

if ((REPLICAS_NUM == 1)); then
  name="$(container_name 0)"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_args 0 "${BASE_PORT_NUM}" "${DETACH}"
  if [[ "${DETACH}" == "1" ]]; then
    "${args[@]}"
    echo "${name}: 127.0.0.1:${BASE_PORT_NUM}->${CONTAINER_PORT_NUM}"
    exit 0
  fi
  exec "${args[@]}"
fi

names=()
cleanup_enabled=1
cleanup() {
  if ((cleanup_enabled == 1 && ${#names[@]} > 0)); then
    docker rm -f "${names[@]}" >/dev/null 2>&1 || true
  fi
}
handle_signal() {
  local status="$1"
  cleanup
  cleanup_enabled=0
  trap - EXIT
  exit "${status}"
}
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

for ((idx = 0; idx < REPLICAS_NUM; idx++)); do
  host_port=$((BASE_PORT_NUM + idx))
  name="$(container_name "${idx}")"
  names+=("${name}")
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_args "${idx}" "${host_port}" 1
  "${args[@]}" >/dev/null
  echo "${name}: 127.0.0.1:${host_port}->${CONTAINER_PORT_NUM}"
done

if [[ "${DETACH}" == "1" ]]; then
  cleanup_enabled=0
  exit 0
fi

echo "Press Ctrl-C to stop local driver replicas."
while true; do
  sleep 60
done
