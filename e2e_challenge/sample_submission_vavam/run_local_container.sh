#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-vavam-driver:latest}"
BASE_PORT="${ALPASIM_DRIVER_BASE_PORT:-${ALPASIM_DRIVER_PORT:-6789}}"
CONTAINER_PORT="${ALPASIM_DRIVER_CONTAINER_PORT:-6789}"
REPLICAS="${ALPASIM_DRIVER_REPLICAS:-1}"
GPUS="${ALPASIM_DOCKER_GPUS:-all}"

if (( REPLICAS < 1 )); then
  echo "ALPASIM_DRIVER_REPLICAS must be >= 1" >&2
  exit 2
fi

container_name() {
  printf 'alpasim-e2e-vavam-driver-%s' "$1"
}

docker_args() {
  local idx="$1"
  local host_port="$2"

  args=(
    docker run --rm
    --init
    --cap-drop ALL
    --security-opt no-new-privileges:true
    --read-only
    --pids-limit 1024
    --memory 32g
    --cpus 8
    --tmpfs /tmp:rw,nosuid,nodev,size=2g
    --tmpfs /run:rw,nosuid,nodev,size=64m
    -p "127.0.0.1:${host_port}:${CONTAINER_PORT}"
    -e ALPASIM_DRIVER_HOST=0.0.0.0
    -e "ALPASIM_DRIVER_PORT=${CONTAINER_PORT}"
    -e "ALPASIM_CONTESTANT_REPLICA_INDEX=${idx}"
    -e "ALPASIM_CONTESTANT_REPLICAS=${REPLICAS}"
    -e "ALPASIM_DRIVER_GRPC_WORKERS=${ALPASIM_DRIVER_GRPC_WORKERS:-4}"
    -e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
    -e "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}"
    -e "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}"
    -e "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}"
    -e "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}"
    -e "TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}"
    -e "TORCH_NUM_INTEROP_THREADS=${TORCH_NUM_INTEROP_THREADS:-1}"
  )

  if [[ -n "$GPUS" && "$GPUS" != "none" ]]; then
    args+=(--gpus "$GPUS")
  fi

  args+=("$IMAGE")
}

if (( REPLICAS == 1 )); then
  docker_args 0 "$BASE_PORT"
  exec "${args[@]}"
fi

names=()
cleanup() {
  if ((${#names[@]})); then
    docker rm -f "${names[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

for ((idx = 0; idx < REPLICAS; idx++)); do
  host_port=$((BASE_PORT + idx))
  name="$(container_name "$idx")"
  names+=("$name")
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker_args "$idx" "$host_port"
  "${args[@]:0:2}" --detach --name "$name" "${args[@]:2}"
  echo "${name}: 127.0.0.1:${host_port}->${CONTAINER_PORT}"
done

echo "Press Ctrl-C to stop local driver replicas."
while true; do
  sleep 3600
done
