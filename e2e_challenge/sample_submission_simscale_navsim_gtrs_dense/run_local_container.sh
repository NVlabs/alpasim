#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${GTRS_BACKBONE:-resnet}"
METHOD="${GTRS_METHOD:-reward}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

case "${BACKBONE}" in
  resnet)
    case "${METHOD}" in
      reward) DEFAULT_IMAGE="alpasim-e2e-simscale-gtrs-dense:latest" ;;
      expert) DEFAULT_IMAGE="alpasim-e2e-simscale-gtrs-dense-resnet-expert:latest" ;;
      *) fail "GTRS_METHOD=${METHOD} is invalid; expected reward or expert" ;;
    esac
    ;;
  vov)
    case "${METHOD}" in
      reward) DEFAULT_IMAGE="alpasim-e2e-simscale-gtrs-dense-vov-reward:latest" ;;
      expert) DEFAULT_IMAGE="alpasim-e2e-simscale-gtrs-dense-vov-expert:latest" ;;
      *) fail "GTRS_METHOD=${METHOD} is invalid; expected reward or expert" ;;
    esac
    ;;
  *) fail "GTRS_BACKBONE=${BACKBONE} is invalid; expected resnet or vov" ;;
esac

IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
BASE_PORT="${ALPASIM_DRIVER_BASE_PORT:-${ALPASIM_DRIVER_PORT:-6789}}"
CONTAINER_PORT="${ALPASIM_DRIVER_CONTAINER_PORT:-6789}"
REPLICAS="${ALPASIM_DRIVER_REPLICAS:-1}"
GPUS="${ALPASIM_DOCKER_GPUS:-all}"
DETACH="${ALPASIM_DRIVER_DETACH:-0}"
PREFIX="${ALPASIM_DRIVER_NAME_PREFIX:-alpasim-e2e-simscale-gtrs-dense}"
SPEED_ENHANCEMENT="${GTRS_SPEED_ENHANCEMENT:-1}"

case "${SPEED_ENHANCEMENT}" in
  1)
    DEFAULT_EP_EXPONENT=3
    DEFAULT_SPEED_TOP_K=64
    DEFAULT_SPEED_WEIGHT=3
    ;;
  0)
    DEFAULT_EP_EXPONENT=1
    DEFAULT_SPEED_TOP_K=0
    DEFAULT_SPEED_WEIGHT=0
    ;;
  *) fail "GTRS_SPEED_ENHANCEMENT must be 0 or 1" ;;
esac

SCORER_MODE="${GTRS_SCORER_MODE:-nc_dac_ep}"
EP_EXPONENT="${GTRS_EP_EXPONENT:-${DEFAULT_EP_EXPONENT}}"
SPEED_TOP_K="${GTRS_SPEED_TOP_K:-${DEFAULT_SPEED_TOP_K}}"
SPEED_WEIGHT="${GTRS_SPEED_WEIGHT:-${DEFAULT_SPEED_WEIGHT}}"
SPEED_PROXY="${GTRS_SPEED_PROXY:-longitudinal}"
CURVATURE_WEIGHT="${GTRS_CURVATURE_WEIGHT:-0.0}"
HEADING_CHANGE_WEIGHT="${GTRS_HEADING_CHANGE_WEIGHT:-0.0}"
TRAJECTORY_TIME_SCALE="${GTRS_TRAJECTORY_TIME_SCALE:-1.0}"

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
[[ "${SCORER_MODE}" == "release" || \
   "${SCORER_MODE}" == "nc_dac_ep" || \
   "${SCORER_MODE}" == "safety_gate_ep" ]] || \
  fail "GTRS_SCORER_MODE=${SCORER_MODE} is invalid; expected release, nc_dac_ep, or safety_gate_ep"
[[ "${EP_EXPONENT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  fail "GTRS_EP_EXPONENT=${EP_EXPONENT} is invalid; expected a positive number"
[[ "${SPEED_TOP_K}" =~ ^[0-9]+$ ]] || \
  fail "GTRS_SPEED_TOP_K=${SPEED_TOP_K} is invalid; expected an integer"
((10#${SPEED_TOP_K} <= 16384)) || \
  fail "GTRS_SPEED_TOP_K=${SPEED_TOP_K} is invalid; expected 0..16384"
[[ "${SPEED_WEIGHT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  fail "GTRS_SPEED_WEIGHT=${SPEED_WEIGHT} is invalid; expected a non-negative number"
[[ "${SPEED_PROXY}" == "longitudinal" || \
   "${SPEED_PROXY}" == "longitudinal_0p5s" || \
   "${SPEED_PROXY}" == "path_length" ]] || \
  fail "GTRS_SPEED_PROXY=${SPEED_PROXY} is invalid; expected longitudinal, longitudinal_0p5s, or path_length"
[[ "${CURVATURE_WEIGHT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  fail "GTRS_CURVATURE_WEIGHT=${CURVATURE_WEIGHT} is invalid; expected a non-negative number"
[[ "${HEADING_CHANGE_WEIGHT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  fail "GTRS_HEADING_CHANGE_WEIGHT=${HEADING_CHANGE_WEIGHT} is invalid; expected a non-negative number"
[[ "${TRAJECTORY_TIME_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  fail "GTRS_TRAJECTORY_TIME_SCALE=${TRAJECTORY_TIME_SCALE} is invalid; expected 1.0..1.25"
awk -v scale="${TRAJECTORY_TIME_SCALE}" \
  'BEGIN { exit !(scale >= 1.0 && scale <= 1.25) }' || \
  fail "GTRS_TRAJECTORY_TIME_SCALE=${TRAJECTORY_TIME_SCALE} is invalid; expected 1.0..1.25"
if [[ "${SPEED_WEIGHT}" != "0" && "${SPEED_WEIGHT}" != "0.0" ]] || \
   [[ "${CURVATURE_WEIGHT}" != "0" && "${CURVATURE_WEIGHT}" != "0.0" ]] || \
   [[ "${HEADING_CHANGE_WEIGHT}" != "0" && "${HEADING_CHANGE_WEIGHT}" != "0.0" ]]; then
  ((10#${SPEED_TOP_K} > 0)) || fail "GTRS_SPEED_TOP_K must be positive when reranking is enabled"
fi

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
    -e "GTRS_DEVICE=${GTRS_DEVICE:-cuda}"
    -e "GTRS_SPEED_ENHANCEMENT=${SPEED_ENHANCEMENT}"
    -e "GTRS_SCORER_MODE=${SCORER_MODE}"
    -e "GTRS_EP_EXPONENT=${EP_EXPONENT}"
    -e "GTRS_SPEED_TOP_K=${SPEED_TOP_K}"
    -e "GTRS_SPEED_WEIGHT=${SPEED_WEIGHT}"
    -e "GTRS_SPEED_PROXY=${SPEED_PROXY}"
    -e "GTRS_CURVATURE_WEIGHT=${CURVATURE_WEIGHT}"
    -e "GTRS_HEADING_CHANGE_WEIGHT=${HEADING_CHANGE_WEIGHT}"
    -e "GTRS_TRAJECTORY_TIME_SCALE=${TRAJECTORY_TIME_SCALE}"
    -e "GTRS_MAX_BATCH_SIZE=${GTRS_MAX_BATCH_SIZE:-1}"
    -e "GTRS_BATCH_WINDOW_MS=${GTRS_BATCH_WINDOW_MS:-2}"
  )
  if [[ "${GTRS_CHECKPOINT_PATH+x}" == "x" ]]; then
    args+=(-e "GTRS_CHECKPOINT_PATH=${GTRS_CHECKPOINT_PATH}")
  fi
  if [[ "${GTRS_VOCAB_PATH+x}" == "x" ]]; then
    args+=(-e "GTRS_VOCAB_PATH=${GTRS_VOCAB_PATH}")
  fi
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
