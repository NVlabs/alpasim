#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-simscale-diffusiondrive:latest}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SAMPLE_DIR}/../.." && pwd)"
SOURCE_DIR="e2e_challenge/${SAMPLE_DIR##*/}"
ASSET_DIR="${SAMPLE_DIR}/assets/diffusiondrive"
ASSET_NAME="diffusiondrive_sim_navhard.ckpt"
ASSET="${DIFFUSIONDRIVE_ASSET_PATH:-${ASSET_DIR}/${ASSET_NAME}}"
EXPECTED_SIZE="${DIFFUSIONDRIVE_EXPECTED_SIZE:-243596717}"
EXPECTED_SHA256="${DIFFUSIONDRIVE_EXPECTED_SHA256:-8fdbdb3fdfa7b496e7d7a438efbb5c2022377e59cbfd7095270d89623c5d963f}"
SNAPSHOT=""
PROBE_SOURCE="$(<"${SCRIPT_DIR}/verify_image.py")"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "${ASSET}" ]] || fail "missing checkpoint: ${ASSET}"
mkdir -p "${ASSET_DIR}"

cleanup() {
  if [[ -n "${SNAPSHOT}" ]]; then
    rm -f -- "${SNAPSHOT}"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SNAPSHOT="$(mktemp "${ASSET_DIR}/.diffusiondrive-build-XXXXXX")"
cp -aL -- "${ASSET}" "${SNAPSHOT}"
chmod 0444 -- "${SNAPSHOT}"

size="$(wc -c < "${SNAPSHOT}")"
[[ "${size}" == "${EXPECTED_SIZE}" ]] || \
  fail "checkpoint size ${size}, expected ${EXPECTED_SIZE}"
sha="$(sha256sum "${SNAPSHOT}" | awk '{print $1}')"
[[ "${sha}" == "${EXPECTED_SHA256}" ]] || \
  fail "checkpoint sha256 ${sha}, expected ${EXPECTED_SHA256}"

snapshot_name="${SNAPSHOT##*/}"
docker build \
  --tag "${IMAGE}" \
  --file "${SAMPLE_DIR}/Dockerfile" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "DIFFUSIONDRIVE_SOURCE_DIR=${SOURCE_DIR}" \
  --build-arg "DIFFUSIONDRIVE_ASSET_FILE=${snapshot_name}" \
  "${REPO_ROOT}"
configured_user="$(
  docker image inspect --format '{{.Config.User}}' "${IMAGE}"
)"
[[ "${configured_user}" == "10001:10001" ]] || \
  fail "image Config.User ${configured_user:-<empty>}, expected 10001:10001"
docker run --rm \
  --user 0:0 \
  --entrypoint python \
  "${IMAGE}" -c "${PROBE_SOURCE}" filesystem
docker run --rm \
  --entrypoint python \
  --env "DIFFUSIONDRIVE_PROBE_EXPECTED_SIZE=${EXPECTED_SIZE}" \
  --env "DIFFUSIONDRIVE_PROBE_EXPECTED_SHA256=${EXPECTED_SHA256}" \
  "${IMAGE}" -c "${PROBE_SOURCE}" runtime
