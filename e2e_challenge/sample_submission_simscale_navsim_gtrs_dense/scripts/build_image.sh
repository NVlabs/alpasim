#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SAMPLE_DIR}/../.." && pwd)"
SOURCE_DIR="e2e_challenge/${SAMPLE_DIR##*/}"
CONTEXT_ASSET_DIR="${SAMPLE_DIR}/assets/gtrs_dense"
METHOD="${GTRS_METHOD:-reward}"
BACKBONE="${GTRS_BACKBONE:-resnet}"
NAVHARD_VOCABULARY="${CONTEXT_ASSET_DIR}/navhard_8192.npy"
NAVHARD_VOCABULARY_SIZE="3932288"
NAVHARD_VOCABULARY_SHA256="cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35"
RELEASE_VOCABULARY="${CONTEXT_ASSET_DIR}/navsim_16384.npy"
RELEASE_VOCABULARY_SIZE="7864448"
RELEASE_VOCABULARY_SHA256="e8c29cfc25add59ae8b64769a4554c6518878726178c0bd889fc8518ebe1261d"
SNAPSHOT=""

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

case "${BACKBONE}" in
  resnet)
    EXPECTED_SIZE="269095388"
    case "${METHOD}" in
      reward)
        ASSET_NAME="gtrs_dense_resnet_sim_reward_navhard.ckpt"
        EXPECTED_SHA256="8dad0395332ccd844785cbfc7c9e24cb3f8d8dbf5cb9ca7f8f8dc75478fcf409"
        DEFAULT_IMAGE="alpasim-e2e-simscale-gtrs-dense:latest"
        SERVICE_VERSION="simscale-gtrs-dense-e2e"
        ;;
      expert)
        ASSET_NAME="gtrs_dense_resnet_sim_expert_navhard.ckpt"
        EXPECTED_SHA256="2496b82f5f256d7de09fca656c7634967b8660eb12e5c10386a587283629a7ff"
        DEFAULT_IMAGE="alpasim-e2e-simscale-gtrs-dense-resnet-expert:latest"
        SERVICE_VERSION="simscale-gtrs-dense-resnet-expert-e2e"
        ;;
      *) fail "GTRS_METHOD=${METHOD} is invalid; expected reward or expert" ;;
    esac
    ;;
  vov)
    EXPECTED_SIZE="332348155"
    case "${METHOD}" in
      reward)
        ASSET_NAME="gtrs_dense_vov_sim_reward_navhard.ckpt"
        EXPECTED_SHA256="7567d269bd8d0757cf906c30612bf1ad167ac7310e8af0ead74dc7798fe54c99"
        DEFAULT_IMAGE="alpasim-e2e-simscale-gtrs-dense-vov-reward:latest"
        SERVICE_VERSION="simscale-gtrs-dense-vov-reward-e2e"
        ;;
      expert)
        ASSET_NAME="gtrs_dense_vov_sim_expert_navhard.ckpt"
        EXPECTED_SHA256="badcf3e7c3e2ecc1d7ecb9fc744c78420c368f96e47b89d1681ade7833cd5e57"
        DEFAULT_IMAGE="alpasim-e2e-simscale-gtrs-dense-vov-expert:latest"
        SERVICE_VERSION="simscale-gtrs-dense-vov-expert-e2e"
        ;;
      *) fail "GTRS_METHOD=${METHOD} is invalid; expected reward or expert" ;;
    esac
    ;;
  *) fail "GTRS_BACKBONE=${BACKBONE} is invalid; expected resnet or vov" ;;
esac
DEFAULT_ASSET_DIR="${CONTEXT_ASSET_DIR}"

BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
INSTALL_DEPENDENCIES="${INSTALL_DEPENDENCIES:-1}"
case "${INSTALL_DEPENDENCIES}" in
  0 | 1) ;;
  *)
    fail \
      "INSTALL_DEPENDENCIES=${INSTALL_DEPENDENCIES} is invalid; expected 0 or 1"
    ;;
esac

IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
ASSET="${GTRS_ASSET_PATH:-${DEFAULT_ASSET_DIR}/${ASSET_NAME}}"

[[ -f "${ASSET}" ]] || fail "missing checkpoint: ${ASSET}"
validate_vocabulary() {
  local path="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local expected_candidates="$4"
  [[ -f "${path}" ]] || fail "missing vocabulary: ${path}"
  python - "${path}" "${expected_size}" "${expected_sha}" "${expected_candidates}" <<'PY'
import hashlib
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
expected_size = int(sys.argv[2])
expected_sha = sys.argv[3]
expected_candidates = int(sys.argv[4])
if path.stat().st_size != expected_size:
    raise SystemExit("vocabulary has an unexpected size")
if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
    raise SystemExit("vocabulary has an unexpected SHA256")
array = np.load(path, allow_pickle=False)
if array.shape != (expected_candidates, 40, 3) or array.dtype != np.float32:
    raise SystemExit("vocabulary has an unexpected array contract")
if not np.isfinite(array).all():
    raise SystemExit("vocabulary contains non-finite values")
PY
}
validate_vocabulary \
  "${NAVHARD_VOCABULARY}" \
  "${NAVHARD_VOCABULARY_SIZE}" \
  "${NAVHARD_VOCABULARY_SHA256}" \
  8192
validate_vocabulary \
  "${RELEASE_VOCABULARY}" \
  "${RELEASE_VOCABULARY_SIZE}" \
  "${RELEASE_VOCABULARY_SHA256}" \
  16384
mkdir -p "${CONTEXT_ASSET_DIR}"

cleanup() {
  if [[ -n "${SNAPSHOT}" ]]; then
    rm -f -- "${SNAPSHOT}"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SNAPSHOT="$(mktemp "${CONTEXT_ASSET_DIR}/.gtrs-build-XXXXXX")"
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
  --build-arg "INSTALL_DEPENDENCIES=${INSTALL_DEPENDENCIES}" \
  --build-arg "GTRS_SOURCE_DIR=${SOURCE_DIR}" \
  --build-arg "GTRS_ASSET_FILE=${snapshot_name}" \
  --build-arg "GTRS_CHECKPOINT_NAME=${ASSET_NAME}" \
  --build-arg "GTRS_BACKBONE=${BACKBONE}" \
  --build-arg "GTRS_SERVICE_VERSION=${SERVICE_VERSION}" \
  "${REPO_ROOT}"
docker image inspect "${IMAGE}" >/dev/null
docker run --rm \
  --user 10001:10001 \
  --entrypoint python \
  "${IMAGE}" -c '
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

checkpoint = Path(sys.argv[1])
expected_size = int(sys.argv[2])
expected_sha = sys.argv[3]
expected_version = sys.argv[4]
expected_backbone = sys.argv[11]
digest = hashlib.sha256()
size = 0
with checkpoint.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
        size += len(chunk)
        digest.update(chunk)

actual_sha = digest.hexdigest()
if size != expected_size:
    raise RuntimeError(f"checkpoint size {size}, expected {expected_size}")
if actual_sha != expected_sha:
    raise RuntimeError(f"checkpoint sha256 {actual_sha}, expected {expected_sha}")
if os.environ.get("GTRS_SERVICE_VERSION") != expected_version:
    raise RuntimeError("unexpected GTRS service version")
if os.environ.get("GTRS_BACKBONE") != expected_backbone:
    raise RuntimeError("unexpected GTRS backbone")

vocabulary_specs = (
    (Path(sys.argv[5]), int(sys.argv[6]), sys.argv[7], (8192, 40, 3)),
    (Path(sys.argv[8]), int(sys.argv[9]), sys.argv[10], (16384, 40, 3)),
)
for vocabulary_path, expected_size, expected_sha, expected_shape in vocabulary_specs:
    vocabulary_digest = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
    if vocabulary_path.stat().st_size != expected_size:
        raise RuntimeError("unexpected vocabulary size")
    if vocabulary_digest != expected_sha:
        raise RuntimeError("unexpected vocabulary SHA256")
    vocabulary = np.load(vocabulary_path, allow_pickle=False)
    if vocabulary.shape != expected_shape or vocabulary.dtype != np.float32:
        raise RuntimeError("unexpected vocabulary array contract")
    if not np.isfinite(vocabulary).all():
        raise RuntimeError("vocabulary contains non-finite values")

import navsim_gtrs_dense_challenge.driver
' "/app/assets/gtrs_dense/${ASSET_NAME}" \
  "${EXPECTED_SIZE}" "${EXPECTED_SHA256}" "${SERVICE_VERSION}" \
  "/app/assets/gtrs_dense/navhard_8192.npy" \
  "${NAVHARD_VOCABULARY_SIZE}" "${NAVHARD_VOCABULARY_SHA256}" \
  "/app/assets/gtrs_dense/navsim_16384.npy" \
  "${RELEASE_VOCABULARY_SIZE}" "${RELEASE_VOCABULARY_SHA256}" \
  "${BACKBONE}"
