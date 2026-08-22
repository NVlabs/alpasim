#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-simscale-ltf:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAMPLE_DIR="${REPO_ROOT}/e2e_challenge/sample_submission_simscale_navsim_transfuser"
ASSET="${LTF_ASSET_PATH:-${SAMPLE_DIR}/assets/ltf/ltf_sim_navtest.ckpt}"
ASSET_DIR="${SAMPLE_DIR}/assets/ltf"
EXPECTED_SIZE="${LTF_EXPECTED_SIZE:-224560669}"
EXPECTED_SHA256="${LTF_EXPECTED_SHA256:-9c1a17651bb2cd8e2edf006ea45634432c38554a8f44e0714f64d11ea31f2c69}"
SNAPSHOT=""

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

SNAPSHOT="$(mktemp "${ASSET_DIR}/.ltf-build-XXXXXX")"
cp -aL -- "${ASSET}" "${SNAPSHOT}"
chmod a-w -- "${SNAPSHOT}"

if LC_ALL=C head -c 64 "${SNAPSHOT}" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
  fail "checkpoint is a Git LFS pointer: ${ASSET}"
fi
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
  --build-arg "LTF_ASSET_FILE=${snapshot_name}" \
  "${REPO_ROOT}"
