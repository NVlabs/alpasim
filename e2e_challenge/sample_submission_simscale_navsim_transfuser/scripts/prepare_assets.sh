#!/usr/bin/env bash
set -euo pipefail

SRC="${1:?usage: prepare_assets.sh /path/to/ltf_sim_navtest.ckpt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DST="$(cd "${SCRIPT_DIR}/.." && pwd)/assets/ltf"
DST="${LTF_ASSET_DIR:-${DEFAULT_DST}}"
EXPECTED_SIZE="${LTF_EXPECTED_SIZE:-224560669}"
EXPECTED_SHA256="${LTF_EXPECTED_SHA256:-9c1a17651bb2cd8e2edf006ea45634432c38554a8f44e0714f64d11ea31f2c69}"
TEMP_ASSET=""

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -f "${SRC}" ]] || fail "missing checkpoint: ${SRC}"
mkdir -p "${DST}"

cleanup() {
  if [[ -n "${TEMP_ASSET}" ]]; then
    rm -f -- "${TEMP_ASSET}"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

TEMP_ASSET="$(mktemp "${DST}/.ltf-prepare-XXXXXX")"
cp -aL -- "${SRC}" "${TEMP_ASSET}"

if LC_ALL=C head -c 64 "${TEMP_ASSET}" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
  fail "checkpoint is a Git LFS pointer: ${SRC}"
fi
size="$(wc -c < "${TEMP_ASSET}")"
[[ "${size}" == "${EXPECTED_SIZE}" ]] || \
  fail "checkpoint size ${size}, expected ${EXPECTED_SIZE}"
actual_sha="$(sha256sum "${TEMP_ASSET}" | awk '{print $1}')"
[[ "${actual_sha}" == "${EXPECTED_SHA256}" ]] || \
  fail "checkpoint sha256 ${actual_sha}, expected ${EXPECTED_SHA256}"

mv -f -- "${TEMP_ASSET}" "${DST}/ltf_sim_navtest.ckpt"
TEMP_ASSET=""
echo "prepared ${DST}/ltf_sim_navtest.ckpt"
