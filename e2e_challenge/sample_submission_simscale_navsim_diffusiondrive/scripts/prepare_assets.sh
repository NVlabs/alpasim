#!/usr/bin/env bash
set -euo pipefail

SRC="${1:?usage: prepare_assets.sh /path/to/diffusiondrive_sim_navhard.ckpt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DST="$(cd "${SCRIPT_DIR}/.." && pwd)/assets/diffusiondrive"
DST="${DIFFUSIONDRIVE_ASSET_DIR:-${DEFAULT_DST}}"
EXPECTED_SIZE="${DIFFUSIONDRIVE_EXPECTED_SIZE:-243596717}"
EXPECTED_SHA256="${DIFFUSIONDRIVE_EXPECTED_SHA256:-8fdbdb3fdfa7b496e7d7a438efbb5c2022377e59cbfd7095270d89623c5d963f}"
ASSET_NAME="diffusiondrive_sim_navhard.ckpt"
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

TEMP_ASSET="$(mktemp "${DST}/.diffusiondrive-prepare-XXXXXX")"
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

chmod 0444 -- "${TEMP_ASSET}"
mv -f -- "${TEMP_ASSET}" "${DST}/${ASSET_NAME}"
TEMP_ASSET=""
echo "prepared ${DST}/${ASSET_NAME}"
