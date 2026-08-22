#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
METHOD="${GTRS_METHOD:-reward}"
BACKBONE="${GTRS_BACKBONE:-resnet}"
TEMP_ASSET=""

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
        ;;
      expert)
        ASSET_NAME="gtrs_dense_resnet_sim_expert_navhard.ckpt"
        EXPECTED_SHA256="2496b82f5f256d7de09fca656c7634967b8660eb12e5c10386a587283629a7ff"
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
        ;;
      expert)
        ASSET_NAME="gtrs_dense_vov_sim_expert_navhard.ckpt"
        EXPECTED_SHA256="badcf3e7c3e2ecc1d7ecb9fc744c78420c368f96e47b89d1681ade7833cd5e57"
        ;;
      *) fail "GTRS_METHOD=${METHOD} is invalid; expected reward or expert" ;;
    esac
    ;;
  *) fail "GTRS_BACKBONE=${BACKBONE} is invalid; expected resnet or vov" ;;
esac
DEFAULT_DST="${SAMPLE_DIR}/assets/gtrs_dense"

SRC="${1:-}"
[[ -n "${SRC}" ]] || fail "usage: prepare_assets.sh /path/to/${ASSET_NAME}"
DST="${GTRS_ASSET_DIR:-${DEFAULT_DST}}"

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

TEMP_ASSET="$(mktemp "${DST}/.gtrs-prepare-XXXXXX")"
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

mv -f -- "${TEMP_ASSET}" "${DST}/${ASSET_NAME}"
TEMP_ASSET=""
echo "prepared ${DST}/${ASSET_NAME}"
