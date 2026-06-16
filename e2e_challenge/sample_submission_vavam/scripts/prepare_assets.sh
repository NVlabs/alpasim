#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${SCRIPT_DIR}/../assets/vavam"
SRC="${1:-${VAVAM_ASSET_SRC:-}}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -z "${SRC}" ]]; then
  fail "usage: bash e2e_challenge/sample_submission_vavam/scripts/prepare_assets.sh /path/to/vavam_weights"
fi

[[ -d "${SRC}" ]] || fail "asset source directory does not exist: ${SRC}"
[[ -f "${SRC}/VAM_width_1024_pretrained_139k.pt" ]] || fail "missing VAM_width_1024_pretrained_139k.pt in ${SRC}"
[[ -f "${SRC}/VQ_ds16_16384_llamagen_encoder.jit" ]] || fail "missing VQ_ds16_16384_llamagen_encoder.jit in ${SRC}"

mkdir -p "${DST}"
cp -aL "${SRC}/VAM_width_1024_pretrained_139k.pt" "${DST}/"
cp -aL "${SRC}/VQ_ds16_16384_llamagen_encoder.jit" "${DST}/"
echo "VAVAM assets prepared in ${DST}"
