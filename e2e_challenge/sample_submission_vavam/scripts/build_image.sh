#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-vavam-driver:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SAMPLE_DIR="${REPO_ROOT}/e2e_challenge/sample_submission_vavam"
ASSET_DIR="${SAMPLE_DIR}/assets/vavam"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

check_asset() {
  local path="$1"
  local min_bytes="$2"
  local rel_path="${path#${REPO_ROOT}/}"
  local size

  [[ -f "${path}" ]] || fail "Missing required asset: ${rel_path}"

  if LC_ALL=C head -c 64 "${path}" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
    fail "${rel_path} is a Git LFS pointer. Run 'git lfs pull' and 'git lfs checkout ${rel_path}' before rebuilding."
  fi

  size="$(wc -c < "${path}")"
  if (( size < min_bytes )); then
    fail "${rel_path} is only ${size} bytes; expected at least ${min_bytes} bytes."
  fi
}

check_asset "${ASSET_DIR}/VAM_width_1024_pretrained_139k.pt" 1000000000
check_asset "${ASSET_DIR}/VQ_ds16_16384_llamagen_encoder.jit" 50000000

docker build -t "${IMAGE}" -f "${SAMPLE_DIR}/Dockerfile" "${REPO_ROOT}"
