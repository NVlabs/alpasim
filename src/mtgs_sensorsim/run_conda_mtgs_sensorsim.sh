#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/a8001/anaconda3/envs/mtgs}"
MTGS_REPO="${MTGS_REPO:-${REPO_ROOT}/MTGS}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/data/nre-artifacts/road_block-331220_4690660_331190_4690710}"
CONFIG_PATH="${CONFIG_PATH:-${MTGS_REPO}/experiments/main_mt/MTGS/road_block-331220_4690660_331190_4690710/config.yml}"

export PYTHONPATH="${REPO_ROOT}/src/mtgs_sensorsim:${REPO_ROOT}/src/grpc:${MTGS_REPO}:${PYTHONPATH:-}"
export MTGS_REPO
export CUDA_HOME="${CUDA_HOME:-${CONDA_ENV_PREFIX}}"
export PATH="${CONDA_ENV_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_mtgs_${USER:-user}}"
# Keep gsplat JIT from failing when CUDA devices are not visible during compile.
# Override this for a known GPU architecture if needed, e.g. TORCH_CUDA_ARCH_LIST=8.6.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9}"
mkdir -p "${MPLCONFIGDIR}" "${TORCH_EXTENSIONS_DIR}"

cd "${MTGS_REPO}"

if [[ "$#" -gt 0 ]]; then
  exec "${CONDA_ENV_PREFIX}/bin/python" -m alpasim_mtgs_sensorsim "$@"
fi

exec "${CONDA_ENV_PREFIX}/bin/python" -m alpasim_mtgs_sensorsim \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-50053}" \
  --config "${CONFIG_PATH}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --scene-id "${SCENE_ID:-mtgs-road_block-331220_4690660_331190_4690710}" \
  --travel-id "${TRAVEL_ID:-7}" \
  --native-height "${NATIVE_HEIGHT:-1080}" \
  --native-width "${NATIVE_WIDTH:-1920}" \
  --warmup-renders "${WARMUP_RENDERS:-1}" \
  --warmup-height "${WARMUP_HEIGHT:-180}" \
  --warmup-width "${WARMUP_WIDTH:-320}" \
  --log-level "${LOG_LEVEL:-INFO}"
