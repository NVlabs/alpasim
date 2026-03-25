#!/usr/bin/env bash
set -euo pipefail

EVENT_NAME="${1:-${GITHUB_EVENT_NAME:-}}"
BASE_SHA="${2:-}"
HEAD_SHA="${3:-${GITHUB_SHA:-HEAD}}"

ZERO_SHA="0000000000000000000000000000000000000000"

if [[ -z "${BASE_SHA}" || "${BASE_SHA}" == "${ZERO_SHA}" ]]; then
  git ls-files
  exit 0
fi

if git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null; then
  git diff --name-only "${BASE_SHA}" "${HEAD_SHA}"
  exit 0
fi

echo "Base SHA ${BASE_SHA} is not available; using fallback diff." >&2
if [[ "${EVENT_NAME}" == "pull_request" ]]; then
  git diff --name-only HEAD^1 HEAD 2>/dev/null || git ls-files
else
  git diff --name-only HEAD~1 HEAD 2>/dev/null || git ls-files
fi
