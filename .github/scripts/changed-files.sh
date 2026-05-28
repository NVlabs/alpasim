#!/usr/bin/env bash
set -euo pipefail

# Resolve changed files between BASE and HEAD with robust fallbacks for
# CI checkouts where base history can be missing.
EVENT_NAME="${1:-${GITHUB_EVENT_NAME:-}}"
BASE_SHA="${2:-}"
HEAD_SHA="${3:-${GITHUB_SHA:-HEAD}}"
DEFAULT_BRANCH="${4:-${GITHUB_DEFAULT_BRANCH:-main}}"

ZERO_SHA="0000000000000000000000000000000000000000"
REF_NAME="${GITHUB_REF_NAME:-}"
if [[ -z "${REF_NAME}" && "${GITHUB_REF:-}" == refs/heads/* ]]; then
  REF_NAME="${GITHUB_REF#refs/heads/}"
fi

# copy-pr-bot mirrors trusted PRs into pull-request/<number> branches and
# triggers CI via push. Diff those branches against the default branch rather
# than against the previous mirror push.
if [[ "${REF_NAME}" == pull-request/* ]]; then
  BASE_REF=""
  if git rev-parse --verify --quiet "origin/${DEFAULT_BRANCH}^{commit}" >/dev/null; then
    BASE_REF="origin/${DEFAULT_BRANCH}"
  elif git rev-parse --verify --quiet "${DEFAULT_BRANCH}^{commit}" >/dev/null; then
    BASE_REF="${DEFAULT_BRANCH}"
  fi

  if [[ -n "${BASE_REF}" ]]; then
    MERGE_BASE="$(git merge-base "${BASE_REF}" "${HEAD_SHA}" 2>/dev/null || true)"
    if [[ -n "${MERGE_BASE}" ]]; then
      git diff --name-only "${MERGE_BASE}" "${HEAD_SHA}"
      exit 0
    fi
  fi

  echo "Default branch ${DEFAULT_BRANCH} is not available; using fallback diff." >&2
fi

# Initial pipeline or missing base SHA: return all tracked files so downstream
# steps still run instead of failing due to no diff baseline.
if [[ -z "${BASE_SHA}" || "${BASE_SHA}" == "${ZERO_SHA}" ]]; then
  git ls-files
  exit 0
fi

# Normal path: BASE_SHA is available locally, so diff exactly base..head.
if git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null; then
  git diff --name-only "${BASE_SHA}" "${HEAD_SHA}"
  exit 0
fi

# Fallback path for shallow checkouts or missing base commit. Use a previous
# commit diff and fall back to tracked files if that also fails.
echo "Base SHA ${BASE_SHA} is not available; using fallback diff." >&2
if [[ "${EVENT_NAME}" == "pull_request" ]]; then
  git diff --name-only HEAD^1 HEAD 2>/dev/null || git ls-files
else
  git diff --name-only HEAD~1 HEAD 2>/dev/null || git ls-files
fi
