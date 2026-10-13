#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend_dir="${repo_root}/.external/ls20-solver"
backend_url="https://github.com/zzZYHarold/arc-agi-3-ls20-agent.git"
backend_commit="bcf25348fd88fddbaaabb5c332f6bf3bb3a3fb60"

mkdir -p "${repo_root}/.external"
if [[ ! -d "${backend_dir}/.git" ]]; then
  git clone "${backend_url}" "${backend_dir}"
fi

git -C "${backend_dir}" fetch --quiet origin
git -C "${backend_dir}" checkout --quiet --detach "${backend_commit}"

actual_commit="$(git -C "${backend_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${backend_commit}" ]]; then
  echo "backend commit mismatch: ${actual_commit}" >&2
  exit 1
fi

echo "ls20 solver backend ready: ${backend_dir} @ ${actual_commit}"
