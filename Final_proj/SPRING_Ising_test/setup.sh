#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v uv >/dev/null || { printf '%s\n' 'Install uv first: https://docs.astral.sh/uv/'; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/spring_ising_uv_cache}"
export UV_LINK_MODE=copy
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python --requirement requirements-lock.txt
