#!/usr/bin/env bash
# Run after loading the Python/module collection intended for the compute node.
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
backend=${1:-cpu}
venv_dir=${2:-"$project_dir/.venv"}
python_command=${FQH_PYTHON:-python3.12}

case "$backend" in cpu|cuda12|cuda13) ;;
  *) printf 'Usage: %s [cpu|cuda12|cuda13] [VENV_DIRECTORY]\n' "$0" >&2; exit 2;;
esac
"$python_command" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'
"$python_command" -m venv "$venv_dir"
venv_python="$venv_dir/bin/python"
"$venv_python" -m pip install --upgrade pip
if [[ "$backend" == cpu ]]; then
  "$venv_python" -m pip install -r "$project_dir/requirements-cpu.txt"
elif [[ "$backend" == cuda12 ]]; then
  "$venv_python" -m pip install -r "$project_dir/requirements-gpu.txt"
else
  "$venv_python" -m pip install -r "$project_dir/requirements-common.txt" 'jax[cuda13]==0.9.1'
fi
"$venv_python" -m pip install --no-deps "$project_dir/vendor/jaqmc_spring"
"$venv_python" -m pip install --no-deps -e "$project_dir"
"$venv_python" -m pip check
printf '\nEnvironment prepared: %s\n' "$venv_dir"
printf 'Check jax.devices() inside a compute allocation before a production run.\n'
