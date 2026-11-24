#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 /path/to/new-output" >&2
    exit 2
fi

experiment_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repository="$(cd -- "$experiment_dir/../../.." && pwd)"
study_python="$(command -v "${PYTHON:-python}")"
mkdir -- "$1"
output="$(cd -- "$1" && pwd)"
cp -- "$experiment_dir"/scripts/*.py "$output/"
ln -s -- "$repository" "$output/source"

cd -- "$output"
unset PYTHONPATH
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ising_cpus="$("$study_python" - <<'PY'
import os
allowed = sorted(os.sched_getaffinity(0))
preferred = [0, 2, 4, 6]
print(",".join(map(str, preferred if set(preferred) <= set(allowed) else allowed[:4])))
PY
)"
taskset -c "$ising_cpus" "$study_python" ising_study.py --output ising
taskset -c "$ising_cpus" "$study_python" ising_controls.py --frozen-settings ising/frozen_settings.json --output controls
"$study_python" - <<'PY'
import os
from pathlib import Path
import fqhe_study

allowed = sorted(os.sched_getaffinity(0))
if not all(set(group) <= set(allowed) for group in fqhe_study.CPU_GROUPS):
    fqhe_study.CPU_GROUPS = (
        tuple(tuple(allowed[i:i + 2]) for i in range(0, min(len(allowed) - 1, 8), 2))
        or (tuple(allowed),)
    )
fqhe_study.main(Path.cwd() / "fqhe")
PY
taskset -c "$ising_cpus" "$study_python" fqhe_exact_control.py --output fqhe_exact
"$study_python" summarize.py --output analysis
