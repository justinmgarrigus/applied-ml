#!/usr/bin/env bash
# Submitted through submit.sh, which supplies account, queue, logs, and array.
# One Python/JAX process per array member: this application is not MPI.
#SBATCH --job-name=fqh-spring
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@120

set -euo pipefail
: "${FQH_PROJECT_DIR:?Submit with tacc/submit.sh}"
: "${FQH_RUN_ROOT:?Missing run directory}"
: "${FQH_CONFIG:?Missing configuration}"
: "${FQH_VENV:?Missing environment}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=true
export PYTHONPATH="$FQH_PROJECT_DIR/src:$FQH_PROJECT_DIR/vendor/jaqmc_spring/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${FQH_BACKEND:-cpu} == gpu ]]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi

IFS=',' read -r -a seeds <<< "${FQH_SEEDS:-71,137}"
array_index=${SLURM_ARRAY_TASK_ID:-0}
seed=${seeds[$array_index]:?Array index has no corresponding seed}
run_output="$FQH_RUN_ROOT/seed${seed}"
mkdir -p -- "$FQH_RUN_ROOT"
cd -- "$FQH_RUN_ROOT"

arguments=( -m fqh_spring.train --config "$FQH_CONFIG"
  --output "$run_output" --steps "${FQH_STEPS:-200000}"
  --seed "$seed" --cpus "${SLURM_CPUS_PER_TASK:-4}"
  --backend "${FQH_BACKEND:-cpu}" )
if [[ ${FQH_RESUME:-0} == 1 ]]; then
  [[ -f "$run_output/checkpoint_latest.npz" ]] || {
    printf 'Resume checkpoint is missing: %s\n' "$run_output/checkpoint_latest.npz" >&2
    exit 2
  }
  arguments+=( --resume )
fi

# The batch shell receives the early Slurm signal and forwards it to Python.
# Python writes a complete checkpoint and exits after its current JIT update.
child_pid=''
forward_usr1() { [[ -z "$child_pid" ]] || kill -USR1 "$child_pid" 2>/dev/null || true; }
forward_term() { [[ -z "$child_pid" ]] || kill -TERM "$child_pid" 2>/dev/null || true; }
trap forward_usr1 USR1
trap forward_term TERM INT
printf 'Job %s, seed %s, backend %s, output %s\n' "${SLURM_JOB_ID:-local}" "$seed" "${FQH_BACKEND:-cpu}" "$run_output"
"$FQH_VENV/bin/python" "${arguments[@]}" &
child_pid=$!

# A trapped signal interrupts wait; continue waiting until Python has stopped.
while true; do
  set +e
  wait "$child_pid"
  exit_status=$?
  set -e
  if ! kill -0 "$child_pid" 2>/dev/null; then break; fi
done
exit "$exit_status"
