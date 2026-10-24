#!/usr/bin/env bash
# Prepare the directory/environment and submit one independent seed per job.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
account=''
partition=''
config='configs/ne8_dense_200k.json'
run_name='ne8_dense_200k'
seeds='71,137'
steps=200000
cpus=4
walltime='08:00:00'
backend=cpu
venv_dir="$project_dir/.venv"
resume=0
dry_run=0
run_root=''

usage() {
  cat <<'HELP'
Usage: bash tacc/submit.sh --account ALLOCATION --partition QUEUE [options]
  --config PATH       Project-relative or absolute JSON configuration
  --run-name NAME     Name below $SCRATCH/fqh_spring (default ne8_dense_200k)
  --run-root PATH     Explicit absolute output root instead
  --seeds A,B         Independent seeds (default 71,137)
  --steps N           Target total SPRING updates (default 200000)
  --cpus N            CPU affinity per Python process (default 4)
  --time HH:MM:SS     Slurm wall time (default 08:00:00)
  --backend cpu|gpu   Select JAX device type (default cpu)
  --venv PATH         Environment prepared by setup_env.sh
  --resume            Resume every selected seed from checkpoint_latest.npz
  --dry-run           Print the submission command without calling sbatch
HELP
}
while (($#)); do
  case "$1" in
    --account|--partition|--config|--run-name|--run-root|--seeds|--steps|--cpus|--time|--backend|--venv)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      case "$1" in
        --account) account=$2;; --partition) partition=$2;;
        --config) config=$2;; --run-name) run_name=$2;;
        --run-root) run_root=$2;; --seeds) seeds=$2;;
        --steps) steps=$2;; --cpus) cpus=$2;; --time) walltime=$2;;
        --backend) backend=$2;; --venv) venv_dir=$2;;
      esac
      shift 2;;
    --resume) resume=1; shift;;
    --dry-run) dry_run=1; shift;;
    --help|-h) usage; exit 0;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2;;
  esac
done
[[ -n "$account" && -n "$partition" ]] || { usage >&2; exit 2; }
[[ "$steps" =~ ^[1-9][0-9]*$ && "$cpus" =~ ^[1-9][0-9]*$ ]] || { printf 'steps and cpus must be positive integers\n' >&2; exit 2; }
[[ "$seeds" =~ ^[0-9]+(,[0-9]+)*$ ]] || { printf 'seeds must be comma-separated nonnegative integers\n' >&2; exit 2; }
[[ "$backend" == cpu || "$backend" == gpu ]] || { printf 'backend must be cpu or gpu\n' >&2; exit 2; }
[[ "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { printf 'run-name must be a simple directory name\n' >&2; exit 2; }
if [[ "$config" != /* ]]; then config="$project_dir/$config"; fi
[[ -f "$config" ]] || { printf 'Configuration does not exist: %s\n' "$config" >&2; exit 2; }
if [[ -z "$run_root" ]]; then
  : "${SCRATCH:?Set SCRATCH on TACC or pass --run-root /absolute/path}"
  run_root="$SCRATCH/fqh_spring/$run_name"
fi
[[ "$run_root" == /* ]] || { printf 'run-root must be absolute\n' >&2; exit 2; }
if [[ "$venv_dir" != /* ]]; then venv_dir="$PWD/$venv_dir"; fi

IFS=',' read -r -a seed_values <<< "$seeds"
declare -A seen_seeds=()
for seed in "${seed_values[@]}"; do
  [[ -z ${seen_seeds[$seed]+present} ]] || { printf 'Duplicate seed would share an output directory: %s\n' "$seed" >&2; exit 2; }
  seen_seeds[$seed]=1
  if ((resume)); then
    [[ -f "$run_root/seed${seed}/checkpoint_latest.npz" ]] || {
      printf 'Resume checkpoint is missing: %s\n' "$run_root/seed${seed}/checkpoint_latest.npz" >&2
      exit 2
    }
  fi
done
array_end=$((${#seed_values[@]} - 1))
export FQH_PROJECT_DIR="$project_dir" FQH_CONFIG="$config"
export FQH_RUN_ROOT="$run_root" FQH_VENV="$venv_dir"
export FQH_SEEDS="$seeds" FQH_STEPS="$steps" FQH_BACKEND="$backend" FQH_RESUME="$resume"
submission=( sbatch --account="$account" --partition="$partition"
  --cpus-per-task="$cpus" --time="$walltime" --array="0-$array_end"
  --output="$run_root/logs/%A_%a.out"
  --error="$run_root/logs/%A_%a.err" "$project_dir/tacc/train.sbatch" )
if ((dry_run)); then
  printf 'Exported FQH_PROJECT_DIR=%q FQH_CONFIG=%q FQH_RUN_ROOT=%q\n' "$project_dir" "$config" "$run_root"
  printf 'Exported FQH_VENV=%q FQH_SEEDS=%q FQH_STEPS=%q FQH_BACKEND=%q FQH_RESUME=%q\n' "$venv_dir" "$seeds" "$steps" "$backend" "$resume"
  printf '%q ' "${submission[@]}"; printf '\n'
else
  [[ -x "$venv_dir/bin/python" ]] || { printf 'Python environment is missing: %s\n' "$venv_dir" >&2; exit 2; }
  mkdir -p -- "$run_root/logs"
  "${submission[@]}"
fi
