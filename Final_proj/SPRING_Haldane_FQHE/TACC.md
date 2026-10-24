# Run the 200,000-update experiment on TACC

This package trains the LLL occupation-basis neural wavefunction with actual
SPRING and ordinary Born sampling. The long-run configurations keep the
Hamiltonian and optimizer choices from the pilot. **200,000 means parameter
updates, not epochs over a fixed dataset.** A larger budget tests whether the
pilot was limited by training length; convergence is an experimental result.

The supplied Slurm job runs one Python process per independent seed. The
trainer does not implement distributed MPI or multi-GPU optimization. On TACC,
one serial or threaded application is launched directly; `ibrun` is for MPI.
See the official [Stampede3 launching instructions](https://docs.tacc.utexas.edu/hpc/stampede3/#launching-applications).

## 1. Transfer the clean project and prepare Python

Transfer the entire `SPRING_Haldane_FQHE` folder, including `checkpoints/`, `configs/`,
`src/`, and `vendor/jaqmc_spring/`. No sibling Ising folder or local virtual
environment is required. Store the project and environment in your chosen
TACC locations; numerical output defaults to `$SCRATCH/fqh_spring/`.

After loading a Python **3.12** module available on your system:

```bash
cd /path/to/SPRING_Haldane_FQHE
module list
export FQH_PYTHON=python3.12
bash tacc/setup_env.sh cpu "$WORK/venvs/fqh_spring_cpu"
```

The separate `Haldane_FQHE_TACC.tar.gz` snapshot extracts into `Haldane_FQHE`.
If using that archive, enter its extracted directory instead and verify
`sha256sum -c SHA256SUMS` before setup. The commands below work from either
package directory.

The requirements pin direct numerical dependencies to the locally verified
versions. The setup installs the bundled SPRING subset and this project with
`--no-deps`; it does not install JaQMC's unrelated chemistry stack. Compiled
packages must be installed on the target architecture. Do not copy an x86
virtual environment to Vista's ARM nodes.

## 2. Do a short compute-node check

Choose your actual allocation and an accessible CPU partition. Inspect current
limits with `qlimits`; queue availability is system-specific. For example,
Stampede3 CPU partitions include `spr`, `icx`, and `skx`. Their current guide
also explains [queue selection and whole-node charging](https://docs.tacc.utexas.edu/hpc/stampede3/#running-jobs).

```bash
bash tacc/submit.sh \
  --account YOUR_ALLOCATION --partition YOUR_CPU_QUEUE \
  --venv "$WORK/venvs/fqh_spring_cpu" \
  --config configs/ne8_dense_200k.json \
  --run-name ne8_dense_check --seeds 71 \
  --steps 1000 --time 00:30:00 --cpus 4 --dry-run
```

Remove `--dry-run` to submit. Inspect the output and checkpoint before the long
run. Four CPU cores are a starting point from the local benchmark, not a TACC
performance claim. Benchmark a few CPU counts on the allocated node. Stampede3
charges for a whole node even when one process uses only part of it, so several
independent runs packed on one node can improve utilization after profiling.
The supplied array keeps the first experiment simple and reproducible.

## 3. Submit the long run

```bash
bash tacc/submit.sh \
  --account YOUR_ALLOCATION --partition YOUR_CPU_QUEUE \
  --venv "$WORK/venvs/fqh_spring_cpu" \
  --config configs/ne8_dense_200k.json \
  --run-name ne8_dense_200k --seeds 71,137,223 \
  --steps 200000 --time 08:00:00 --cpus 4
```

Each array member starts from the same saved wavefunction and uses a different
Monte Carlo seed. The seeds do not create independently initialized neural
models. The eight-hour request is an editable scheduling budget, not an
estimated runtime or a guarantee that 200,000 updates fit on every node.
Use `configs/ne8_attention_200k.json` with a different `--run-name` to test the
attention branch. The default dense configuration continues the best dense
pilot, whereas the attention configuration adds its branch to the earlier
six-layer source. These are different starting wavefunctions. For a controlled
architecture comparison, pair `configs/ne8_dense_common_source_200k.json` with
`configs/ne8_attention_200k.json`. Both use `checkpoints/source_ne8.npz` and
start from the same wavefunction; match update count, batch, optimizer settings,
and seeds, and use separate run names.

The script retains native BLAS/OpenMP thread counts of one and passes the CPU
budget to the trainer's affinity setting before JAX initializes. This avoids
the oversubscription seen in the original local experiments.

## 4. Monitor, resume, and collect

```bash
squeue -u "$USER"
tail -f "$SCRATCH/fqh_spring/ne8_dense_200k/logs/JOBID_ARRAYINDEX.out"
```

Within each `seedNN/` directory, `checkpoint_latest.npz` contains parameters,
SPRING memory, walkers, RNG state, and the completed update count. The CSV
trajectory is appended incrementally, and only the configured number of
historical checkpoints is retained. Logs and checkpoints therefore do not
grow by writing a full parameter snapshot on every update.

Slurm sends an early `SIGUSR1` two minutes before wall time. The batch script
forwards it to Python, which checkpoints after its current update and stops.
The same forwarding handles `SIGTERM`. A node failure or immediate hard kill
can still lose work after the most recent periodic checkpoint.

Resume a selected seed using the **same run directory and configuration**:

```bash
bash tacc/submit.sh \
  --account YOUR_ALLOCATION --partition YOUR_CPU_QUEUE \
  --venv "$WORK/venvs/fqh_spring_cpu" \
  --config configs/ne8_dense_200k.json \
  --run-name ne8_dense_200k --seeds 71 \
  --steps 200000 --time 08:00:00 --cpus 4 --resume
```

`--steps 200000` is the target total for this continuation, not 200,000 more
updates on every restart. Do not run two jobs against the same seed directory.
After completion, use the exact evaluator described in the main README for
the N=8 energies and compare seed-to-seed variation, energy variance,
acceptance, and angular momentum. Training batch energy alone is insufficient
to establish ground-state convergence.

Copy completed output to persistent project storage. Both
[Stampede3](https://docs.tacc.utexas.edu/hpc/stampede3/#file-systems) and
[Vista](https://docs.tacc.utexas.edu/hpc/vista/#file-systems) document scratch
purge policies; `$SCRATCH` is the working location, not the permanent archive.

## Optional: one NVIDIA GPU

The GPU route is supplied for profiling and has not been benchmarked locally.
Use a compatible NVIDIA queue and a fresh environment for that node's CPU
architecture. Vista Grace-Hopper nodes use ARM CPUs; Stampede3's H100 nodes
use x86 CPUs. Intel PVC nodes require a different software stack and are not
covered by the CUDA requirements. The [Vista guide](https://docs.tacc.utexas.edu/hpc/vista/)
describes its node types and job submission.

For a driver compatible with CUDA 12:

```bash
bash tacc/setup_env.sh cuda12 "$WORK/venvs/fqh_spring_cuda12"
bash tacc/submit.sh \
  --account YOUR_ALLOCATION --partition YOUR_NVIDIA_QUEUE \
  --venv "$WORK/venvs/fqh_spring_cuda12" --backend gpu \
  --config configs/ne8_dense_200k.json \
  --run-name ne8_dense_gpu_check --seeds 71 \
  --steps 1000 --time 00:30:00 --cpus 4
```

For CUDA 13, use `setup_env.sh cuda13` with a separate environment. Current
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)
list Linux x86 and ARM support and the driver/toolkit compatibility rules.
Confirm `jax.devices()` inside the allocated compute node and keep the
site's Python/CUDA modules consistent between setup and the batch job.
The GPU job fails if a GPU backend is unavailable instead of silently becoming
a CPU run. The script selects GPU 0 when `CUDA_VISIBLE_DEVICES` is unset and
does not request `--gres` or `--gpus-per-task`, which the current Stampede3 and
Vista guides do not support. A small batch may run faster on the CPU; compare
measured update throughput before committing the full GPU allocation.

Documentation checked against the official guides on **2026-10-23**. No TACC
account was accessed and no remote jobs were submitted while packaging.
