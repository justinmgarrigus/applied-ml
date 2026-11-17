# Run memory-free damped MinSR control for Ising

Install and activate the Python 3.12 environment using the
[shared installation instructions](../../shared/README.md#install-once-from-the-courserepository-root).
From the repository root, enter `Final_proj` and keep the same interpreter:

```bash
cd Final_proj
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/ising-matplotlib
python MinSR/Ising/training/run.py \
  --n 4 --hidden 3 --batch-size 16 --steps 3 --log-every 1 --mu 0
```

The solver receives `g/2`, with positive damping and `mu=0`.
The wrapper rejects nonzero `--mu`; it does not silently turn on SPRING memory.
This is a damped SR/MinSR control, not the original paper's eigenvalue filter.
The wrapper fixes `--optimizer damped-minsr` and rejects another optimizer name.
See [PHYSICS.md](PHYSICS.md) for the common energy loss, input/output and
sampling, and [PARAMETERS.md](training/PARAMETERS.md) for all flags.

Each default run creates a uniquely named directory under
`MinSR/Ising/results/runs/`. For an explicit destination, add
`--output /absolute/path/to/a/new/run_directory`; that directory must not exist.
This protects stored results from accidental replacement.

The command is an execution check, not a request for long training or proof of
convergence. Increase `--steps` deliberately for a convergence study, matching
Hamiltonian, ansatz, initial parameters, batch, sampling and evaluation across
methods. Different optimizers may need different learning-rate choices.
A three-update check establishes no ranking in accuracy or speed.

Inspect `metrics.json`, `training.csv` and `diagnostics.png`/`.pdf` for exact
complete-basis evaluation and sampled update diagnostics. The
`trained_parameters.npz` file contains `W` and `c` only; optimizer and RNG
states are not saved, so this runner cannot resume exactly. Gradient-check
arrays are saved separately. Shared model/optimizer unit tests can be run with
`python -m pytest -q shared/tests`; no full retraining is needed to run them.

The [shared ED benchmark index](../../benchmarks/README.md) supplies independent
references. ED eigenvectors do not enter the sampled updates. No historical
SPRING result is relabelled as a result from this method; see
[results](results/README.md) for the checks actually performed.
