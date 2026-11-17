# Run the SPRING Ising example

First install and activate the Python 3.12 environment using the
[shared installation instructions](../../shared/README.md#install-once-from-the-courserepository-root).
The source-vendored optimizer is shared by every algorithm; no full JaQMC
checkout or editable install of an archived project is needed.

From the repository root, enter `Final_proj` and use that same interpreter:

```bash
cd Final_proj
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/ising-matplotlib
python SPRING/Ising/training/run.py --help
```

The local check used Python 3.12 with JAX/JAXlib 0.9.1, Optax 0.2.8 and
NumPy 2.5.4; exact versions are recorded in the result receipt. The entry point
imports `shared/vendor/jaqmc_spring/src`, whose Apache-2.0 license and provenance
remain beside that source. SciPy is used by the shared independent ED generator;
this runner's dense evaluation uses NumPy.

## Small check first

```bash
python -m pytest -q shared/tests
python SPRING/Ising/training/run.py \
  --n 4 --hidden 3 --batch-size 16 --steps 3 --log-every 1
```

The default output is a **new uniquely named** directory inside
`SPRING/Ising/results/runs/`. It prints that directory after completion.
For a chosen location, add `--output /absolute/path/to/a/new/run_directory`.
The destination must not exist; the runner refuses to overwrite any run.
The tests check the Pauli Hamiltonian, exact local energies, covariance
gradients, spin symmetry, and the actual upstream SPRING update against an
independent regularized linear solve, including memory and active clipping.

## Six-site training

```bash
python SPRING/Ising/training/run.py \
  --n 6 --hidden 12 --steps 250 --batch-size 128 \
  --learning-rate 0.05 --damping 0.001 --mu 0.9 --max-norm 0.1 --seed 7
```

This runs the same small model as the retained reference. Sampling is IID
from the enumerated Born distribution, so this is not a scalable MCMC example.
The wrapper fixes `--optimizer spring`; use the relevant algorithm folder for
other optimizers. See [PARAMETERS.md](training/PARAMETERS.md) for meanings and
output formats, and [PHYSICS.md](PHYSICS.md) for the loss and factor conventions.

Inspect `metrics.json` for the exact initial/final energy, variance, fidelity,
gradient checks, environment and source hashes. `training.csv` contains the
saved trajectory; `diagnostics.png`/`.pdf` plot it. `trained_parameters.npz`
contains only `W` and `c`. It is a wavefunction artifact, **not an exact-resume
checkpoint**: this small runner does not save the optimizer or RNG state.

The [historical reference](results/reference_spring_n6/PROVENANCE.json) is copied
without changing its original metadata. New runs do not write into
`SPRING_Ising_test` or into that reference directory. Shared ED code/data are
indexed in [benchmarks](../../benchmarks/README.md).
