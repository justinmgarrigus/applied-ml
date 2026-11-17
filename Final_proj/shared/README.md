# Shared implementation

All algorithm folders use the same scientific backend for each problem:

- `ising_backend.py`: periodic Pauli Ising model, positive RBM, exact Born
  sampling over the small enumerated basis, energy and gradient diagnostics.
- `fqh_spring/`: LLL sphere operator, complex neural wavefunctions,
  fixed-sector MCMC, connected-configuration local energies, checkpoint I/O.
- `vendor/jaqmc_spring/`: licensed JaQMC runtime subset, upstream commit
  `71f94158f2aca88ded904451f5314f71cdff5666`.
- `tests/`: optimizer equivalence and physics/execution checks.
- `ising_annealing.py`: gradient-free, exact-objective Ising weight-space SA;
  see [its separate experiment protocol](../SA/README.md).

These are portable snapshots with explicit new method adapters. The original
research trees stay frozen locally. The historical Python package name
fqh_spring does not mean every selected optimizer is SPRING.
Its other historical experiment modules are retained for source compatibility;
the supported comparison entry points are the six algorithm/physics wrappers.
The advanced angular-head source is separately indexed under
SPRING/FQHE/training/research_snapshot.

## Install once, from the course/repository root

Use Python 3.12. These pins reproduce the captured project environment.

```bash
python3.12 -m venv .venv-project
source .venv-project/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Final_proj/shared/requirements-cpu.txt
python -m pip install --no-deps ./Final_proj/shared/vendor/jaqmc_spring
```

Per-problem training/run.py locates shared/ itself; no editable install of the
old project is required. For direct module commands from the repository root,
set `PYTHONPATH="$PWD/Final_proj/shared"`. Use the same interpreter throughout.

```bash
JAX_PLATFORMS=cpu python -m pytest -q Final_proj/shared/tests
python Final_proj/tools/validate_reorganization.py
```

CPU examples are deliberately small. GPU runs use the same JAX numerical code,
with a CUDA-compatible JAX environment on an allocated GPU machine. A detected
device or accepted job is not evidence of completed training. Deployment
bundles and credentials are excluded from the Git export.

## Comparison rules

Match Hamiltonian, initial parameters, architecture, readout, batch, sampling,
dtype and evaluation. Record each optimizer's damping, memory, learning rate
and cap; report updates and wall time. VMC-SGD uses the full gradient 2*force;
SR-family directions use force. Equal numerical learning rates alone do not
make a fair comparison.

MinSR here is the finite-damping, memory-free SR control, not bitwise identity
to another paper's cutoff scheme. A full matched convergence study remains
future work. ED is independent evaluation, never supervised training labels.
