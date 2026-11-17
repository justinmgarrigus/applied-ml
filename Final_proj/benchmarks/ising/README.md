# Shared Ising ED reference

`ed_solver.py` constructs the periodic Pauli Hamiltonian independently of JAX,
diagonalizes both spin-flip parity sectors, and validates complete eigenpairs,
orthonormality and finite-ring analytical energies. `ed/` contains the retained
six instances (N=4,6,8,10 at h=J=1, plus N=6 at h/J=0.5 and 2).

Input: N, J, h, periodic boundaries; configuration rows are N spins in {-1,+1}.
Output per case: sparse H, basis bits/spins, complete eigenvalues and parity
labels, low eigenvectors, ground subspace and diagnostics. `ed/manifest.json`
binds the frozen data. These are references, not supervised labels.

Generate a fresh N=6, J=h=1 reference from the repository root:

```bash
python Final_proj/benchmarks/run_ed.py --system ising --ne 6 --output /tmp/ising_ed_new
```

The output directory must not exist. The solver deliberately limits dense ED
to 3 <= N <= 10. Install the common environment described in
[shared](../../shared/README.md). The copied `ed/README.md` records historical
commands; the command above is the current layout's entry point.
