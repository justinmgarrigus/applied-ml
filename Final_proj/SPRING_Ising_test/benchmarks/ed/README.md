# Exact periodic Ising benchmarks

The Hamiltonian is `H = -J sum_i Z_i Z_(i+1 mod N) - h sum_i X_i; Pauli eigenvalues +/-1`. Z has eigenvalues +/-1, not +/-1/2.
Integer configurations 0..2**N-1; site i is bit i (least significant first); bit 0 has Z_i=-1 and bit 1 has Z_i=+1. X_i flips bit i.

| N | J | h | Dimension | Ground energy | Full gap | Even-parity gap |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1 | 16 | -5.226251859506 | 0.397824734759 | 3.061467458921 |
| 6 | 1 | 1 | 64 | -7.727406610313 | 0.263304995175 | 2.070552360820 |
| 8 | 1 | 1 | 256 | -10.251661790966 | 0.196982806714 | 1.560722576129 |
| 10 | 1 | 1 | 1024 | -12.784906442999 | 0.157403413649 | 1.251475720322 |
| 6 | 1 | 0.5 | 64 | -6.384694563604 | 0.006892444970 | 2.478627349855 |
| 6 | 1 | 2 | 64 | -12.769389127207 | 2.013784889940 | 4.957254699710 |

These are complete finite-ring spectra, obtained by diagonalizing both exact global-spin-flip
parity blocks and merging their eigenvalues. The full gap is the first energy above the
entire ground eigenspace. The even-parity gap is the first excitation within parity +1.
The existing zero-bias RBM enforces parity +1 and cannot access the first odd-parity state.
All listed h>0 ground states are unique and even.

The retained N=6 SPRING checkpoint was reevaluated independently with NumPy over all 64 configurations. Its energy is -7.727390852047, energy error 1.57582658e-05, squared ground-state fidelity 0.999998569116, and variance 0.000205363749. This is a reevaluation of the existing 250-update run, not a new optimization. Its numerical even-sector gap bound is F >= 0.999992389342.

At criticality J=h>0 the finite-ring Jordan-Wigner sums give
`E_even=-2J*csc(pi/(2N))` and `E_odd=-2J*cot(pi/(2N))`.
Consequently `E0=-2J*csc(pi/(2N))`, `gap=2J*tan(pi/(4N))`, and
`even_gap=8J*sin(pi/(2N))`. These analytical values independently check the ED results;
they are not thermodynamic-limit approximations. Ground-state energy decomposes into
`-J*N*<ZZ> - h*N*<X>`, another saved consistency check.

Regenerate from the `spring_ising_test` directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python ed_benchmark.py
.venv/bin/python -m pytest -q tests/test_ed_benchmark.py
```

The ED program itself requires only NumPy and SciPy and never imports JAX or the optimizer.
It uses no random numbers. Different BLAS/LAPACK implementations can rotate eigenvectors
inside a degenerate same-parity space; energy, projectors, and observables are the invariant references.

`summary.json` records configuration, versions, all diagnostics, and the checkpoint comparison;
`summary.csv` is a flat table. Each case directory contains sparse `hamiltonian.npz`,
complete `basis_bits.npy`, `basis_spins.npy`, `eigenvalues.npy`, `eigenstate_parities.npy`,
the first 8 `low_eigenvectors.npy` (columns, or more if required to include the complete
ground eigenspace), `ground_subspace.npy`, and `diagnostics.json`.
The normalized saved neural state is `retained_spring_n6_wavefunction.npy` when N=6 is included.
`manifest.json` gives SHA256 hashes of all benchmark artifacts except itself.

Checks cover Hermiticity, all eigenpair residuals, complete orthonormality,
parity, trace, observable energy reconstruction, and the critical finite-N formulas.
These outputs are benchmarks only; no ED labels enter the SPRING force, model initialization,
or Born-sampling distribution. No new neural training is performed.
