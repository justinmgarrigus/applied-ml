# FQHE exact-diagonalization benchmarks

These results freshly diagonalize the sphere Hamiltonian

\[
H/V_1=\sum_{i<j}P_{R=1}(i,j),\qquad N_\phi=3(N-1),\qquad L_z=0.
\]

The basis is the **complete fixed-N, fixed-Lz Slater basis in one lowest Landau level**. Orbital `a=m+N_phi/2` runs from 0 through `N_phi`; creation operators are ordered by ascending `a`, and canonical fermionic signs are included. Antisymmetry is built into this basis. There is no Landau-level mixing, higher-LL form factor, Coulomb tail, temperature, MCMC, neural approximation, or training in ED. The pair projectors have unit nonzero eigenvalue, so energies below are in the same `V1=1` convention as SPRING training.

| N | Flux | M=0 dimension | Ground E/V1 | Neutral gap/V1 | First excited L |
|---:|---:|---:|---:|---:|---:|
| 4 | 9 | 18 | 0 within roundoff | 0.680604246855 | 4 |
| 5 | 12 | 73 | 0 within roundoff | 0.609317081152 | 4 |
| 6 | 15 | 338 | 0 within roundoff | 0.585779967260 | 5 |
| 7 | 18 | 1,656 | 0 within roundoff | 0.548748754612 | 5 |
| 8 | 21 | 8,512 | 0 within roundoff | 0.531576794705 | 5 |

Every ground state is the unique zero-energy mode in the complete M=0 sector, with L=0 and uniform one-body density to numerical precision. This is the Laughlin parent Hamiltonian at its Laughlin flux. Its ground-state energy is zero **and its neutral excitation gap is positive**. A nonzero neural energy around `0.49 V1` is a variational error, not the ED ground energy or the exact neutral gap.

The neutral gap is the first positive energy minus E0 **at fixed N and flux**. All many-body L values here are integers; a rotationally invariant multiplet has an M=0 representative. The complete M=0 calculation therefore includes the lowest excitation across all L. This is not the charged gap, which needs additional flux sectors and an explicit energy/background convention. The lowest L=0 excitation is a different quantity: N=8 gives `1.056888888541 V1` within the saved spectrum, while the physical first neutral excitation has L=5. A missing L=0 excitation among retained levels is recorded as unknown, never as an absent state.

## Files and checks

- `summary.csv` / `summary.json`: size comparison, solver versions, runtime and source hash. JSON `cases` contains the table rows and `nncomparisons` contains six neural comparisons.
- `neN/eigenpairs.npz`: freshly solved low eigenvalues, eigenvectors and their exact basis bitstrings. Default is 20 low states; N=4 has only 18 states. All Hamiltonians act on the complete sector; N=7/8 use sparse low-eigenpair diagonalization, not a many-body basis truncation.
- `neN/spectrum.csv` / `summary.json`: energies, L labels, full-space residuals, ground occupations and reference hashes. Extra computed levels protect the saved energy-cluster cutoff. Degenerate energy clusters are rotated by L2; an L label is accepted only if its full-space L2 residual passes. No ground-state L projection is imposed on the neural states.
- `checkpoint_comparison.csv` / `ne8_neural_comparison.json`: all six retained N=8 pilot checkpoints, freshly evaluated over all 8,512 configurations against the newly solved ground vector.
- `model_evaluations/`: full frozen-checkpoint reports and normalized wavefunction coefficients, with relative project paths and SHA256 input hashes.

All retained low eigenpairs pass Hermiticity, normalization/orthogonality, eigenvalue residual and angular-momentum checks. The maximum eigenpair residual across N=4–8 is below `1.1e-14 V1`. Ground-state occupation uniformity and the integrated density are checked. For N<=6, the full sparse Hamiltonian and raising operator are rebuilt from the runtime connections and compared with the saved reference matrices. This reconstruction shares the runtime pair-projector construction; the separate analytic two-electron unit test checks unit projector normalization. The independent ladder-construction test additionally checks pair amplitudes and many-body signs without diagonalizing pair L2.

The original `references/ne6`–`ne8` files are unchanged. New portable reference directories `references/ne4` and `ne5` were generated. ED eigenvectors and matrices remain evaluation data; training never reads them.

## What the neural benchmark shows

The selected dense seed-71 endpoint gives:

- Exact energy `0.490844981335 V1`, versus ED ground energy zero within roundoff.
- Ground-state fidelity `0.365698813869` and energy error `0.923375486335` times the neutral gap.
- Residual variance `0.465712164209 V1^2`, or residual norm `0.682431069199 V1`.
- L2 expectation `30.0167563206` and L=0 probability `0.368918007224`.

This endpoint is a pure variational wavefunction with components in several angular-momentum sectors and energy eigenstates. Its energy being below the first exact excitation does not establish ground-state convergence. The exact variational bound `F >= max(0, 1-(E-E0)/gap)` gives only `0.0766245` here; the directly evaluated fidelity is more informative. Longer runs can be assessed by changes in energy error, variance and fidelity against this reference. These pilot data do not establish a CDW or a higher-LL competing phase.

## Reproduce

From the clean `Haldane_FQHE` project directory, after the documented environment installation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m fqh_spring.ed_benchmark --root . --sizes 4 5 6 7 8 \
  --eigenpairs 20 --output results/ed_benchmarks
python -m pytest tests/test_ed_benchmark.py tests/test_ed_independent.py
```

The default run regenerates ED benchmark outputs and checkpoint evaluations. Existing reference directories are integrity-checked and reused rather than overwritten; the eigensystem is nevertheless solved afresh from their Hamiltonians. `--skip-model-comparison` performs ED alone. The CLI intentionally rejects N>8 to avoid accidentally launching a large local ED calculation. Recorded runtimes describe this machine and are not a benchmark against neural training.
