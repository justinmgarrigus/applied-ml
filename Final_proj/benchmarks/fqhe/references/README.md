# Independent exact references

The N=4,5,6,7,8 folders contain the complete M=0 occupation basis, sparse pure-V1
Hamiltonian, L-plus map, and low eigenpairs (eight for N=4/5, four for N=6–8).
The N=6–8 reference files are unchanged from the retained training comparison;
N=4/5 were added with the ED benchmark suite. `manifest.json` uses only
relative filenames and SHA256 hashes. The unique zero-energy ground vector
is used for post-training fidelity diagnostics, never as training data.

The training module does not import these references. They are small-system
validation artifacts, not a promise that exact diagonalization scales to large
particle numbers. All fluxes follow Nphi=3(N−1); orbital a has m=a−Nphi/2 and
fermion creation operators are ordered by increasing orbital index.

The separate `results/ed_benchmarks/` suite freshly solves these matrices and
saves 20 low eigenpairs (all 18 for N=4), neutral gaps and angular-momentum
diagnostics. It does not overwrite the original references.
