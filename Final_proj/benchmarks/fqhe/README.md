# Shared FQHE ED reference

`references/ne4` through `ne8` contain complete fixed-N, Lz=0 occupation bases,
sparse pure-V1 Hamiltonians, raising operators, low eigenpairs and integrity
manifests. The implementation is shared/fqh_spring/{physics,references}.py.
`ed/` records the more detailed low-spectrum and angular-momentum analysis.

Input: Ne, flux 3(Ne-1), fixed Lz=0, LLL, normalized V1 pair projectors (V1=1).
Output: basis_bits.npy, hamiltonian.npz, raising_operator.npz, eigenvalues.npy,
eigenvectors.npy, metadata.json and manifest.json. Decode orbital occupations
with `(bits[:, None] >> arange(flux+1)) & 1`. The largest retained sector is
8,512 x 22 for Ne=8. Fermionic signs are included in H.

Create and verify a fresh small reference from the repository root:

```bash
python Final_proj/benchmarks/run_ed.py --system fqhe --ne 4 --output /tmp/fqhe_ed_new
```

The output directory must be new. Generation refuses a sector above 20,000
states; it is not a large-Ne ED launcher. Optional `--v3` changes the model,
so compare only with a matching training Hamiltonian. The frozen database is
pure V1 with analytical ground energy zero. The finite neutral gap is an
excitation gap, not a positive ground energy. ED states never enter training.
