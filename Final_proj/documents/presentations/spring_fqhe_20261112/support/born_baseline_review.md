# Revised slide 16: original Born-pool SPRING benchmark

Status: **PASS** for the scientific content and numerical provenance of the replacement frame, checked 12 November 2026. No training, diagonalization or source-result changes were performed for this review.

The new title, “Born-sampled SPRING already approaches the ED state,” and its statement that SPRING learns a good variational approximation are supported by the retained three-seed results. The slide reports an approximation, with visible nonzero energy error and variance, rather than exact convergence.

## Verified method and target

- The method is the existing **GG-SPRING with original Born candidate pool, beta=1**, including static geometric row selection and its inverse-selection weights. It is not a separate vanilla-SPRING benchmark. The introduction and footer label this accurately.
- The physical target is N=7, flux18, all19LLL orbitals, Lz=0, pureV1 plus the projected U=10V1 wall at x_c=0.88. The same Hamiltonian and sector are used for the subsequent broader-pool comparison.
- ED E0=2.82308466308447V1 is the minimum in this fixed sector. The footer correctly avoids claiming a global minimum across every Lz sector.

## Independently checked endpoint values

All values below are arithmetic means over seeds71,137,211, with no best-seed selection:

| Quantity | Original Born-pool result |
|---|---:|
| E/V1 | 2.97470637100747 |
| (E-E0)/V1 | 0.15162170792300 |
| ED squared-overlap fidelity | 0.91276901517049 |
| Residual variance/V1² | 0.37714393922427 |

The slide's rounded values 2.97471, 0.15162, 91.28%, and 0.37714 match. Final fidelity ranges from88.88%to93.52% across the three seeds. The stated broader-pool mean94.84% matches the same-target static broader-pool result94.8448057%.

## Full-trajectory figure provenance

The companion figure joins the wall-trained 0–8k history and its original-pool 8k–10k continuation. This review independently checked all14 input hashes recorded in `figures/born_baseline_evidence.json` and all three 8k joins. The paired evaluations have identical checkpoint hashes, coefficient-file hashes, reference manifests, orbital profiles, energies, energy errors, fidelities and exact residual variances. Recomputed final means agree within1e-14.

The figure includes all three sampling trajectories and their mean; its 13points per seed are not confidence intervals. Its initial mean energy14.75476456V1 decreases to2.97470637V1, while mean fidelity rises from0.0158%to91.28%. The resulting98.73%reduction in initial energy excess indicates substantial optimization progress. Remaining positive energy error and variance preclude an exact-ground-state claim.

## Presentation consistency

README slide16 and scientific-interpretation text have been updated to this baseline. The removed N=8 depth frame is retained under support/archive, and original result files/figures remain intact. The root build should include the new baseline evidence receipt in its input validation and packaging; page count remains20main+2backup. Layout/render validation is performed separately by the parent build process.
