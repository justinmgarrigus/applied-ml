# Frozen angular-momentum-aware SPRING research source

This source snapshot preserves the later FQHE work alongside the portable
energy-only comparison. It is **not** the implementation selected by the
parent `run.py`, and it is not a new training result. It contains the latest
stopped driver's 22 hash-pinned sources, their local import dependencies,
the original portable package and licensed vendor subset. SOURCE_MANIFEST.json
records which files were explicitly pinned by the stopped run. Earlier solver
classes are retained because the final driver inherits them; their presence
does not activate an earlier failed solver.

## Physics, model and iteration

The stopped run has Ne=12, flux=33, 34 LLL orbitals, fixed Lz=0, pure normalized
V1 projectors, no V3/V5 and no edge wall. The neural model has six shared
64-wide tanh layers with all-layer amplitude/phase readouts, plus 132 private
two-layer, width-16 angular heads (877,454 real parameters). It does not embed
an ED target or a Laughlin factor.

Unlike the portable baseline, the sampled objective includes the head
auxiliary and the positive coherent angular term:

```math
J=E_H+A_{\mathrm{heads}}+\kappa\langle L^2/\hbar^2\rangle,
\quad \kappa=10^{-3},\qquad
(G_{\mathrm{aug}}+\delta I)\phi
=\tfrac12\nabla\widehat J+\delta\mu\phi_{\mathrm{prev}}.
```

This is a sampled SPRING solve with an augmented metric, not plain SGD and
not an exact total-L projector. At Lz=0, the coherent angular term is
`||L+ psi||^2 / ||psi||^2`; it depends on connected amplitudes and cannot be
assigned as an L label of an individual occupation bitstring. The final leg
uses the original plain CG recurrence with ceiling 65,536, tolerance 1e-6,
damping 0.1 and memory 0.9. No GMRES is active. Batch=256, pool=1024 and the
separate L+ output sample has 256 rows. Nominal eta=0.001 and parameter/state
step caps=0.015/0.01; applied learning rates are often smaller.

For pure V1, the Laughlin state has both E_H=0 and L=0, so a positive coherent
L² term preserves that exact physical minimizer. More generally, `[H,C]=0`
only supplies simultaneous eigenvectors: adding lambda*C can reorder energies.
Do not claim it preserves arbitrary H ground states or all energy ordering.
The head auxiliary introduces an additional approximation/tradeoff of its own.

## Reproduction boundary

`historical_ne12_config.json` is copied without rewriting its original paths,
deadline, source hashes or parameter metadata. It records history; it is not a
fresh portable configuration. The 178 MB full-state checkpoint, earlier
checkpoints and raw sampling/continuation ledgers remain in the original local
archive. Their checkpoint SHA is recorded in SOURCE_MANIFEST.json. The new
portable trainer has a different model/state format and must not load it.

Use this directory to inspect the algorithm and its exact source. A validated
continuation uses the frozen original archive, restores all six RNGs, memory,
guards and caches, verifies source hashes, and requires a new explicit training
request. Merely copying these scripts to another machine does not transfer the
raw inputs. There is no automatic long-run/resume command in this migration.

See [results](../../results/README.md) for the stopped endpoint. Its physical
MC energy is 3.63448 +/- 0.02660 V1, above analytical E0=0; this is not a
converged large-system Laughlin solution.
