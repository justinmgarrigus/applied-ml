# SPRING for fractional quantum Hall wavefunctions

Physics-audience presentation by PassengerAn, 12 November 2026.

Open **SPRING_FQHE.pdf**. It contains **21 main slides and 2 backup slides** in widescreen 16:9 format. The editable source is **SPRING_FQHE.tex**; **SPRING_FQHE_source.zip** bundles the source and graphics. Equations and diagrams remain vector graphics in the PDF.

## Slide sequence

1. Motivation and route through the talk.
2. Haldane sphere, monopole harmonics, LLL truncation, and Laughlin shift.
3. Canonical Slater occupation representation, antisymmetry, and fixed particle/angular-momentum sector.
4. Haldane pseudopotentials and the normalized pure-V1 interaction.
5. Finite spatial wall and its exact projection into LLL orbital energies.
6. Variational energy loss and local energy computed from coefficient ratios.
7. SGD, SR and MinSR inside the common VMC framework.
8. SPRING's projected correction of direction memory; original-paper evidence and scope.
9. Complex coefficients represented by log magnitude and phase.
10. Six hidden layers with a readout from every layer.
11. Actual network, candidate-pool, update-batch and optimizer settings.
12. The training loop with short ML terminology.
13. Logarithmic derivatives, sampled force and quantum metric.
14. Six-spin Ising implementation and ED validation.
15. FQHE ED database and the retained near-ground-state N=6 example.
16. Original Born-sampling SPRING convergence toward ED in the N=7 wall benchmark.
17. Brief GG-SPRING introduction: candidate pool, derivative-space novelty, full-support selection and importance correction.
18. N=7 geometry-guided sampling; why adding the wall changes the physical target.
19. Exact residual-variance tails and inadequate candidate coverage.
20. Broader proposal plus selection reweighting, with the energy objective fixed.
21. Paired 8k-to-10k continuation results and measured accuracy/cost tradeoffs.
22. Backup: primary references and comparison scope.
23. Backup: complex-state adapter dimensions, metric, signs and parameter count.

## Scientific interpretation

VMC is the estimation framework; SPRING is an optimizer inside it. The original paper's SPRING-versus-MinSR/KFAC result concerns tuned molecular FermiNet calculations. The latest local FQHE comparison keeps SPRING fixed while changing sampling and selection.

The main current test is N=7, flux18, Lz=0, all19LLL orbitals, six64-unit tanh layers and additive readouts from all layers. The projected wall has U=10V1 and x_c=0.88; its ED target energy is 2.82308466308447V1. This target is different from the no-wall zero-energy Laughlin state. No angular-momentum penalty or exact Laughlin factor enters the neural ansatz.

The original Born-pool GG-SPRING baseline already reaches91.28%mean ED fidelity, with mean energy excess0.15162V1 after10kupdates. Slide16 shows all three sampling trajectories and their mean, with the matching ED reference. This demonstrates useful variational accuracy before the broader-pool refinement. It retains the existing geometric row selection; it is not a separate vanilla-SPRING versus GG comparison.

The broader pool has q_beta proportional to |psi|^(2beta), beta=0.5, and improves mean fidelity to94.84%. Reweighting retains the physical energy objective, but self-normalized finite-batch covariance/ratio bias and MCMC errors remain. Three paired learned states are a pilot, not a significance test or asymptotic-convergence demonstration. The earlier N=8 depth-ablation frame and its figures are preserved in support/archive/ and support/figures/ rather than the main slide sequence.

## Rebuild

With TeX Live, Beamer, TikZ, Latin Modern and Poppler installed:

```bash
python3 build.py --render
```

This compiles the bundled figures, checks page count and layout warnings, renders all pages and writes validation.json. It does not train a model or run ED. Recompiling the deck needs only the bundled assets; regenerating numerical figures also needs the original project result files.

The numerical-figure generators are support/generate_slide_results.py, support/generate_results_figures.py, and support/generate_born_baseline_figure.py. The wall/sphere generator is support/build_physics_figures.py. These use NumPy, SciPy and Matplotlib in the existing project environment. Input hashes are retained in support/figures/*inputs.json, born_baseline_evidence.json and physics_figures_metadata.json. The briefs and review records in support/ map statements to source files and papers.

All 23 rendered pages were visually inspected. The finalized deck has no LaTeX overfull-box, underfull-box or package warnings. Saved upstream validation confirms all9latest continuation arms and45exact evaluations; this deck reuses those results without new optimization.
