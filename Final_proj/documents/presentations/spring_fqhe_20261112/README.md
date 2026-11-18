# Symmetry-guided SPRING on the Haldane sphere

Updated 16 November 2026. Open **SPRING_FQHE.pdf**: 20 main slides and 5 technical/reference slides in 16:9 format. Editable source and vector graphics are supplied in **SPRING_FQHE.tex** and **SPRING_FQHE_source.zip**.

The current deck replaces spatial-confinement content with a conserved-operator extension of SPRING. Previous presentation files are preserved locally in support/archive/pre_symmetry_update_20261116/ and excluded from the current package.

## Method and ordering condition

The ideal single-state target is exp[-tau_H(H+lambda C)], with lambda=tau_C/tau_H. Its SPRING residual changes from e_H to e_H+lambda e_C. Numerical damping is written delta to distinguish it from the conserved-operator weight.

Commuting Hermitian H and C admit a joint eigenbasis. Ordering within each fixed-C sector is unchanged. Global ordering additionally requires E_b-E_a+lambda(c_b-c_a)>0 for every ordered pair. If C is positive semidefinite, annihilates an original ground state, and lambda is nonnegative, that ground state remains a ground state. Counterexamples to the stronger global-ordering claim are included.

Our dimensionless test uses h=H/V1, K=L²/hbar² and h+kappa K with kappa=0.1. ED verifies an L=0 ground for this benchmark. The coherent penalty explicitly favors L=0; there is no hard projection or ED coefficient input during training.

## Physical model and implemented network

All current numerical comparisons use Ne=7, flux18, 19 strict-LLL orbitals, fixed Lz=0 and

    H/V1 = sum(P1 + 0.8 P3 + 0.6 P5).

The ED ground energy is **6.244768260215 V1**. This mixed-pseudopotential ground is different from the zero-energy pure-V1 Laughlin state. The quoted full fidelity targets the mixed-H ED ground, not the Laughlin reference. Filling and geometry alone do not establish a topological phase.

The current network has a shared six-layer width64 tanh trunk with additive readouts from every layer and 42 private complex relative heads, each with two width16 tanh layers. Learned coherent mixing gives one physical coefficient. All **294,464 real parameters** are optimized. Heads are learned angular features, not exact sector projectors.

The actual differentiated sampled loss and update are

    J = E_h + gamma A_heads + kappa <K>, gamma=1,
    (G_aug + delta I)d_k = 0.5 grad(J) + delta mu d_(k-1),
    theta_(k+1) = theta_k - alpha_k d_k.

The ordinary augmented projective metric is retained; no added Gauss–Newton residual metric is claimed. This implemented equation is distinguished from the ideal single-state residual explanation.

Current settings: batch128 from a uniform valid-sector pool512, geometry/residual selection with inverse-selection weights, eight sampled H connections per root, independent128 Lplus output rows, damping0.1, nominal eta0.001, memory0.9, parameter radius0.015, augmented-state radius0.01, CG ceiling1024/tolerance1e-6, and held-out checks every50 attempts.

## Comparison and scope

| Saved result | Physical E/V1 | Full mixed-H ED fidelity | P(L=0) |
|---|---:|---:|---:|
| No coherent L2 term, historical16k | 7.199291 | 2.066% | 2.451% |
| Matched kappa0 after +1k | 7.376361 | 1.582% | 1.780% |
| Matched kappa0.1 after +1k | 7.464363 | 3.094% | 4.200% |
| Latest kappa0.1,54k | 6.404269 | 77.817% | 82.411% |

Both matched arms retain the same angular-head auxiliary loss. “No coherent L2 penalty” does not mean energy-only original SPRING.

The matched1k test improves sector population and fidelity but has higher physical energy with kappa0.1. The longer updated workflow gives a much better same-H saved state:83.3% less energy excess than the historical16k source. More training, larger batch/pool, guard/anchor changes and a higher CG ceiling prevent attributing the entire gain to L2 alone.

The latest unchanged10k reaches absolute54000, lowering error29.6% and raising full fidelity70.18% to77.82%. All21 saved states and10000 attempts passed independent checks. Remaining physical energy error is0.159501V1; variance0.136171V1²; conditional L=0 fidelity94.426%. About72.36% of remaining energy error comes from outside L=0.

## Slide sequence

1. Motivation and scope.
2. Haldane sphere and rotational symmetry.
3. Occupation representation, antisymmetry and fixed Lz.
4. Same mixed-pseudopotential Hamiltonian and correct ED target.
5. Physical energy and coefficient-ratio evaluation.
6. SGD, SR and MinSR.
7. Original SPRING direction memory.
8. Conserved-operator imaginary-time target.
9. Modified SPRING residual and primal equation.
10. Positive coherent Lplus evaluation of L2.
11. Actual loss and augmented metric.
12. Shared network, all-layer readouts and complex relative heads.
13. Current sampling and solver parameters.
14. Ising gradient/ED implementation check.
15. Matched short kappa0 versus kappa0.1 test.
16. Historical versus latest same-H state.
17. Long retained batch128 trajectory.
18. Remaining energy and sector errors.
19. Commuting-operator theorem and conditions.
20. Reusable workflow and next equal-budget ablation.
21–25. Tangent derivation, ordering/counterexamples, head loss/metric, provenance/limits and references.

## Rebuild and evidence

From this folder:

    python3 build.py --render

The build checks25pages, verifies available saved-result hashes, renders each page and packages current source/used graphics. It performs no training, ED or new neural-state evaluation. The portable package compiles from bundled vector figures without the external result tree; missing external provenance files are explicitly reported.

Numerical extraction: support/extract_constraint_comparison_20261116.py. Its manifest verifies269 original artifacts. Figure generators: support/generate_constraint_figures_20261116.py and support/build_symmetry_sphere_20261116.py. New theory/comparison reviews and metadata are included. A PDF-hash-bound visual-inspection receipt records review of the final25pages.

Primary references: [SPRING](https://arxiv.org/abs/2401.10190), [FQH/FNN](https://arxiv.org/abs/2412.00618), and [symmetry penalty methods](https://arxiv.org/abs/2010.13951). Penalty Hamiltonians are established methodology; this project implements and tests their coupling to SPRING.
