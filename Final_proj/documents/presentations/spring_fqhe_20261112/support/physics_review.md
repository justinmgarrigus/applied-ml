# Physics review of SPRING_FQHE.tex

Reviewed 12 November 2026. Scope: slides 2–5, 17 and the complex-metric expressions on slide 22. Final source was re-read after the parent's notation fixes. No deck edits were made by this reviewer.

**Final verdict: PASS. No outstanding physics or covariance-formula bug was found in the reviewed slides.** The wall operator, covariance indices and finite-batch-gradient distinction are corrected. The newly added monopole-harmonic and spinor conventions agree with the code's orbital ordering and wall profile.

## Corrections verified as resolved

1. **Slide 22: covariance matrix indices are now explicit.** The final source uses indexed batch estimators, resolving the original row/outer-product ambiguity:

   \[
   S_{\alpha\beta}=\Re\sum_i w_i\,\delta O_{i\alpha}^*\delta O_{i\beta},
   \qquad f_\alpha=\Re\sum_iw_i\,\delta O_{i\alpha}^*\delta e_i.
   \]

   Its actual source notation uses j,k instead of α,β and hats on the estimators. The force conjugation and sign are correct. The change clarifies dimensions without changing the computation.

2. **Slide 5: the wall operator is now distinguished from its scalar height.** The final source writes `epsilon_m=<Qm|widehat U(x)|Qm>`. The subsequent beta-distribution integral remains correct.

3. **Slide 13: the empirical force now has the estimator hat.** The final source explicitly states `hat S=A^T A` and `hat f=A^T e approx 1/2 grad E`, avoiding the earlier exact-population-gradient reading of a finite weighted batch.

## Physics checked without finding an error

- Slide 2: N_phi=2Q, radius sqrt(Q) l_B, l=Q, m=-Q…Q, 2Q+1 orbitals, and Laughlin flux shift N_phi=3(N_e−1). The finite-size filling qualification is correct.
- Slide 3: ascending-m canonical Slater ordering, fixed N and Lz occupation constraints, antisymmetry in the basis, and NN output as one coefficient of an entire configuration. N7/F18/Lz0 dimension1,656 is correct. Lz0 does not imply L²0.
- Slide 4: odd relative pair angular momentum for polarized fermions; pair total L=2Q−R; normalized pure-V1 projector Hamiltonian; no-wall Laughlin zero-energy target; no full-Coulomb claim. The implementation constructs projectors by diagonalizing two-particle L², as the footer states.
- Slide 5: k=F−a with a=m+Q, so epsilon_a=U integral_xc^1 x^k(1−x)^(F−k)/B(k+1,F−k+1) dx. Large m means north-localized orbitals. All19orbitals remain. The wall penalizes the southern12% of sphere area, preserves Lz, breaks full rotation symmetry, and shifts the ground energy. No L² penalty is present.
- Slide 17: the two panels explicitly distinguish Hamiltonians and budgets. The wall-target E0=2.823084663V1, mean8k error .154239V1, mean fidelity90.8512%, and wall/no-wall ground squared overlap2.75% agree with current saved results. The caution against interpreting wall fidelity as optimizer improvement for the original Laughlin problem is necessary and correct.
- Slide 22: the signed adapter with ±sqrt(Bw) deltaO has2Bcomplexrows. JaQMC division by sqrt(2B), followed by real/imaginary stacking, preserves the real weighted metric with4Brealrows. Thus B32 produces a128×128Gram matrix. Rank≤2(B−1)=62 is correct; signed copies add no information. The code/paper direction-sign mapping is consistent with the deck's definitions.
- The final slide2 addition `varphi_m proportional to u^(Q+m) v^(Q−m)` and the backup22 gauge `u=cos(vartheta/2) exp(+i phi/2)`, `v=sin(vartheta/2) exp(−i phi/2)` agree: the orbital phase is exp(i m phi), its density is proportional to (1−x)^(Q+m) x^(Q−m), and k=Q−m=F−a. Hence large positive m is north-localized and the wall beta-profile formula has the correct orientation.

## Two small notation cautions

- The sphere graphic uses θ_c for polar angle, while slide5 uses vartheta for polar angle to reserve theta for network parameters. This is a visual symbol mismatch rather than a wrong formula. Label the polar-angle relation consistently or state that both denote the polar angle.
- The final slide13 hats/approximation and slide19 finite-batch-bias caveat now make the population/estimator distinction explicit.

## Sources actually checked

Implementation: `physics.py:1–9,40–58,61–100,102–145,201–224`; `north_droplet.py:1–7,33–59,72–91`; `local_energy.py:77–115`; `geometry_sampling.py:103–117`; JaQMC `jasr.py:389–450,647–662`.

Saved arrays: `results/ne7_gg_edge_20261111/references/wall10/{metadata.json,eigenvalues.npy,eigenvectors.npy,epsilon_a.npy}`, compared with `references/ne7/eigenvectors.npy`. Direct array evaluation gives wall E0=2.82308466308447V1, gap=.1857586203776065V1, and wall/no-wall squared ground-state overlap=.027462567925589612. `results/ne7_gg_explore_20261111/validation.json` is passed and contains the same wall profile hash.

Primary literature verified online:

- [Haldane, PRL51,605 (1983)](https://doi.org/10.1103/PhysRevLett.51.605): spherical quantum Hall construction.
- [Laughlin, PRL50,1395 (1983)](https://doi.org/10.1103/PhysRevLett.50.1395): variational Laughlin state.
- [Haldane and Rezayi, PRL54,237 (1985)](https://doi.org/10.1103/PhysRevLett.54.237): finite-size spherical calculations.
- [Teng, Dai and Fu, arXiv:2412.00618v2; PRB111,205117 (2025)](https://arxiv.org/abs/2412.00618): disk real-space attention FNN with Landau-level mixing, correctly distinguished from our strictly LLL occupation MLP.
