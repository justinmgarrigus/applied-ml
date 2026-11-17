# Method review of the slide draft

Reviewed 2026-11-12. Scope: slides6–13,19,22 of `SPRING_FQHE.tex`, against the verified current implementation and original SPRING paper. No deck or scientific-source changes were made by the reviewer.

## Verdict: issues resolved

The algorithmic sequence, positive-force direction convention, real/imaginary treatment, all-layer readout architecture, parameter count, hyperparameters, cap convention, importance factors and implementation dimensions are consistent with the code. A final narrow read of the current `SPRING_FQHE.tex` verified that all three requested corrections below were applied: coefficient interference on slide9, sampled-estimator notation on slides7/13, and indexed covariance on backup22. The optional log-wavefunction wording is also corrected. No unresolved method issue remains within this review's scope.

## Resolved corrections (review history)

1. **Slide9: phase and the fixed Hamiltonian.** The sentence “phase controls interference and signs in Hamiltonian matrix elements” can imply that trainable phase changes the matrix elements of H. The Hamiltonian is fixed. Use “phase controls coefficient interference in Hamiltonian expectation values” or “phase controls the complex coefficient ratios in local energies.” This is a scientific distinction, not an aesthetic change.

2. **Slide13: population identity versus a finite weighted sample.** `S=A^T A` and `f=A^T e` are the sampled metric/force constructed from the batch. The population force is exactly half the physical energy gradient, but a normalized, selected finite-batch covariance is not exactly that gradient. Use hats for sampled objects, or a short note: “A^T A and A^T e estimate the physical metric and half-gradient.” For example `hat f=A^T e ≈ (1/2)∇E`. This aligns slide13 with the finite-batch ratio/covariance-bias caveat already present on slide19. Slide7 can continue to define population f=(1/2)∇E conceptually.

3. **Slide22: write the covariance as an explicit outer product.** With O_i a row, `O_i^* O_i` lacks the transpose needed to produce a P×P matrix. Either use `S=Re sum_i w_i (delta O_i)^dagger delta O_i`, or the unambiguous component form `S_jk=Re sum_i w_i delta O_ij^* delta O_ik`; similarly `f_j=Re sum_i w_i delta O_ij^* delta e_i`. The main-slide `A^T A` already makes the intended matrix clear.

Optional precision: describe O as the Jacobian of the **log wavefunction** on slide13. Its displayed derivative definition is already correct.

## Checks passed

- **Slide6:** Rayleigh quotient, Born distribution, sparse local-energy coefficient ratios, diagonal wall term and ED-only-validation distinction are correct. Finite-sample local energies can have an imaginary mean although the exact expectation is real.
- **Slide7:** SGD has the correct factor2 for f=half-gradient. Damped SR and its dual MinSR expression are equivalent in the sampled system. VMC is correctly distinguished from its optimizer. Dense primal P×P and minimal complex realification2B×2B costs are qualified.
- **Slide8:** `d_k=mu d_prev+A^T(AA^T+lambdaI)^−1(e−Amu d_prev)` is consistent with positive force and `theta←theta−alpha d`. The omitted mean-null lift is harmless to the conceptual exact-arithmetic centered formula and is explained in backup. Mu0 recovers damped MinSR. The Kaczmarz picture is explicitly schematic and no nonlinear convergence guarantee is asserted.
- **Slides9–10:** The heads are log magnitude and phase; `Re psi=e^a cos(phi)`, `Im psi=e^a sin(phi)`. Tanh recursion, summed six-layer readouts, training all W,b,R,c, and depth independent of particle number match `legacy_model.py`. No analytic Laughlin factor is claimed.
- **Slide11:** N7/F18 gives19 orbitals; six width64 layers and all-layer heads give22850 real parameters. Candidate pool128 differs from selected update batch32. Eta.01, lambda.01, mu.9, max_norm radius.03, and paper C=.0009 match saved configurations. Damping is not a Hamiltonian or angular-momentum term.
- **Slide12:** Candidate evolution→scores→selection→local energies→force/metric→SPRING/cap/update matches the actual step. It correctly distinguishes sampler state, raw direction memory and NN parameters.
- **Slide13:** Complex log derivatives include both heads and every layer. Weighted realification and its matrix dimensions are correct, subject to the finite-estimator notation above.
- **Slide19:** q_beta∝|psi|^(2beta) with beta.5; weights proportional to `|psi|^(2(1−beta))/(M*pi_t)`; conditional positive-probability selection with replacement; beta is auxiliary, not physical temperature; no cooling schedule; frozen warmup and finite-batch bias caveats all match the protocol.
- **Slide22:** Signed copies J=[+sqrt(Bw)deltaO;−sqrt(Bw)deltaO] plus JaQMC normalization and complex realification preserve the weighted covariance. Actual solve128×128 at B32 is correct. Rank bound2(B−1)=62 and no extra information from signed duplicates are correct. Positive-force versus paper descent sign conversion is correct.

## Evidence

- `src/fqh_spring/train_exploration.py:52–108`: the complete implemented update, positive half-gradient, scaling and negative parameter update.
- `src/fqh_spring/legacy_model.py:74–98`: heads, summed readouts, complex log derivative.
- `src/fqh_spring/optimizer.py:21–29`: genuine JaQMC settings.
- `src/fqh_spring/geometry_sampling.py:103–117`: signed weighted rows.
- `src/fqh_spring/exploration_sampling.py:36–53`: combined stable importance weights.
- `vendor/jaqmc_spring/src/jaqmc/optimizer/sr/jasr.py:201–205,405–450,477–496,641–664,707–745`: sign, normalization, realification, damped solve, null lift and radius cap.
- [SPRING paper](https://arxiv.org/html/2401.10190v2), Eqs.20,27–34 and Algorithm1; [MinSR paper](https://arxiv.org/abs/2302.01941).
